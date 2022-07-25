# %% [markdown]
# ## Download Data

# %%
!pip install torchdata
%matplotlib inline

# %%
from torchtext.datasets import WikiText2
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from fake_bert.models.dextra_emb import DExTraEmb

# %%
import math
import argparse
DEFAULT_WIDTH_MULTIPLIER = 2.0
DEFAULT_MIN_DEXTRA_LAYERS = 4
DEFAULT_MAX_DEXTRA_LAYERS = 8
DEFAULT_BASE_GROUPS = 16
MIN_ELEMENTS_PER_GROUP = 32
DEFAULT_FFN_RED_FACTOR = 4
DEFAULT_DROPOUT = 0.1
DEFAULT_STD_DROPOUT = 0.1
ADAPTIVE_SCALE_FACTOR = 2


def base_architecture(args):
    # DeLighT Embedding layer
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.delight_emb_map_dim = getattr(args, "delight_emb_map_dim", 128)
    args.delight_emb_out_dim = getattr(args, "delight_emb_out_dim", 128)
    # compute the max groups in GLT
    assert args.delight_emb_out_dim % MIN_ELEMENTS_PER_GROUP == 0, 'remainder({}, {}) should be equal to 0'.format(
        args.delight_emb_out_dim, MIN_ELEMENTS_PER_GROUP)
    max_groups = 2 ** math.ceil(math.log(args.delight_emb_out_dim // MIN_ELEMENTS_PER_GROUP, 2))

    args.delight_emb_max_groups = getattr(args, "delight_emb_max_groups", max_groups)
    args.delight_emb_dropout = getattr(args, "delight_emb_dropout", DEFAULT_DROPOUT)
    args.delight_emb_depth = getattr(args, "delight_emb_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_emb_width_mult = getattr(args, "delight_emb_width_mult", DEFAULT_WIDTH_MULTIPLIER)

    # Encoder arguments in DeLighT
    args.delight_enc_scaling = getattr(args, "delight_enc_scaling", 'block')
    args.delight_enc_layers = getattr(args, "delight_enc_layers", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_enc_min_depth = getattr(args, "delight_enc_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_enc_max_depth = getattr(args, "delight_enc_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_enc_width_mult = getattr(args, "delight_enc_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    args.delight_enc_ffn_red = getattr(args, "delight_enc_ffn_red", DEFAULT_FFN_RED_FACTOR)
    args.delight_enc_max_groups = getattr(args, "delight_enc_max_groups", max_groups)

    # Decoder arguments in DeLighT
    args.delight_dec_scaling = getattr(args, "delight_dec_scaling", 'block')
    args.delight_dec_layers = getattr(args, "delight_dec_layers", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_dec_min_depth = getattr(args, "delight_dec_min_depth", DEFAULT_MIN_DEXTRA_LAYERS)
    args.delight_dec_max_depth = getattr(args, "delight_dec_max_depth", DEFAULT_MAX_DEXTRA_LAYERS)
    args.delight_dec_width_mult = getattr(args, "delight_dec_width_mult", DEFAULT_WIDTH_MULTIPLIER)
    args.delight_dec_ffn_red = getattr(args, "delight_dec_ffn_red", DEFAULT_FFN_RED_FACTOR)
    args.delight_dec_max_groups = getattr(args, "delight_dec_max_groups", max_groups)

    ## Others
    args.no_glt_shuffle = getattr(args, "no_glt_shuffle", False)
    args.glt_shuffle = not args.no_glt_shuffle
    args.define_iclr = getattr(args, "define_iclr", False)
    args.delight_dropout = getattr(args, "delight_dropout", DEFAULT_DROPOUT)

    # normalization and activation layers
    args.norm_type = getattr(args, "norm_type", 'ln')
    args.act_type = getattr(args, "act_type", 'swish')

    # ADAPTIVE INPUT AND OUTPUT PARAMS
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.adaptive_softmax_factor = getattr(args, 'adaptive_softmax_factor', 4)
    args.tie_adaptive_weights = getattr(args, 'tie_adaptive_weights', False)
    args.tie_adaptive_proj = getattr(args, 'tie_adaptive_proj', False)

    # Print  stats
    args.print_stats = getattr(args, "print_stats", False)
    args.src_len_ps = getattr(args, "src_len_ps", 20)
    args.tgt_len_ps = getattr(args, "tgt_len_ps", 20)

    # DROPOUTS
    args.attention_dropout = getattr(args, "attention_dropout", DEFAULT_DROPOUT)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.dropout = getattr(args, "dropout", DEFAULT_DROPOUT)
    args.delight_dropout = getattr(args, "delight_dropout", 0.0)
    args.pe_dropout = getattr(args, "pe_dropout", DEFAULT_DROPOUT)
    args.ffn_dropout = getattr(args, "ffn_dropout", DEFAULT_DROPOUT)

    # Other parameters
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)

    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)

args = argparse.Namespace()
base_architecture(args)

# %%


train_iter = WikiText2(split='train')
tokenizer = get_tokenizer('basic_english')
vocab = build_vocab_from_iterator(map(tokenizer, train_iter), specials=['<unk>'])
vocab.set_default_index(vocab['<unk>']) 

def data_process(raw_text_iter: dataset.IterableDataset) -> Tensor:
    """Converts raw text into a flat Tensor."""
    data = [torch.tensor(vocab(tokenizer(item)), dtype=torch.long) for item in raw_text_iter]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))

# train_iter was "consumed" by the process of building the vocab,
# so we have to create it again
train_iter, val_iter, test_iter = WikiText2()
train_data = data_process(train_iter)
val_data = data_process(val_iter)
test_data = data_process(test_iter)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batchify(data: Tensor, bsz: int) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    seq_len = data.size(0) // bsz
    data = data[:seq_len * bsz]
    data = data.view(bsz, seq_len).t().contiguous()
    return data.to(device)

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape [seq_len, batch_size]
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)

# %%
bptt = 35
def get_batch(source: Tensor, i: int) -> Tuple[Tensor, Tensor]:
    """
    Args:
        source: Tensor, shape [full_seq_len, batch_size]
        i: int

    Returns:
        tuple (data, target), where data has shape [seq_len, batch_size] and
        target has shape [seq_len * batch_size]
    """
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].reshape(-1)
    return data, target

# %% [markdown]
# # Model Transformers

# %% [markdown]
# ## PositionalEncoding

# %%
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# %% [markdown]
# # Transformers model

# %% [markdown]
# - Add Dextral Embedd instate encoder 
# - Ke thua transformers encoder de chinnh sua phan foward

# %%


# %%
from fake_bert.models.nn_functions import get_embedding_layer

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        # -------------Add deextra layer-------------
        args.delight_emb_map_dim = d_model
        args.delight_emb_out_dim = d_model
        map_layer = get_embedding_layer(num_embeddings=ntoken,
                embedding_dim=args.delight_emb_map_dim,
                padding_idx=None)
        self.encoder =  DExTraEmb(args, map_layer=map_layer)
        self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len]

        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

# %%


# %%


# %%


# %%
ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2  # number of heads in nn.MultiheadAttention
dropout = 0.2  # dropout probability
model = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout).to(device)

# %%
import copy
import time

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

def train(model: nn.Module) -> None:
    model.train()  # turn on train mode
    total_loss = 0.
    log_interval = 200
    start_time = time.time()
    src_mask = generate_square_subsequent_mask(bptt).to(device)

    num_batches = len(train_data) // bptt
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        batch_size = data.size(0)
        if batch_size != bptt:  # only on last batch
            src_mask = src_mask[:batch_size, :batch_size]
        output = model(data, src_mask)
        loss = criterion(output.view(-1, ntokens), targets)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()

        total_loss += loss.item()
        if batch % log_interval == 0 and batch > 0:
            lr = scheduler.get_last_lr()[0]
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval
            ppl = math.exp(cur_loss)
            print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
            total_loss = 0
            start_time = time.time()

def evaluate(model: nn.Module, eval_data: Tensor) -> float:
    model.eval()  # turn on evaluation mode
    total_loss = 0.
    src_mask = generate_square_subsequent_mask(bptt).to(device)
    with torch.no_grad():
        for i in range(0, eval_data.size(0) - 1, bptt):
            data, targets = get_batch(eval_data, i)
            batch_size = data.size(0)
            if batch_size != bptt:
                src_mask = src_mask[:batch_size, :batch_size]
            output = model(data, src_mask)
            output_flat = output.view(-1, ntokens)
            total_loss += batch_size * criterion(output_flat, targets).item()
    return total_loss / (len(eval_data) - 1)

# %%
best_val_loss = float('inf')
epochs = 3
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train(model)
    val_loss = evaluate(model, val_data)
    val_ppl = math.exp(val_loss)
    elapsed = time.time() - epoch_start_time
    print('-' * 89)
    print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
          f'valid loss {val_loss:5.2f} | valid ppl {val_ppl:8.2f}')
    print('-' * 89)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = copy.deepcopy(model)

    scheduler.step()

# %%



