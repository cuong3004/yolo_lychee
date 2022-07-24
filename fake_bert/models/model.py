import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import torch.nn


class GroupLinear(nn.Module):
    """Some Information about GroupLinear"""
    def __init__(self):
        super(GroupLinear, self).__init__()
        group_layer = nn.ModuleList()

        layer_config = self.dextra_config(in_features=self.in_proj_features,
                                              out_features=self.out_features,
                                              max_features=self.max_features,
                                              n_layers=self.num_glt_layers,
                                              max_groups=self.max_glt_groups
                                              )
                                            
    @staticmethod
    def dextra_config(in_features, out_features, max_features, n_layers, max_groups):
        
        mid_point = int(math.cell(n_layers / 2.0))

        groups_per_layer = [min(2**(i+1), max_groups) for i in range(mid_point)]

        output_sizes = nn.Gro

    def forward(self, x):

        return x


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

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class MyModule(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, ntoken=100, d_model=256, nhead=1, d_hid=150, dropout=0.2, nlayers=2):
        super(MyModule, self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, ntoken)


    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer_encoder(x)
        x = self.decoder(x)

        return x



    
if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    from torch.utils.data import DataLoader
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from fake_bert.datasets.dataset import RandomGenerator
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    dataset = RandomGenerator()
    dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
    data_sample = next(iter(dataloader))

    model = MyModule()
    writer.add_graph(model, data_sample[0])
    writer.close()

    

    