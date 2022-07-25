import torch
import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim


class GroupLinear(nn.Module):
    '''
        This class implements the Grouped Linear Transform
        This is based on the Pyramidal recurrent unit paper:
            https://arxiv.org/abs/1808.09029
    '''

    def __init__(self, in_features: int, out_features: int, n_groups: int = 4, dropout: float = 0.0):
        '''
        :param in_features: number of input features
        :param out_features: number of output features
        :param n_groups: number of groups in GLT
        :param use_bias: use bias or not
        :param use_shuffle: shuffle features between different groups
        :param norm_type: Normalization type (e.g. LayerNorm)
        :param dropout: Dropout value (default is 0.0)
        :param act_type: Activation type (e.g., Gelu or ReLU)
        '''
        super(GroupLinear, self).__init__()

        if in_features % n_groups != 0:
            err_msg = "Input dimensions ({}) must be divisible by n_groups ({})".format(in_features, n_groups)
            print_error_message(err_msg)
        if out_features % n_groups != 0:
            err_msg = "Output dimensions ({}) must be divisible by n_groups ({})".format(out_features, n_groups)
            print_error_message(err_msg)

        # warning_message = 'Please install custom cuda installation for faster training and inference'

        in_groups = in_features // n_groups
        out_groups = out_features // n_groups

        self.weights = nn.Parameter(torch.Tensor(n_groups, in_groups, out_groups))
        self.bias = None

        self.use_dropout = False
        self.drop_p = dropout
        if dropout > 0:
            self.drop_layer = nn.Dropout(p=dropout)
            self.use_dropout = True

        self.n_groups = n_groups

        self.in_features = in_features
        self.out_features = out_features
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights.data)

    def process_input_bmm(self, x):
        '''
        N --> Input dimension
        M --> Output dimension
        g --> groups
        G --> gates
        :param x: Input of dimension B x N
        :return: Output of dimension B x M
        '''
        bsz = x.size(0)
        # [B x N] --> [B x g  x N/g]
        x = x.contiguous().view(bsz, self.n_groups, -1)
        # [B x g x N/g] --> [g x B  x N/g]
        x = x.transpose(0, 1)  # transpose so that group is first

        # [g x B  x N/g] x [g x N/g x M/g] --> [g x B x M/g]
        x = torch.bmm(x, self.weights)  # multiply with Weights

        # [g x B x M/g] --> [B x g x M/g]
        x = x.transpose(0, 1)  # transpose so that batch is first

        return x

    def forward(self, x):
        '''
        :param x: Input of shape [T x B x N] (should work with [B x T x N]
        :return:
        '''
        T, B, N = x.size()
        x = x.contiguous().view(B * T, -1)
        x = self.process_input_bmm(x)
        x = x.contiguous().view(T, B, -1)

        # dropout
        if self.use_dropout:
            x = self.drop_layer(x)
        return x