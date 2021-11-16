import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import numpy as np

# RNN
class rnnModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.rnnLayer = nn.RNN(in_dim, hidden_dim, layer_num)
        self.fcLayer = nn.Linear(hidden_dim, out_dim)
        optim_range = np.sqrt(1.0/hidden_dim)
        self.weightInit(optim_range)

    def forward(self, x):
        out, _ = self.rnnLayer(x)
        out = out[12:]
        out = self.fcLayer(out)
        return out

    def weightInit(self, gain=1):
        # Init weight
        for name, param in self.named_parameters():
            if 'rnnLayer.weight' in name:
                init.orthogonal(param, gain)

# LSTM
class lstmModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layer_num):
        super().__init__()
        self.lstmLayer = nn.LSTM(in_dim, hidden_dim, layer_num)
        self.relu = nn.ReLU()
        self.fcLayer = nn.Linear(hidden_dim, out_dim)

        self.weightInit = (np.sqrt(1.0/hidden_dim))

    def forward(self, x):
        out, _ = self.lstmLayer(x)
        s, b, h = out.size()  # seq,batch,hidden
        # out=out.view(s*b,h)
        out = self.relu(out)
        out = out[12:]
        out = self.fcLayer(out)
        # out=out.view(s,b,-1)

        return out

    def weightInit(self, gain=1):
        # Init weight
        for name, param in self.named_parameters():
            if 'lstmLayer.weight' in name:
                init.orthogonal(param, gain)

# GRU
class gruModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layer):
        super().__init__()
        self.gruLayer = nn.GRU(in_dim, hidden_dim, hidden_layer)
        self.fcLayer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out, _ = self.gruLayer(x)
        out = out[12:]
        out = self.fcLayer(out)
        return out
