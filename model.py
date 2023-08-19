import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
from utils import *

from torch_scatter import scatter

""" adapted from https://github.com/iMoonLab/HGNN.git """


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()
        self.lin = nn.Linear(in_ft, out_ft, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, x, G):
        x = F.relu(self.lin(x))
        x = torch.matmul(G, x)

        return x

class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, num_nodes, device, dropout=0.3):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_hid) 
        # self.emb = nn.Embedding(num_nodes, n_hid)
        # self.idt = torch.eye(num_nodes).to(device)
        self.param = nn.Parameter(torch.Tensor([1 for _ in range(num_nodes)]))

    def reset_parameters(self):
        self.hg1.reset_parameters()
        self.hg2.reset_parameters()
        
    def forward(self, x, G):
        x = self.hgc1(x, G)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        x = torch.cat((x, torch.mm(torch.diag(self.param), G)), dim=1)

        return x

