import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import math
from utils import *


from torch_scatter import scatter

class HNHN(nn.Module):
    """
    """

    def __init__(self, args):
        super(HNHN, self).__init__()

        self.num_layers = args.All_num_layers
        self.dropout = args.dropout
        
        self.convs = nn.ModuleList()
        # two cases
        if self.num_layers == 1:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
        else:
            self.convs.append(HNHNConv(args.num_features, args.MLP_hidden, args.MLP_hidden,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            for _ in range(self.num_layers - 2):
                self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.MLP_hidden,
                                           nonlinear_inbetween=args.HNHN_nonlinear_inbetween))
            self.convs.append(HNHNConv(args.MLP_hidden, args.MLP_hidden, args.num_classes,
                                       nonlinear_inbetween=args.HNHN_nonlinear_inbetween))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, data):

        x = data.x
        
        if self.num_layers == 1:
            conv = self.convs[0]
            x = conv(x, data)
            # x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            for i, conv in enumerate(self.convs[:-1]):
                x = F.relu(conv(x, data))
                x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.convs[-1](x, data)

        return x


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
