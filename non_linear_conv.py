import math
import torch
import copy

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from scipy.sparse import coo_matrix

from torch import Tensor
from torch.nn import Linear
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add, scatter
from torch_geometric.typing import Adj, Size, OptTensor
from typing import Optional, Callable

from sklearn.metrics import average_precision_score as apscore # True / Pred
from sklearn.metrics import roc_auc_score as auroc

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear as Linear2
from torch_geometric.nn.inits import zeros

from tqdm import tqdm

class NonLHGNN(MessagePassing) :

    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, dropout: float = 0.0,
                 bias: bool = False, cached: bool = False, row_norm: bool = True, **kwargs) :
        kwargs.setdefault('aggr', 'add')
        kwargs.setdefault('flow', 'source_to_target')
        super().__init__(node_dim=0, **kwargs)

        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.act = torch.nn.ReLU()
        self.row_norm = row_norm
        self.cached = cached

        self.lin_n2e = Linear2(in_dim, hid_dim, bias=False)
        self.lin_e2n = Linear2(hid_dim, out_dim, bias=False)

        if bias:
            self.bias_n2e = Parameter(torch.Tensor(hid_dim))
            self.bias_e2n = Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias_n2e', None)
            self.register_parameter('bias_e2n', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_n2e.reset_parameters()
        self.lin_e2n.reset_parameters()
        zeros(self.bias_n2e)
        zeros(self.bias_e2n)
        self._cached_norm_n2e = None
        self._cached_norm_e2n = None

    def forward(self, x: Tensor, hyperedge_index: Tensor,
                num_nodes: Optional[int] = None, num_edges: Optional[int] = None):

        if num_nodes is None:
            num_nodes = x.shape[0]
        if num_edges is None and hyperedge_index.numel() > 0:
            num_edges = int(hyperedge_index[1].max()) + 1

        cache_norm_n2e = self._cached_norm_n2e
        cache_norm_e2n = self._cached_norm_e2n

        if (cache_norm_n2e is None) or (cache_norm_e2n is None):
            hyperedge_weight = x.new_ones(num_edges)

            node_idx, edge_idx = hyperedge_index
            Dn = scatter_add(hyperedge_weight[hyperedge_index[1]],
                             hyperedge_index[0], dim=0, dim_size=num_nodes)
            De = scatter_add(x.new_ones(hyperedge_index.shape[1]),
                             hyperedge_index[1], dim=0, dim_size=num_edges)
                
            if self.row_norm:
                Dn_inv = 1.0 / Dn
                Dn_inv[Dn_inv == float('inf')] = 0
                De_inv = 1.0 / De
                De_inv[De_inv == float('inf')] = 0

                norm_n2e = De_inv[edge_idx]
                norm_e2n = Dn_inv[node_idx]

            else:
                Dn_inv_sqrt = Dn.pow(-0.5)
                Dn_inv_sqrt[Dn_inv_sqrt == float('inf')] = 0
                De_inv_sqrt = De.pow(-0.5)
                De_inv_sqrt[De_inv_sqrt == float('inf')] = 0

                norm = De_inv_sqrt[edge_idx] * Dn_inv_sqrt[node_idx]
                norm_n2e = norm
                norm_e2n = norm

            if self.cached:
                self._cached_norm_n2e = norm_n2e
                self._cached_norm_e2n = norm_e2n
        else:
            norm_n2e = cache_norm_n2e
            norm_e2n = cache_norm_e2n

        x = self.lin_n2e(x)
        e = self.propagate(hyperedge_index, x=x, norm=norm_n2e,
                           size=(num_nodes, num_edges))  # Node to edge

        e = self.act(e)
        x = self.lin_e2n(e)
        n = self.propagate(hyperedge_index.flip([0]), x=x, norm=norm_e2n,
                           size=(num_edges, num_nodes))  # Edge to node
        return n, e  # No act, act

    def message(self, x_j: Tensor, norm: Tensor):
        return norm.view(-1, 1) * x_j

class HypergraphEncoder(nn.Module):
    
    def __init__(self, num_nodes, num_edges, in_dim, edge_dim, node_dim, drop_p = 0.3, num_layers=2, 
                 cached = False) :
        super(HypergraphEncoder, self).__init__()
        self.num_nodes = num_nodes
        self.num_edges = num_edges

        self.in_dim = in_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.num_layers = num_layers
        self.act = nn.ReLU()
        self.DropLayer = nn.Dropout(p=drop_p)

        self.convs = nn.ModuleList()
        if num_layers == 1:
            self.convs.append(NonLHGNN(self.in_dim, self.edge_dim, self.node_dim, cached=cached))
        else:
            self.convs.append(NonLHGNN(self.in_dim, self.edge_dim, self.node_dim, cached=cached))
            for _ in range(self.num_layers - 2):
                self.convs.append(NonLHGNN(self.node_dim, self.edge_dim, self.node_dim, cached=cached))
            self.convs.append(NonLHGNN(self.node_dim, self.edge_dim, self.node_dim, cached=cached))
        self.reset_parameters()
        
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x: Tensor, hyperedge_index: Tensor) :
        for i in range(self.num_layers):
            x, e = self.convs[i](x, hyperedge_index, self.num_nodes, self.num_edges)
            x = self.DropLayer(x)
            if i < self.num_layers - 1 : # For previous layers, do activation
                x = self.act(x)
        return x # Only Returns Node Embeddings