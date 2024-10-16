import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb


""" adapted from https://github.com/juho-lee/set_transformer.git """
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    # Q : (batch_size * n * dim)
    def forward(self, Q, K):
        #pdb.set_trace()
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)
        dim_split = self.dim_V // self.num_heads

        Q_ = torch.cat(Q.split(dim_split, 1), 0)
        K_ = torch.cat(K.split(dim_split, 1), 0)
        V_ = torch.cat(V.split(dim_split, 1), 0)
        
        A = torch.softmax(torch.matmul(Q_,K_.transpose(1,2))/math.sqrt(self.dim_V), 1)
        O = torch.cat((Q_ + torch.matmul(A, V_)).split(Q.size(0), 0), 1)

        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)

class SetAggregator(nn.Module):
    def __init__(self, dim_vertex, num_heads=1, layers=1, num_seeds=1, ln=True):
        super(SetAggregator, self).__init__()
        self.sab = SAB(dim_vertex, dim_vertex, num_heads, ln=ln)

    def forward(self, embeddings, weights = None):
        embedding = self.sab(embeddings).mean(dim=1)

        return embedding

class MeanAggregator(nn.Module):
    def __init__(self, dim_vertex):
        super(MeanAggregator, self).__init__()
    
    def forward(self, embeddings):
        embedding = torch.mean(embeddings, dim=1)

        return embedding

class MeanMLPAggregator(nn.Module):
    def __init__(self, dim_vertex, dim_hid):
        super(MeanMLPAggregator, self).__init__()
        self.lin = nn.Linear(dim_vertex, dim_hid)
        self.lin2 = nn.Linear(dim_hid, 1)
        
    def forward(self, embeddings):
        embedding = torch.mean(embeddings, dim=0)
        embedding = F.dropout(F.relu(self.lin(embedding)), p=0.2)
        embedding = F.sigmoid(self.lin2(embedding))
        
        return embedding

