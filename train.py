import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import pickle
import os
import random
from tqdm import tqdm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

from model import *
from non_linear_conv import *
from utils import *
from aggregator import *

import itertools
import wandb

from dataset import *

torch.set_num_threads(1)

args = parse_args()

wandb.init(
    project="HEprediction",
    name=f'{args.dataset}_{args.split}_dim{args.dim}_layers{args.layers}_lr{args.lr}',
    config={
    "dataset": args.dataset,
    "split": args.split,
    "model":args.model,
    "aggr":args.aggr,
    "indexing":args.indexing,
    "epochs": args.epochs,
    "lr": args.lr,
    "dim": args.dim,
    "layers": args.layers,
    "max_sampling": args.max_sampling,
    "description": args.desc
    }
)

device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

data_dict = torch.load(f'../hypergraph_data/splits/{args.dataset}split{args.split}.pt')
data = torch.load(f'../hypergraph_data/{args.dataset}.pt')
num_nodes = data['N_nodes']
node_feat = data['node_feat']

node_feat = F.normalize(torch.Tensor(node_feat), dim=1)

ground = data_dict["ground_train"] + data_dict["ground_valid"] + data_dict["train_only_pos"] + data_dict["valid_only_pos"]
train_pos = data_dict["train_only_pos"] + data_dict["ground_train"]
val_pos = data_dict["ground_valid"] + data_dict["valid_only_pos"]
test_pos = data_dict["test_pos"]

max_len = max([len(hedge) for hedge in ground+train_pos+val_pos+test_pos])
pad_id = num_nodes  # pad token


samples = augment(train_pos+val_pos, num_nodes, args.max_sampling, args.indexing)
train_samples, val_samples = train_test_split(samples, test_size=0.1)

train_dataset = HEDataset(train_samples, max_len, pad_id)
val_dataset = HEDataset(val_samples, max_len, pad_id)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

indices = []
for idx, hedge in enumerate(ground):
    for node in hedge:
        indices.append([idx, node])
inc = torch.sparse_coo_tensor(list(zip(*indices)), [1.0 for _ in range(len(indices))], (len(ground), num_nodes))
inc = generate_G_from_H(inc.T.to_dense())

model = HGNN(node_feat.shape[1], args.dim, node_feat.shape[0], device, dropout=0.3)

optimizer = optim.Adam(list(model.parameters()), lr=args.lr)
criterion = nn.CrossEntropyLoss()

model = model.to(device)
node_feat = node_feat.to(device)
inc = torch.Tensor(inc).to(device)
pad_emb = torch.zeros(1, args.dim+num_nodes).to(device)

min_val_loss = 1e+8
cnt = 0
model_parameters, aggr_parameters = 0, 0
early_stopping = 50

for i in tqdm(range(args.epochs)):
    model.train()
    optimizer.zero_grad()

    train_loss = []
    for idx, batch in enumerate(train_loader):
        src = batch['src'].to(device)
        tgt = batch['tgt'].to(device)
        mask = batch['mask'].to(device).unsqueeze(dim=-1)

        H = model(node_feat, inc)
        H = torch.cat((H, pad_emb), dim=0)

        hedge_embs = torch.mul(mask, H[src])
        hedge_emb = hedge_embs.sum(1)/(mask!=0).sum(1)
        
        logits = torch.mm(hedge_emb, H[:-1].T)
        loss = criterion(logits, tgt)
                
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_loss = []
        H = model(node_feat, inc)
        H = torch.cat((H, pad_emb), dim=0)
        
        for idx, batch in enumerate(val_loader):
            src = batch['src'].to(device)
            tgt = batch['tgt'].to(device)
            mask = batch['mask'].to(device).unsqueeze(dim=-1)

            hedge_embs = torch.mul(mask, H[src])
            hedge_emb = hedge_embs.sum(1)/(mask!=0).sum(1)
            
            logits = torch.mm(hedge_emb, H[:-1].T)
            loss = criterion(logits, tgt)
                    
            val_loss.append(loss.item())
        val_loss = np.mean(val_loss)

        if val_loss < min_val_loss:
            cnt = 0
            min_val_loss = val_loss
            model_parameters = model.state_dict()
        
        wandb.log({"train loss": np.mean(train_loss), "val loss": val_loss})
        cnt += 1
        if cnt > early_stopping:
            break

torch.save(model_parameters, f"checkpoints/model_{args.dataset}{args.split}_{args.model}{args.dim}_{args.max_sampling}.pt")

model_check = torch.load(f"checkpoints/model_{args.dataset}{args.split}_{args.model}{args.dim}_{args.max_sampling}.pt")
model.load_state_dict(model_check)
model = model.to(device)
model.eval()

H = model(node_feat, inc)

test_set = torch.load(f'../hypergraph_data/generations/{args.dataset}split{args.split}_generate.pt')
recall_1, recall_5, recall_10, recall_20, recall_50 = 0, 0, 0, 0, 0
adj = H.detach().cpu()
score_mat = torch.mm(adj, adj.T)

for removed_edge, answer in tqdm(test_set):
    candidate_nodes = [i for i in range(num_nodes) if i not in removed_edge]
    
    hedge_embeds = H[removed_edge]
    hedge_emb = torch.mean(hedge_embeds, dim=0).unsqueeze(0)

    cands = H[candidate_nodes]
    preds = torch.mm(hedge_emb, cands.T).squeeze()
    preds = preds.detach().cpu().numpy()

    indices = np.argsort(preds)[::-1][:50]
    top_10_idx = [candidate_nodes[idx] for idx in indices]

    if answer == top_10_idx[0]: recall_1 += 1
    if answer in top_10_idx[:5]: recall_5 += 1
    if answer in top_10_idx[:10]: recall_10 += 1
    if answer in top_10_idx[:20]: recall_20 += 1
    if answer in top_10_idx[:50]: recall_50 += 1

wandb.log({"recall@1" : recall_1/len(test_set),
           "recall@5" : recall_5/len(test_set),
           "recall@10" : recall_10/len(test_set),
           "recall@20" : recall_20/len(test_set),
           "recall@50" : recall_50/len(test_set)
          })
