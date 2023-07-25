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

from model import *
from non_linear_conv import *
from utils import *
from aggregator import *

import itertools
from sklearn import metrics


def score(hedge, score_mat):
    scores = []
    for i in range(len(hedge)):
        for j in range(i, len(hedge)):
            scores.append(score_mat[hedge[i]][hedge[j]])
    return np.mean(scores)

def measure(label, pred):
    auc_roc = metrics.roc_auc_score(np.array(label), np.array(pred))
    ap = metrics.average_precision_score(np.array(label), np.array(pred))
    return auc_roc, ap

def beam_search(H, beam_size, hedge, num_nodes, num_fills, ground):
    curr_beam = [[hedge, 1]]
    for i in range(num_fills):
        new_beam = [[elem[0], elem[1]*data[1]] for data in curr_beam for elem in search(H, beam_size, data[0], num_nodes)]
        curr_beam = sorted(new_beam, key=lambda item: item[1], reverse=True)[:beam_size]
    return curr_beam

def search(H, beam_size, hedge, num_nodes):
    candidate_nodes = [i for i in range(num_nodes) if i not in hedge]
    
    hedge_embeds = H[hedge]
    hedge_emb = torch.mean(hedge_embeds, dim=0).unsqueeze(0)

    cands = H[candidate_nodes]
    preds = torch.mm(hedge_emb, cands.T).squeeze()
    preds = torch.softmax(preds, dim=-1)

    topk = torch.topk(preds, k=beam_size)
    vals = topk.values.cpu().tolist()
    inds = topk.indices.cpu().tolist()
    top_10_idx = [candidate_nodes[idx] for idx in inds]
    
    result = [[hedge+[node],val] for node, val in zip(top_10_idx, vals)]
    return result


torch.set_num_threads(1)
args = parse_args()

device = 'cuda:{}'.format(args.gpu) if args.gpu != -1 else 'cpu'

for split in range(5):
    args.split = split

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

    indices = []
    for idx, hedge in enumerate(ground):
        for node in hedge:
            indices.append([idx, node])

    inc = torch.sparse_coo_tensor(list(zip(*indices)), [1.0 for _ in range(len(indices))], (len(ground), num_nodes))
    inc = generate_G_from_H(inc.T.to_dense())

    model = HGNN(node_feat.shape[1], args.dim, node_feat.shape[0], device, dropout=0.3)
    inc = torch.Tensor(inc).to(device)

    model_check = torch.load(f'checkpoints/model_{args.dataset}{args.split}_{args.model}{args.dim}_{args.max_sampling}.pt')
    model.load_state_dict(model_check)

    model = model.to(device)
    node_feat = node_feat.to(device)

    model.eval()
    with torch.no_grad():
        H = model(node_feat, inc)
        score_mat = torch.mm(H, H.T).detach().cpu().numpy()

        test_pos = data_dict["test_pos"]
        test_sns = data_dict["test_sns"]
        test_mns = data_dict["test_mns"]
        test_cns = data_dict["test_cns"]
        test_half = data_dict["test_half"]
        test_fitb = data_dict["test_fitb"]
        test_avg = test_sns[:len(test_pos)//3] + test_mns[:len(test_mns)//3] + test_cns[:len(test_pos)-len(test_pos)//3-len(test_pos)//3]

        pos_label = [1 for _ in range(len(test_pos))]
        neg_label = [0 for _ in range(len(test_pos))]

        pos_preds = [score(hedge, score_mat) for hedge in test_pos]
        sns_preds = [score(hedge, score_mat) for hedge in test_sns]
        mns_preds = [score(hedge, score_mat) for hedge in test_mns]
        cns_preds = [score(hedge, score_mat) for hedge in test_cns]
        avg_preds = [score(hedge, score_mat) for hedge in test_avg]
        half_preds = [score(hedge, score_mat) for hedge in test_half]
        fitb_preds = [score(hedge, score_mat) for hedge in test_fitb]

        auc_roc_sns, ap_sns = measure(pos_label+neg_label, pos_preds+sns_preds)
        auc_roc_mns, ap_mns = measure(pos_label+neg_label, pos_preds+mns_preds)
        auc_roc_cns, ap_cns = measure(pos_label+neg_label, pos_preds+cns_preds)
        auc_roc_avg, ap_avg = measure(pos_label+neg_label, pos_preds+avg_preds)
        auc_roc_half, ap_half = measure(pos_label+neg_label, pos_preds+half_preds)
        auc_roc_fitb, ap_fitb = measure(pos_label+neg_label, pos_preds+fitb_preds)
    
        print(f"SNS AUROC  : {auc_roc_sns}")
        print(f"MNS AUROC  : {auc_roc_mns}")
        print(f"CNS AUROC  : {auc_roc_cns}")
        print(f"AVG AUROC  : {auc_roc_avg}")
        print(f"HALF AUROC : {auc_roc_half}")
        print(f"FITB AUROC : {auc_roc_fitb}")

        test_set = torch.load(f'../hypergraph_data/generations/{args.dataset}split{args.split}_generate.pt')
        recall_1, recall_5, recall_10, recall_20, recall_50 = 0, 0, 0, 0, 0
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

        print("Recall@1  : ", recall_1/len(test_set))
        print("Recall@5  : ", recall_5/len(test_set))
        print("Recall@10 : ", recall_10/len(test_set))
        print("Recall@20 : ", recall_20/len(test_set))
        print("Recall@50 : ", recall_50/len(test_set))
        
        test_pos = [set(hedge) for hedge in test_pos]
        # ground = [set(hedge) for hedge in ground if hedge not in test_pos]
        ground = [set(hedge) for hedge in ground]

        test_set = torch.load(f'../hypergraph_data/generations/first/{args.dataset}split{args.split}_generate.pt')
        preds = []
        for sample in tqdm(test_set):
            src, tgt = sample[0], sample[1]
            result = beam_search(H, 10, src, num_nodes, len(tgt), ground)
            pred = []
            for res in result:
                pred.append(set(res[0]))
            preds.append(pred)
        cnt = 0
        rank = 0
        for pred in preds:
            for idx, hedge in enumerate(pred):
                if hedge in test_pos:
                    cnt += 1
                    rank += 1/(1+idx)
                    break
        print(f"1-Recall@k: {cnt/len(preds)}")
        print(f"1-MRR@k   : {rank/len(preds)}")
        cnt = 0
        preds = [hedge for pred in preds for hedge in pred]
        for pos in test_pos:
            if pos in preds:
                cnt+=1
        print(f"1-Coverage: {cnt/len(test_pos)}")
        

        test_set = torch.load(f'../hypergraph_data/generations/second/{args.dataset}split{args.split}_generate.pt')
        preds = []
        for sample in tqdm(test_set):
            src, tgt = sample[0], sample[1]
            result = beam_search(H, 10, src, num_nodes, len(tgt), ground)
            pred = []
            for res in result:
                pred.append(set(res[0]))
            preds.append(pred)
        cnt = 0
        rank = 0
        for pred in preds:
            for idx, hedge in enumerate(pred):
                if hedge in test_pos:
                    cnt += 1
                    rank += 1/(1+idx)
                    break
        print(f"2-Recall@k: {cnt/len(preds)}")
        print(f"2-MRR@k   : {rank/len(preds)}")
        cnt = 0
        preds = [hedge for pred in preds for hedge in pred]
        for pos in test_pos:
            if pos in preds:
                cnt+=1
        print(f"2-Coverage: {cnt/len(test_pos)}")
        

        test_set = torch.load(f'../hypergraph_data/generations/third/{args.dataset}split{args.split}_generate.pt')
        preds = []
        for sample in tqdm(test_set):
            src, tgt = sample[0], sample[1]
            result = beam_search(H, 10, src, num_nodes, len(tgt), ground)
            pred = []
            for res in result:
                pred.append(set(res[0]))
            preds.append(pred)
        cnt = 0
        rank = 0
        for pred in preds:
            for idx, hedge in enumerate(pred):
                if hedge in test_pos:
                    cnt += 1
                    rank += 1/(1+idx)
                    break
        print(f"3-Recall@k: {cnt/len(preds)}")
        print(f"3-MRR@k   : {rank/len(preds)}")
        cnt = 0
        preds = [hedge for pred in preds for hedge in pred]
        for pos in test_pos:
            if pos in preds:
                cnt+=1
        print(f"3-Coverage: {cnt/len(test_pos)}")


        test_set = torch.load(f'../hypergraph_data/generations/moderate/{args.dataset}split{args.split}_generate.pt')
        preds = []
        for sample in tqdm(test_set):
            src, tgt = sample[0], sample[1]
            result = beam_search(H, 10, src, num_nodes, len(tgt), ground)
            pred = []
            for res in result:
                pred.append(set(res[0]))
            preds.append(pred)
        cnt = 0
        rank = 0
        for pred in preds:
            for idx, hedge in enumerate(pred):
                if hedge in test_pos:
                    cnt += 1
                    rank += 1/(1+idx)
                    break
        print(f"H-Recall@k: {cnt/len(preds)}")
        print(f"H-MRR@k   : {rank/len(preds)}")
        cnt = 0
        preds = [hedge for pred in preds for hedge in pred]
        for pos in test_pos:
            if pos in preds:
                cnt+=1
        print(f"H-Coverage: {cnt/len(test_pos)}\n")
