import argparse
import numpy as np
import torch
import random

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='cora', help='dataset name')
    parser.add_argument("--seed", dest='fix_seed', action='store_const', default=False, const=True, help='Fix seed for reproducibility and fair comparison.')
    parser.add_argument("--gpu", type=int, default=0, help='gpu number. -1 if cpu else gpu number')
    parser.add_argument("--split", default=0, type=int, help='split')
    parser.add_argument("--epochs", default=100, type=int, help='number of epochs')
    parser.add_argument("--dropout", default=0.3, type=float, help='dropout')
    parser.add_argument("--batch_size", default=128, type=int, help='batch size')
    parser.add_argument("--dim", default=128, type=int, help='dimension of hidden vector')
    parser.add_argument("--layers", default=1, type=int, help='dimension of hidden vector')        

    ## learning rate 
    parser.add_argument("--model", default='hgnn', type=str, help='Use which model')
    parser.add_argument("--aggr", type=str, default='mean', help='aggregation method: mean, set, sagnn')

    parser.add_argument("--lr", default=0.001, type=float, help='learning rate')
    parser.add_argument("--scheduler", type=str, default='multi', help='schedular: multi or plateau')
    parser.add_argument("--lrmode", type=str, default='const', help='learning rate = const / multistep / plateau / accuracy')
    
    parser.add_argument("--indexing", type=str, default='smooth', help='indexing')
    parser.add_argument("--max_sampling", type=int, default=10, help='max sampling per node')
    parser.add_argument("--desc", type=str, default='none', help='description')
    
    return parser.parse_args()
 

def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def generate_negative(hedge, num_nodes):
    candidates = [i for i in range(num_nodes) if i not in hedge]    
    hedge_size = len(hedge)
    num_nodes_to_change = round(hedge_size/2)
    # num_nodes_to_change = 1
    
    half_removed = random.sample(hedge, hedge_size-num_nodes_to_change)
    half_to_add = random.sample(candidates, num_nodes_to_change)
    
    negative_hedge = half_removed + half_to_add
    
    return negative_hedge


def make_batch(hedges, labels, pad_id, batch_size):
    hedge_batches = [hedges[i:min(i+batch_size, len(hedges))] for i in range(0, len(hedges), batch_size)]
    batch_max_lens = [max([len(sample) for sample in hedges[i:min(i+batch_size, len(hedges))]]) for i in range(0, len(hedges), batch_size)]
    labels_batches = [labels[i:min(i+batch_size, len(labels))] for i in range(0, len(labels), batch_size)]

    batches = []
    for i in range(len(hedge_batches)):
        max_len = batch_max_lens[i]
        node_indices = [hedge+[pad_id for _ in range(max_len-len(hedge))] for hedge in hedge_batches[i]]
        node_indices = [idx for indices in node_indices for idx in indices]
        hedge_indices = [[idx for _ in range(len(hedge))]+[len(hedge_batches[i]) for _ in range(max_len-len(hedge))] for idx, hedge in enumerate(hedge_batches[i])]
        hedge_indices = [idx for indices in hedge_indices for idx in indices]

        batches.append([torch.LongTensor(node_indices), torch.LongTensor(hedge_indices), torch.FloatTensor(labels_batches[i])])
    return batches