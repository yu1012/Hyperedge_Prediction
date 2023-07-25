import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
import itertools

def augment(hedges, num_nodes, max_sampling, indexing):
    samples = []
    for pos in hedges:
        sample = []
        for i in range(1, len(pos)):
            if len(pos) > 10:
                idx = [0 if j not in pos else 1.0/len(pos) for j in range(num_nodes)]
                for _ in range(max_sampling):
                    random.shuffle(pos)
                    sample.append((pos[:len(pos)//2], idx))

                    if indexing == 'discrete':
                        tgt = list(set(pos)-set(pos[:len(pos)//2]))
                        idx = [0 if j not in tgt else 1.0/len(tgt) for j in range(num_nodes)]
                        samples.append((tgt, idx))
            else:
                srcs = [list(node) for node in list(itertools.combinations(pos, i))]

                idx = [0 if j not in pos else 1.0/len(pos) for j in range(num_nodes)]
                for src in srcs:
                    tgt = list(set(pos)-set(src))
                    if indexing == 'discrete':
                        idx = [0 if j not in tgt else 1.0/len(tgt) for j in range(num_nodes)]
                    sample.append((src, idx))
            random.shuffle(sample)
            samples.extend(sample[:min(len(sample), max_sampling)])
        random.shuffle(samples)
    return samples

def augment_fitb(hedges, num_nodes, max_sampling, indexing):
    samples = []
    for pos in hedges:
        idx = [0 if j not in pos else 1.0/len(pos) for j in range(num_nodes)]
        
        for node in pos:
            src = list(set(pos)-set([node]))            
            samples.append((src, idx))

    random.shuffle(samples)

    return samples


class HEDataset(Dataset):
    def __init__(self,
                 samples,
                 max_len,
                 pad_id
                 ):

        super().__init__()

        self.samples = samples
        self.max_len = max_len
        self.pad_id = pad_id

    def pad(self, sample):
        if len(sample) < self.max_len:
            sample = sample + [self.pad_id] * (self.max_len - len(sample))
        return sample


    def get_item(self, idx):
        sample = self.samples[idx]
        src = self.pad(sample[0])
        
        mask = [1 if i != self.pad_id else 0 for i in src]
        
        tgt = sample[1]

        src = torch.LongTensor(src)
        tgt = torch.Tensor(tgt)
        mask = torch.Tensor(mask)

        return {
            "src": src,
            "tgt": tgt,
            "mask": mask
        }


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.get_item(index)