# %load ~/jupyter_notebook/user_work/tcrpred/recipes/dataset.py
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import sys
sys.path.append('../')
from recipes.utils import get_03_data, get_all_combination_and_fill_nagatives, get_infer_report
# from utils import get_03_data, get_all_combination_and_fill_nagatives, get_infer_report

NUM_AA_FEATS = 7
NUM_CHAIN_FEATS = 6

dict_chain = {
    'H1': 1,
    'H2': 2, 
    'H3': 3, 
    'L1': 4, 
    'L2': 5, 
    'L3': 6,
}

dict_aa = {  # dimensionality must be a even number
       'C': [1.77, 0.13, 2.43, 1.54, 6.35, 0.17, 0.41],
       'S': [1.31, 0.06, 1.60, -0.04, 5.70, 0.20, 0.28],
       'T': [3.03, 0.11, 2.60, 0.26, 5.60, 0.21, 0.36],
       'P': [2.67, 0.00, 2.72, 0.72, 6.80, 0.13, 0.34],
       'A': [1.28, 0.05, 1.00, 0.31, 6.11, 0.42, 0.23],
       'G': [0.00, 0.00, 0.00, 0.00, 6.07, 0.13, 0.15],
       'N': [1.60, 0.13, 2.95, -0.60, 6.52, 0.21, 0.22],
       'D': [1.60, 0.11, 2.78, -0.77, 2.95, 0.25, 0.20],
       'E': [1.56, 0.15, 3.78, -0.64, 3.09, 0.42, 0.21],
       'Q': [1.56, 0.18, 3.95, -0.22, 5.65, 0.36, 0.25],
       'H': [2.99, 0.23, 4.66, 0.13, 7.69, 0.27, 0.30],
       'R': [2.34, 0.29, 6.13, -1.01, 10.74, 0.36, 0.25],
       'K': [1.89, 0.22, 4.77, -0.99, 9.99, 0.32, 0.27],
       'M': [2.35, 0.22, 4.43, 1.23, 5.71, 0.38, 0.32],
       'I': [4.19, 0.19, 4.00, 1.80, 6.04, 0.30, 0.45],
       'L': [2.59, 0.19, 4.00, 1.70, 6.04, 0.39, 0.31],
       'V': [3.67, 0.14, 3.00, 1.22, 6.02, 0.27, 0.49],
       'F': [2.94, 0.29, 5.89, 1.79, 5.67, 0.30, 0.38],
       'Y': [2.94, 0.30, 6.47, 0.96, 5.66, 0.25, 0.41],
       'W': [3.21, 0.41, 8.08, 2.25, 5.94, 0.32, 0.42]
}
 


def get_aa_feat(protein):
    protein = list(protein)
    unk = [0]*NUM_AA_FEATS
    aa_feat = [dict_aa.get(aa, unk) for aa in protein]
    return np.array(aa_feat)


def get_chain_feat(chain, length):
    one_hot = [0]*NUM_CHAIN_FEATS
    one_hot[dict_chain[chain]-1] = 1
    return np.array([one_hot]*length)


def get_data_loaders_agab(ds, fit_batch_size, fit_pdb_ids, val_pdb_ids):

    def gen_ids(pdb_ids):
        ret = []
        for i in pdb_ids:
            ret += list(range(6*i, 6*(i+1)))
        return ret

    fit_ids = gen_ids(fit_pdb_ids)
    val_ids = gen_ids(val_pdb_ids)
    fit_ds = Subset(ds, fit_ids)
    val_ds = Subset(ds, val_ids)
    fit_dl = DataLoader(fit_ds, batch_size=fit_batch_size, num_workers=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=6, num_workers=6, shuffle=False)  
    return fit_dl, val_dl

def get_data_loaders_pp(ds, fit_batch_size, fit_ids, val_ids):
    fit_ds = Subset(ds, fit_ids)
    val_ds = Subset(ds, val_ids)
    fit_dl = DataLoader(fit_ds, batch_size=fit_batch_size, num_workers=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8, num_workers=8, shuffle=False)  
    return fit_dl, val_dl


class Vocab():
    # not need to be a class (?): align with dict_aa/chain
    def __init__(self):
        self.padding_index = 0
        self.unknown_index = 1
        self.separator_index = 22
        self.seq_to_i = {'<pad>':0, '<unk>':1, 'A':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7, 'H':8, 'I':9, 
                         'K':10, 'L':11, 'M':12, 'N': 13, 'P':14, 'Q':15, 'R':16, 'S':17, 'T':18, 'V':19, 
                         'W':20, 'Y':21, ':':22}
        self.i_to_seq = {v:k for k,v in self.seq_to_i.items()}
        
    def __len__(self):
        return len(self.i_to_seq)

    def get_idxs_from_seq(self, xs):
        return [self.seq_to_i.get(x, self.unknown_index) for x in xs]

class TCRDataset(Dataset): # The very original of Donors (10x experiments)
    def __init__(self, datapath, donors, max_len_ABseq=45, max_len_epitope=12, size=None):
        df = get_03_data(datapath)
        donor_dict = {d:df[df.Donor==d] for d in df.Donor.unique()}
        data_all_by_donor = {d: get_all_combination_and_fill_nagatives(
            donor_dict[d], donor=d) for d in df.Donor.unique()}
        df = pd.concat([d for d in data_all_by_donor.values()])
        df['interact'] = df['interact'].astype(int)
        self.max_len_ABseq = max_len_ABseq
        self.max_len_epitope = max_len_epitope
        self.vocab = Vocab()
        self.donors = donors
        self.size = size
        df = df[df['Donor'].isin(donors)]
        if self.size=='small':
            df = df.sample(200, random_state=0)
        elif self.size=='datasize4hpo':
            df = df.sample(int(len(df)/3), random_state=0)
        df = df.reset_index(drop=True)
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data.loc[item]  # get a row as a series
        a_b_seq = self.add_token_padding_segments(data['ABPairSeq'], self.max_len_ABseq)  # AB sequence
        epitope = self.add_token_padding_segments(data['EpiSeq'], self.max_len_epitope, epitope=True)  # AB sequence
        y = data['interact']
        return (torch.LongTensor(a_b_seq), torch.LongTensor(epitope)), torch.LongTensor([y])[0]

    def add_token_padding_segments(self, sequence, max_len, epitope=False):
        l = len(sequence)
        tok = np.zeros(max_len)  # padding
        pos = np.zeros(max_len)  # padding
        seg = np.zeros(max_len)  # padding
        tok[:l] = np.array(self.vocab.get_idxs_from_seq(sequence)).astype(np.int64)
        

        if epitope:
            pos[:l] = np.arange(1, l+1).astype(np.int64)
            seg[:l] = 3  # a segmentation
            return np.array([tok, pos, seg])
        else:
            a_len = sequence.split(':')[0].__len__()
            b_len = sequence.split(':')[1].__len__()
            pos[:a_len] = np.arange(1, a_len+1).astype(np.int64)
            pos[a_len+1: a_len+1+b_len] = np.arange(1, b_len+1).astype(np.int64)
            seg[:a_len] = 1    # a segmentation
            seg[a_len+1: a_len + 1 + b_len] = 2   # b segmentation
            return np.array([tok, pos, seg])

    def add_token_padding_segments_epitope(self, sequence, max_len):
        l = len(sequence)
        tok = np.zeros(max_len)  # padding
        pos = np.zeros(max_len)  # padding
        seg = np.zeros(max_len)  # padding


class MCPASDataset(Dataset):
    def __init__(self, df, max_len_A=28, max_len_B=28, max_len_epitope=25, size=None):
        if 'tcra' in df.columns:
            df = df[df.tcra!='UNK']
        df['sign'] = df['sign'].astype(int)
        print("datasize and y-vcounts: ", df.shape, df['sign'].value_counts())
        self.max_len_A = max_len_A
        self.max_len_B = max_len_B
        self.max_len_epitope = max_len_epitope
        self.vocab = Vocab()
        self.size = size
        if self.size=='small': df = df.sample(200, random_state=0)
        elif self.size=='datasize4hpo': df = df.sample(int(len(df)/3), random_state=0)
        else: pass
        df = df.reset_index(drop=True)
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data.loc[item]  # get a row as a series
        a_seq = self.add_token_padding_segments(data['tcra'], self.max_len_A, "a")  # A sequence
        b_seq = self.add_token_padding_segments(data['tcrb'], self.max_len_B, "b")  # B sequence
        a_b_seq = np.c_[a_seq, np.array([self.vocab.seq_to_i.get(":"), 0, 4]).reshape(-1,1), b_seq]
        # a_b_seq = np.r_[a_b_seq, np.ones((1, a_b_seq.shape[1]))]
        
        epitope = self.add_token_padding_segments(data['peptide'], self.max_len_epitope, "c")  # AB sequence
        # epitope = np.r_[epitope, 2*np.ones((1, epitope.shape[1]))]
        y = data['sign']
        return (torch.LongTensor(a_b_seq.astype(np.int64)), torch.LongTensor(epitope.astype(np.int64))), torch.LongTensor([y])[0]

    def add_token_padding_segments(self, sequence, max_len, types):
        l = len(sequence)
        tok = np.zeros(max_len)  # padding
        pos = np.zeros(max_len)  # padding
        seg = np.zeros(max_len)  # padding
        tok[:l] = np.array(self.vocab.get_idxs_from_seq(sequence)).astype(np.int64)
        
        if types=="c":
            pos[:l] = np.arange(1, l+1).astype(np.int64)
            seg[:l] = 3  # a segmentation
            return np.array([tok, pos, seg])
        else:
            s_len = sequence.__len__()
            pos[:s_len] = np.arange(1, s_len+1).astype(np.int64)
            seg[:s_len] = 1 if types == "a" else 2   # a or b segmentation
            return np.array([tok, pos, seg])
