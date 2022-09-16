
import os
import subprocess
import pickle
from tqdm import tqdm
import pandas as pd
# from analysis_util import calc_melt_df
from matplotlib import pyplot as plt
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt

# %%writefile -a analysis_util.py
def calc_melt_df(distmat_vis, attn_output_weights1, attn_output_weights2):
    separator = [c for c in attn_output_weights2.columns if ':' in c][0]
    separator_pos = attn_output_weights2.columns.get_loc(separator)
    removers = [':',
                # 'C_0'
                ] + \
        [attn_output_weights2.columns[i] for i in [separator_pos,
                                                   # separator_pos-1, separator_pos+1,
                                                   # len(attn_output_weights2.columns)-1
                                                   ]
         ]
    remove_col = [c for c in attn_output_weights2.columns if c in removers]
    remove_ind = [c for c in attn_output_weights1.index if c in removers]
    attn_output_weights2 = attn_output_weights2.drop(columns=remove_col)
    attn_output_weights1 = attn_output_weights1.drop(index=remove_ind)
    
    attn_output_weights2_vis = attn_output_weights2.melt(ignore_index=False).reset_index().rename(columns={'variable':'tcr', 'index':'peptide'})
    attn_output_weights2_vis= attn_output_weights2_vis.sort_values(by=['peptide', 'tcr'])
    attn_output_weights1_vis = attn_output_weights1.T.melt(ignore_index=False).reset_index().rename(columns={'variable':'tcr', 'index':'peptide'})
    attn_output_weights1_vis = attn_output_weights1_vis.sort_values(by=['peptide', 'tcr'])
    
    merged_xy_1 = pd.merge(distmat_vis, attn_output_weights1_vis, on=['peptide','tcr'], how='inner')
    merged_xy_2 = pd.merge(distmat_vis, attn_output_weights2_vis, on=['peptide','tcr'], how='inner')
    return merged_xy_1, merged_xy_2



def tcr_a_or_b(row):
    if row['chain1_type'] == 'alpha':
        return pd.Series({'tcra_seq': row['chain1_cdr3_seq_calculated'],
                          'tcrb_seq': row['chain2_cdr3_seq_calculated']})
    else: #if row['chain1_type'] == 'beta':
        return pd.Series({'tcrb_seq': row['chain1_cdr3_seq_calculated'],
                          'tcra_seq': row['chain2_cdr3_seq_calculated']})





def pdread(path: str) -> pd.DataFrame:
    if path.endswith("pkl") or path.endswith("pickle"):
        return pd.read_pickle(path)
    if path.endswith("csv"):
        return pd.read_csv(path)
    if path.endswith("tsv"):
        return pd.read_csv(path, delimiter="\t")
    return pd.read_parquet(path)

