
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

def convert_len(seq, maxlen):
    if len(seq) >= maxlen:
        return seq[:maxlen]
    else:  # padding
        pad = '8' * int(maxlen - len(seq))
        return seq + pad
    
def get_mat_from_result_tuple(result_tuple, aseq, bseq, peptide, showplot=False):
    MAXLENGTH_A, MAXLENGTH_B, max_len_epitope = 28, 28, 25
    
    #abseq = convert_len(aseq, MAXLENGTH_A)convert_len(bseq, MAXLENGTH_B) 
    attn_output_weights1 = result_tuple[0]
    attn_output_weights2 = result_tuple[1]
    print(f"aseq={aseq}, bseq={bseq}, peptide={peptide}")
    abseq_with_comma = f'{aseq}:{bseq}'
    abseq_index = convert_len(aseq, MAXLENGTH_A) + ':' + convert_len(bseq, MAXLENGTH_B) 
    
    attn_output_weights2_list = []
    for head_i in range(4):
#         print('head', head_i, attn_output_weights2[head_i].shape)
        a = attn_output_weights2[head_i]
        dfa = pd.DataFrame(a)
        dfa.insert(27, "delimiter", [0.1**9 for _ in range(len(dfa))])
        dfa = dfa.loc[:, ((dfa!=0).sum()!=0).values]
        dfa.columns = list(abseq_with_comma)
        dfa.columns = [f'{c}_{i}' for i,c in enumerate(dfa.columns)]
        dfa = dfa.head(len(peptide.replace('8','')))
        dfa.index = list(peptide.replace('8',''))
        dfa.index = [f'{ind}_{i}' for i,ind in enumerate(dfa.index)]    
        #         print('df.sum(axis=0)', dfa.sum(axis=0))
        #         print('df.sum(axis=1)', dfa.sum(axis=1))
        # display(px.imshow(dfa, width=800, height=480))
        
        if showplot:
            axs[0,head_i].imshow(dfa, aspect='equal')
            axs[0,head_i].set_xticks(range(len(dfa.columns)))
            axs[0,head_i].set_yticks(range(len(dfa.index)))
            axs[0,head_i].set_xticklabels(dfa.columns)
            axs[0,head_i].set_yticklabels(dfa.index)
        attn_output_weights2_list.append(dfa)
    
    attn_output_weights1_list = []
    for head_i in range(4):
#         print('head', head_i, attn_output_weights1[head_i].shape)
        a = attn_output_weights1[head_i]
        dfa = pd.DataFrame(a).T
        dfa.insert(MAXLENGTH_A, "delimiter", [0.1**9 for _ in range(len(dfa))])
        dfa = dfa.T
        dfa = dfa.loc[:, ((dfa!=0).sum()!=0).values]
        dfa.index = list(abseq_index)
        dfa.index = [f'{c}_{i}' for i,c in enumerate(dfa.index)]
        dfa.columns = list(convert_len(peptide, len(dfa.columns)))
        dfa.columns = [f'{ind}_{i}' for i,ind in enumerate(dfa.columns)]
        selector_columns = [c for c in dfa.columns if '8_' not in c]
        selector_index = [c for c in dfa.index if '8_' not in c]
        dfa = dfa.loc[selector_index]
        dfa.index = [f'{c}_{i}' for i,c in enumerate(abseq_with_comma)]
        dfa = dfa[selector_columns]
        #         print('df.sum(axis=0)', dfa.sum(axis=0))
        #         print('df.sum(axis=1)', dfa.sum(axis=1))
        # display(px.imshow(dfa, width=800, height=480))
        if showplot:
            axs[1,head_i].imshow(dfa, aspect='equal')
            axs[1,head_i].set_xticks(range(len(dfa.columns)))
            axs[1,head_i].set_yticks(range(len(dfa.index)))
            axs[1,head_i].set_xticklabels(dfa.columns)
            axs[1,head_i].set_yticklabels(dfa.index)
        attn_output_weights1_list.append(dfa)
    return attn_output_weights1_list, attn_output_weights2_list


def make_distance_mat(traindf, testdf, file_prefix):
    import multiprocessing as mp
    import os
    
    df_tcrsall = pd.concat([traindf, testdf])
    df_tcrsall = df_tcrsall.drop_duplicates(subset="tcr_combined")
    df_tcrsall['tcraXtcrb'] = [s.replace(":", "P"*100) for s in df_tcrsall.tcr_combined]
    sequences = df_tcrsall.tcr_combined.values.tolist()

    ofile = open(f"{file_prefix}_fasta.txt", "w")
    for i, seq in enumerate(df_tcrsall.tcraXtcrb.tolist()):
        ofile.write(">" + str(i) + "\n" + seq + "\n")
    ofile.close()

    if not os.path.exists(f"{file_prefix}_aligned.txt"):
        print("running clustal-omega")
        os.system(f"~/Downloads/clustalo -i {file_prefix}_fasta.txt -o {file_prefix}_aligned.txt --full --distmat-out={file_prefix}_distmat.txt")

    if not os.path.exists(f"{file_prefix}_distmat.parquet"):
        pd_reader = pd.read_csv(f'{file_prefix}_distmat.txt',sep=' 0', skiprows=[0], header=None, 
                                      chunksize=1000)

        # with mp.Pool(os.cpu_count()-1) as p:
        #     df_list = p.map(lambda x:x, pd_reader)

        # Create df_list from chunked dataframes, pd_reader.
        df_list = [d for d in pd_reader]

        
        df_distmat = pd.concat(df_list)
        df_distmat = df_distmat.iloc[:,1:]
        df_distmat.columns = df_distmat.columns.astype(str)
        df_distmat.to_parquet(f"{file_prefix}_distmat.parquet")
    else:
        df_distmat = pd.read_parquet(f'{file_prefix}_distmat.parquet')

    df_distmat.index = sequences
    df_distmat.columns = sequences
    return df_distmat
    