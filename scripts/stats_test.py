#!/usr/bin/env python
# coding: utf-8
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit
from scipy.stats import ttest_ind_from_stats
from scipy import stats
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, roc_curve, auc, precision_recall_curve

flatten = lambda xlist: [x for xx in xlist for x in xx]
pickleload = lambda p: pickle.load(open(p,"rb"))

pd.options.display.float_format = '{:.5g}'.format


prop_list = ['is_connecting_to_pep', 'is_connecting_to_cdr', 
             'is_connecting_to_ownchain_tcr','is_connecting_to_ownchain_cdr',
             'is_connecting_to_opposite_chain_tcr', 'is_connecting_to_opposite_chain_cdr',
             'is_connecting_to_tcr', 'is_connecting_to_notCDR_tcr',
             'digit4_is_in_edge', 'distance_value', 'num_bonds',
            ]



def return_property_by_islarge(temp, prop):
    if prop not in ['distance_value', 'num_bonds']:
        temp = temp[['pdbid', 'residue', 'is_large_atten', prop, ]].copy()
        temp = temp.groupby(['pdbid', 'residue', ]).max().reset_index()
        df = pd.crosstab(temp.is_large_atten, temp[prop], margins=False, )
        
        if len(df) == 0:
            print(temp.pdbid.unique().item(), prop, ' has an error in crosstab')
            return [np.nan, np.nan]
        
        if True not in df.columns:
            df[True] = [0 for _ in range(len(df))]
        if (len(df.columns)!=2) or (len(df.index)!=2):
            return [np.nan, np.nan]
            
        # df_margin = pd.crosstab(temp.is_large_atten, temp[prop], margins=True, normalize=True)
        # oddsr, p = stats.fisher_exact(table=df.to_numpy(), alternative='two-sided')
        return (df[True] / df.sum(axis=1)).sort_index(ascending=False).values
    else:
        temp = temp[['pdbid', 'residue', 'is_large_atten', 'peptide_min','distance_value', 'num_bonds', 'digit4_is_in_edge' ]].copy()
        temp = pd.concat([
            temp.groupby(['residue', ]).max()[['is_large_atten','digit4_is_in_edge']],
            temp.groupby(['residue', ]).min()[['distance_value','peptide_min',]],
            temp.groupby(['residue', ]).mean()[['num_bonds',]],
        ], axis=1).reset_index()

        temp_mean = temp.groupby('is_large_atten').mean()[prop]
        if len(temp_mean)==1:
            return  [np.nan, temp_mean.loc[False]]
        return [temp_mean.loc[True], temp_mean.loc[False]]


def main(args):
    
    PDBENTRIES = pd.read_csv(args.pdblist)["pdbid"].unique().tolist()

    df_distance = pd.read_parquet(args.residue_distances)
    df_explained = pd.read_parquet(args.input_filepath)
    
    if 'pdbid' not in df_explained.columns and 'pdbid_x' in df_explained.columns:
        df_explained['pdbid'] = df_explained['pdbid_x']
    
    temp = df_explained.drop_duplicates(subset='pdbid').copy()
    print("pos pred value_counts = ", (temp.proba>0.5).value_counts())
    print('average precision = ', average_precision_score(temp.proba>0, temp.proba))
    print('precision = ', precision_score(temp.proba>0, temp.proba>0.5))
    print('recall = ', recall_score(temp.proba>0, temp.proba>0.5))

    df_bondinfo = pd.read_parquet(f"../data/{args.datetimehash}__df_bondinfo.parquet")
    df_seq = pd.read_parquet(args.seqfile)

    # Print # of unique PDBs
    print('df_distance.pdbid.nunique() = ',df_distance.pdbid.nunique())
    print('df_explained.pdbid.nunique() = ',df_explained.pdbid.nunique())
    print('df_bondinfo.pdbid.nunique() = ',df_bondinfo.pdbid.nunique())
    print('df_seq.pdbid.nunique() = ',df_seq.pdbid.nunique())

    # Filter out proba<0.5 because we only want to analyze the positive predictions
    df_explained = df_explained[df_explained.proba>0.5]

    if 'tcr_pep_combined' not in df_seq.columns and 'tcr_a' in df_seq.columns:
        df_seq['tcr_pep_combined'] = df_seq['tcr_a'] + ':' + df_seq['tcr_b']  + ':' + df_seq['peptide']

    if 'tcr_pep_combined' not in df_seq.columns and 'tcra' in df_seq.columns:
        df_seq['tcr_pep_combined'] = df_seq['tcra'] + ':' + df_seq['tcrb']  + ':' + df_seq['peptide']


    pdbs_seqs = set(df_seq.pdbid.unique().tolist())
    pdbs_explained = set(df_explained.pdbid.unique().tolist())
    pdbs_bondinfo = set(df_bondinfo.pdbid.unique().tolist())
    pdbs_intersection = list(pdbs_seqs & pdbs_explained & pdbs_bondinfo)
    print('len(pdbs_intersection) = ',len(pdbs_intersection))


    df_seq = df_seq[df_seq.pdbid.isin(pdbs_intersection)]

    # df_seq.drop_duplicates(subset=['tcr_pep_combined'], keep='last', inplace=True)
    df_seq.drop_duplicates(subset=['tcr_pep_combined'], keep='first', inplace=True)

    unique_pdbs_no_seq_dup = df_seq.pdbid.unique().tolist()
    # unique_pdbs_no_seq_dup = ['2VLK', '5WKF', '3PQY', '4MJI', '4P2Q', '2YPL', '1J8H', '4P2R', '5MEN', '3MV8', '4OZF', '3VXR', '3VXS', '4OZG', '5TEZ', '2J8U', '6Q3S', '4JRX', '3VXU', '1U3H', '4JRY', '4Z7V', '4JFE', '4JFD', '3QIU', '2Z31', '2BNR', '3MBE', '4OZH', '2NX5', '5NHT', '4QOK', '5D2L', '1D9K', '4P2O', '5WKH', '6EQB', '2VLR', '6EQA']

    print('unique_pdbs_no_seq_dup = ',unique_pdbs_no_seq_dup)

    df_distance = df_distance[df_distance.pdbid.isin(unique_pdbs_no_seq_dup)]
    df_explained = df_explained[df_explained.pdbid.isin(unique_pdbs_no_seq_dup)]
    df_bondinfo = df_bondinfo[df_bondinfo.pdbid.isin(unique_pdbs_no_seq_dup)]

    distance_tcr_side = pd.merge(df_distance[['pdbid','tcr', 'value']].groupby(['pdbid','tcr'], as_index=False).min(), 
            df_distance, 
            on=['pdbid','tcr', 'value'],
            ).rename(columns={'value':'distance_value', 'peptide':'peptide_min', 'tcr':'residue'})

    print(df_explained.type.unique())
    df_tcr_allhead = pd.merge(df_explained[(df_explained.type=='tcr') | (df_explained.type=='alpha') | (df_explained.type=='beta')],
                            df_bondinfo.query('is_tcr==True'), 
                            how='left', 
                            on=['pdbid', 'residue'], 
                            )


    df_tcr_allhead = pd.merge(df_tcr_allhead, distance_tcr_side, on=['pdbid','residue'], how='left')

    print('len(unique_pdbs_no_seq_dup) = ',len(unique_pdbs_no_seq_dup))
    print('df_tcr_allhead.pdbid.nunique()', df_tcr_allhead.pdbid.nunique())

    table = []
    table_by_pdb = []
    for hhh in range(5):
        print()
        print()
        print('head',hhh, '*'*30)
        df_by_prop = []
        for prop in prop_list:
            vals = []
            for ppp in df_tcr_allhead.pdbid.unique():
                temp = df_tcr_allhead.query('pdbid==@ppp').copy()
                temp['tcr'] = temp['residue'].values
                if hhh==4:
                    temp['is_large_atten'] = temp[f'head_all'].values
                else:
                    temp['is_large_atten'] = temp[f'head_{hhh}'].values
                vals.append(return_property_by_islarge(temp, prop))
            vals = np.array(vals)

            df_minimum = pd.DataFrame(vals, index=df_tcr_allhead.pdbid.unique(), columns=[f'Large {prop}', f'Small {prop}'])
            
            ttest_result = stats.ttest_rel(vals[:,0], vals[:,1], nan_policy='omit')
            means = np.nanmean(vals, axis=0).tolist()
            stds = np.nanstd(vals, axis=0)
            if np.isnan(means[0]):
                assert False
            
            print('  prop', prop)
            print(f"\t mean+std of large = {means[0]}+-{stds[0]}", )
            print(f"\t mean+std of small = {means[1]}+-{stds[1]}", )
            # print("std of (large, small) = ", )
            print(ttest_result)
            if ttest_result[1]<0.05:
                print(f'\t *** Significant! head={hhh}, p={ttest_result[1]}, prop={prop}')
            else:
                print('\t - not significant...')
                    
            table.append(['Proportion ' + prop if 'is_' in prop else prop, 
                        f"{means[0]:.4f}+-{stds[0]:.4f}", 
                        f"{means[1]:.4f}+-{stds[1]:.4f}", 
                        ttest_result[1],
                        hhh if hhh!=4 else 'all']
                        )
            df_by_prop.append(df_minimum)
        table_by_pdb.append(pd.concat(df_by_prop,axis=1).assign(head=hhh if hhh!=4 else 'all'))

    df = pd.DataFrame(table, 
                columns=['property', 'Large Atten Mean. Mean(STD)', 'Small Atten. Mean(STD)', 'P Value', 'Head'])
    df.loc[df['P Value'] < 0.10, ' '] = '*'
    df.loc[df['P Value'] < 0.05, ' '] = '***'
    df.loc[df['P Value'] >= 0.10, ' '] = ''
    df[' '].fillna(' ')
    df.to_csv(args.output_statspath, index=False)
    table_by_pdb = pd.concat(table_by_pdb, axis=0)
    table_by_pdb.reset_index().to_csv(args.output_statspath.replace('.csv', '__by_pdbid.csv'), index=False)
    print(args.output_statspath, 'saved')
    print(args.output_statspath.replace('.csv', '__by_pdbid.csv'), 'saved')


if __name__ == "__main__":
    """
    python stats_test.py --seqfile ../data/pdb_complex_sequences.parquet \
        --checkpointsjson ../hpo_params/checkpoints.json \
            --input_filepath ../data/pdb_complex_sequences_entire_crossatten__explained.parquet \
                --pdblist ../data/pdblist.csv \
                    --residue_distances ./../data/20230828_015709__residue_distances.parquet \
                        --datetimehash 20230828_015709

    python stats_test.py --seqfile ../data/pdb_complex_sequencesV2.parquet \
        --checkpointsjson ../hpo_params/checkpoints.json \
            --input_filepath ../data/pdb_complex_sequencesV2_entire_self__explained.parquet \
                --pdblist ../data/pdblist.csv \
                    --residue_distances ./../data/20230828_015709__residue_distances.parquet \
                        --datetimehash 20230828_015709

    python stats_test.py --seqfile ../data/pdb_complex_sequencesV2.parquet \
        --checkpointsjson ../hpo_params/checkpoints.json \
            --input_filepath ../data/pdb_complex_sequencesV2_entire_cross_newemb__explained.parquet \
                --pdblist ../data/pdblist.csv \
                    --residue_distances ./../data/20230828_015709__residue_distances.parquet \
                        --datetimehash 20230828_015709

    python stats_test.py --seqfile ../data/pdb_complex_sequencesV2.parquet \
        --checkpointsjson ../hpo_params/checkpoints.json \
            --input_filepath ../data/pdb_complex_sequencesV2_entire_self_newemb__explained.parquet \
                --pdblist ../data/pdblist.csv \
                    --residue_distances ./../data/20230828_015709__residue_distances.parquet \
                        --datetimehash 20230828_015709

    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seqfile", type=str, default="../data/pdb_complex_sequences.parquet")
    parser.add_argument("--checkpointsjson", type=str, default="../hpo_params/checkpoints.json")
    parser.add_argument(
        "--input_filepath", type=str, 
        default="../data/pdb_complex_sequences_entire_crossatten__explained.parquet"
    )
    parser.add_argument("--pdblist", type=str, default='../data/pdblist.csv')  # or ../data/pdblist.csv
    parser.add_argument("--residue_distances", type=str, 
                       # default=f"./../data/{datetimehash}__residue_distances.parquet"
                       )
    parser.add_argument("--datetimehash", type=str, default='20230828_015709')

    args = parser.parse_args()

    args.residue_distances = f"./../data/{args.datetimehash}__residue_distances.parquet"
    args.output_statspath = args.input_filepath.replace(".parquet", f"__stats{args.datetimehash}.csv")
    main(args)

