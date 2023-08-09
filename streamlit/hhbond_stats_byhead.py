import streamlit as st
import functools
import os, sys
import pickle
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy import stats
sys.path.append("./analysis")
from pdb_util import AACODES_DICT, AACODES_DICT_upper
from collections import defaultdict
from tqdm import tqdm
from collections import defaultdict
sys.path.append("./streamlit/")
sys.path.append("./")

from streamlit_utils import DICT_PDBID_2_Atten12, DICT_PDBID_2_MELTDIST, STD_THH, \
    DICT_PDBID_2_model_out, DICT_PDBID_2_CHAINNAMES, DICT_PDBID_2_CDRS, DICT_PDBID_2_RESIDUES, \
    POSITIVE_PRED_IDS, HEAD_COUNT, get_mapping_2_4digits, read_hhb_file

import matplotlib.pyplot as plt

NUM2TCRORPEP = {0:'alpha', 1:'beta', 2:'peptide'}


def read_hhb_text_and_find_chain(text, chainsletters):
    ress_atoms = text[:45].replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').split(' ')
    donor_res = ress_atoms[0]
    acceptor_res = ress_atoms[2]
    if donor_res.split('/')[1][:1] in chainsletters or \
            acceptor_res.split('/')[1][:1] in chainsletters:
        return text


# def hhbond_analysis_single2():
def add_resname(r):
    return AACODES_DICT_upper.get(r.get_resname(), 'X')

def add_resseq(r):
    return r.get_full_id()[3][1]

def add_4digits(r):
    res = r.get_full_id()[2]
    digits = r.get_full_id()[3][1]
    return f'{res}{digits:004d}'



def add_restype_cdr(ser):
    if ser['CDR']=='Not_CDR':
        return 3
    else:
        if ser['LargeAtten_CDR']:
            return 1
        else:
            return 2

def add_restype_pep(ser):
    if ser['Peptide']=='Not_Peptide':
        return None
    else:
        if ser['LargeAtten_Peptide']:
            return 4
        else:
            return 5

from streamlit_utils import strong_givenPEP_distributedTCR_by_head, \
    strong_givenTCR_distributedPEP_by_head, general_givenPEP_distributedTCR_by_head, general_givenTCR_distributedPEP_by_head

def hhbond_stats_byhead():
    st.write("# Statistics of 10 types of hhbond ")

    df_general_pdbids = pd.concat(
        [pd.concat([pd.concat(general_givenPEP_distributedTCR_by_head[iii])]) for iii in range(4)])
    st.write("pdbid.nunique 1 = ", df_general_pdbids.pdbid.nunique())

    st.write("pdbid.nunique 2 = ", df_general_pdbids.drop_duplicates(subset=['tcr','peptide','pdbid']).pdbid.nunique())

    #st.dataframe(df_general_pdbids)

    #########################################################################
    # st.write("# Attention file pre-processing: ")
    for headi in range(4):

        st.write('# head', headi)
        df_strong_attention = pd.concat([pd.concat(strong_givenPEP_distributedTCR_by_head[headi])])
        df_strong_attention = df_strong_attention.drop_duplicates(subset=['tcr','pdbid'])
        df_strong_attention_pep = pd.concat([pd.concat(strong_givenTCR_distributedPEP_by_head[headi])])
        df_strong_attention_pep = df_strong_attention_pep.drop_duplicates(subset=['peptide','pdbid'])
        df_normal_attention = pd.concat([pd.concat(general_givenPEP_distributedTCR_by_head[headi])])
        df_normal_attention = df_normal_attention.drop_duplicates(subset=['tcr','pdbid'])

        isin_strong = df_normal_attention[['tcr','pdbid']].apply(tuple, axis=1).isin(df_strong_attention[['tcr','pdbid']].apply(tuple, axis=1))
        df_normal_attention = df_normal_attention[~isin_strong]

        st.write('# Ratio of AA Residue')
        st.write('### when attention is large')
        large_atten = df_strong_attention['tcr'].apply(lambda x: x[0]).value_counts()
        if 'H' not in large_atten.index:
            large_atten = large_atten.append(pd.Series({'H':0}))
        aa_residues = df_normal_attention['tcr'].apply(lambda x: x[0]).unique()
        aa_residues_selected = [aa for aa in aa_residues if aa in large_atten.index]
        st.write((large_atten.loc[aa_residues_selected]/large_atten.sum()))
        st.write('### when attention is small')
        small_atten = df_normal_attention['tcr'].apply(lambda x: x[0]).value_counts()
        st.write(small_atten.loc[aa_residues]/small_atten.sum())

        for ind in small_atten.index:
            if ind not in large_atten.index:
                large_atten[ind] = 0
        chi2_stat, p_val, dof, ex  = stats.chi2_contingency([large_atten, small_atten])

        fig, ax = plt.subplots()
        (large_atten.loc[aa_residues]/large_atten.sum()).plot.bar(ax=ax, color='r', alpha=0.2)
        (small_atten.loc[aa_residues]/small_atten.sum()).plot.bar(ax=ax, alpha=0.2)
        
        ax.legend(['large','small'])
        st.pyplot(fig)
        st.write('chisquare_ttest_result:')

        st.write((chi2_stat, p_val))
        if p_val<0.05:
            st.write('#### Significant!')
        else:
            st.write('#### not significant...')
        

        large_atten_types = (large_atten.loc[aa_residues]/large_atten.sum())
        small_atten_types = (small_atten.loc[aa_residues]/small_atten.sum())
        DICT_restype_ratio = {}
        for t in small_atten_types.index:
            if t in large_atten_types.index:
                # st.write(t, large_atten_types.loc[t]/small_atten_types.loc[t])
                DICT_restype_ratio[t] = large_atten_types.loc[t]/small_atten_types.loc[t]
            else: DICT_restype_ratio[t] = 0.0
        st.write("DICT_restype_ratio")
        # st.write(DICT_restype_ratio)
        fig, ax = plt.subplots()
        pd.Series(DICT_restype_ratio).plot.bar(ax=ax)
        st.pyplot(fig)
        

        ## Dedup pdbid_list
        unique_pdbid_list = df_strong_attention['pdbid'].unique()
        st.write("PDB IDs before deleting duplicated entries:", len(unique_pdbid_list), "\n",
                ", ".join(unique_pdbid_list.tolist()) )
        df_cdrs_and_peps = pd.DataFrame([tuple([DICT_PDBID_2_Atten12[p][0][0].index.tolist() + DICT_PDBID_2_Atten12[p][0][0].columns.tolist()])
                                         for p in unique_pdbid_list], columns=['pairs'], index=unique_pdbid_list)['pairs'].apply(tuple)
        unique_pdbid_list = df_cdrs_and_peps.drop_duplicates().index
        #st.dataframe(df_cdrs_and_peps)
        st.write("Duplicated count:", df_cdrs_and_peps.shape[0] - len(df_cdrs_and_peps.unique()))
        st.write("Unique PDB count:", len(unique_pdbid_list))

        ######################################################################
        ### PDBID FOR LOOP ####################################################
        vals_dict = defaultdict(list)
        for pdbid in tqdm(reversed(unique_pdbid_list)):

            residues = DICT_PDBID_2_RESIDUES[pdbid]
            chains = DICT_PDBID_2_CHAINNAMES[pdbid.upper()]
            chainsletters = ''.join(chains)
            dfhhb = read_hhb_file(pdbid)

            donors_n_acceptors = [read_hhb_text_and_find_chain(row.item(), chainsletters) for i, row in dfhhb.iterrows()]
            donors_n_acceptors = [a for a in donors_n_acceptors if a is not None]

            bond_tuple_fromto = []
            for da in donors_n_acceptors:
                a = da.replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').split(' ')
                from_d = a[0].split('/')[1].split('-')[0][:5]
                to_d = a[2].split('/')[1].split('-')[0][:5]
                bond_tuple_fromto.append((from_d, to_d))
            anyhbond_related_4digits = list(set([tup[0] for tup in bond_tuple_fromto] + [tup[1] for tup in bond_tuple_fromto]))

            CDRname_2_4digits, Peptide_2_4digits, bond_list = get_mapping_2_4digits(donors_n_acceptors, pdbid)

            D4digits_2_CDR = {v:k for k,v in CDRname_2_4digits.items()}
            D4digits_2_Peptide = {v:k for k,v in Peptide_2_4digits.items()}

            df_strong_temp = df_strong_attention[df_strong_attention.pdbid == pdbid].copy()
            df_strong_temp['tcr_residue'] = df_strong_temp['tcr'].map(CDRname_2_4digits)
            df_strong_temp['pep_residue'] = df_strong_temp['peptide'].map(Peptide_2_4digits)

            df_strong_temp_peptide = df_strong_attention_pep[df_strong_attention_pep.pdbid == pdbid].copy()

            def add_info(df):
                df['AA resname'] = df[0].apply(add_resname)
                df['Resseq'] = df[0].apply(add_resseq)
                df['4digits'] = df[0].apply(add_4digits)
                df['CDR'] = df['4digits'].map(D4digits_2_CDR)
                df['CDR'] = df['CDR'].fillna('Not_CDR')
                df['LargeAtten_CDR'] = df['CDR'].isin(df_strong_temp['tcr'].values)
                df['Peptide'] = df['4digits'].map(D4digits_2_Peptide)
                df['Peptide'] = df['Peptide'].fillna('Not_Peptide')
                df['LargeAtten_Peptide'] = df['Peptide'].isin(df_strong_temp_peptide['peptide'].values)
                df['has_bond'] = df['4digits'].isin(anyhbond_related_4digits)
                return df

            df_alpha = pd.DataFrame([residues[0]]).T
            df_beta = pd.DataFrame([residues[1]]).T
            df_pep = pd.DataFrame([residues[2]]).T
            df_alpha = add_info(df_alpha)
            df_alpha['chain_name'] = chains[0]
            df_beta = add_info(df_beta)
            df_beta['chain_name'] = chains[1]
            df_pep = add_info(df_pep)
            df_pep['chain_name'] = chains[2]

            df_alpha['restype'] = df_alpha.apply(add_restype_cdr, axis=1)
            df_beta['restype'] = df_beta.apply(add_restype_cdr, axis=1)
            df_pep['restype'] = df_pep.apply(add_restype_pep, axis=1)

            df_very_all = pd.concat([df_pep, df_alpha, df_beta])
            df_bond_all_4digits = pd.DataFrame(anyhbond_related_4digits, columns=['4digits'])
            df_bond_all_4digits = pd.merge(df_bond_all_4digits, df_very_all, on=['4digits'], how='inner')

            def get_acceptor(donor):
                return [a for (d,a) in bond_tuple_fromto if d==donor]

            def get_donor(aceptor):
                return [d for (d,a) in bond_tuple_fromto if a==aceptor]

            df_bond_all_4digits['bonding_to'] = df_bond_all_4digits['4digits'].apply(get_acceptor)
            df_bond_all_4digits['bonding_from'] = df_bond_all_4digits['4digits'].apply(get_donor)
            df_bond_all_4digits['bonding_tofrom'] = df_bond_all_4digits['bonding_to'] + df_bond_all_4digits['bonding_from']
            BOND_MAPPING = {k:v for k,v in zip(df_bond_all_4digits['4digits'], df_bond_all_4digits['bonding_tofrom'])}
            df_very_all['bonding_tofrom'] = [np.array([]) for _ in range(len(df_very_all))]
            df_very_all['bonding_tofrom'] = df_very_all['4digits'].map(BOND_MAPPING)
            df_very_all.loc[df_very_all['bonding_tofrom'].isna(), 'bonding_tofrom'] = [[None] for _ in range(df_very_all['bonding_tofrom'].isna().sum())]


            ######################################################################
            # Main Stats
            ######################################################################

            # CDRがLarge Attentionの時に、CDR以外のAnyTCRとBondを作る?
            not_cdr_tcr = df_very_all[df_very_all['restype']==3]['4digits'].values
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            large_count = (largeatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in not_cdr_tcr) else 0 for d in x]))).sum()
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            small_count = (smallatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in not_cdr_tcr) else 0 for d in x]))).sum()
            a = None if len(largeatten_cdr)==0 else large_count / len(largeatten_cdr)
            b = None if len(smallatten_cdr)==0 else small_count / len(smallatten_cdr)
            vals_dict['p_notCDR_TCR_Bond'].append([a, b])

            # CDRがLarge Attentionの時に、Peptide とBondを作る?
            peptide_4digits = df_very_all.query('restype==4 or restype==5')['4digits'].values
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            large_count = (largeatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in peptide_4digits) else 0 for d in x]))).sum()
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            small_count = (smallatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in peptide_4digits) else 0 for d in x]))).sum()
            a = None if len(largeatten_cdr)==0 else large_count / len(largeatten_cdr)
            b = None if len(smallatten_cdr)==0 else small_count / len(smallatten_cdr)
            vals_dict['p_peptideBond'].append([a, b])


            # Large Attentionの時に、CDR Chain?
            cdr_4digits = df_very_all.query('restype==1 or restype==2')['4digits'].values
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            large_count = (largeatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in cdr_4digits) else 0 for d in x]))).sum()
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            small_count = (smallatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in cdr_4digits) else 0 for d in x]))).sum()
            a = None if len(largeatten_cdr)==0 else large_count / len(largeatten_cdr)
            b = None if len(smallatten_cdr)==0 else small_count / len(smallatten_cdr)
            vals_dict['p_CDRBond'].append([a, b])

            # Large Attentionの時に、CDR Chain or Peptide Bond?
            pep_cdr_4digits = df_very_all.query('restype==1 or restype==2 or restype==4 or restype==5')['4digits'].values
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            large_count = (largeatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in pep_cdr_4digits) else 0 for d in x]))).sum()
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            small_count = (smallatten_cdr['bonding_tofrom'].apply(lambda x: any([1 if (d in pep_cdr_4digits) else 0 for d in x]))).sum()
            a = None if len(largeatten_cdr)==0 else large_count / len(largeatten_cdr)
            b = None if len(smallatten_cdr)==0 else small_count / len(smallatten_cdr)
            vals_dict['p_CDRBond_or_peptideBond'].append([a, b])

            # Large Attentionの時に、Any Hbond ?
            a = None if len(largeatten_cdr)==0 else len([c for c in largeatten_cdr['4digits'] if c in BOND_MAPPING.keys()]) / len(largeatten_cdr)
            b = None if len(smallatten_cdr)==0 else len([c for c in smallatten_cdr['4digits'] if c in BOND_MAPPING.keys()]) / len(smallatten_cdr)
            vals_dict['p_anyHbond'].append([a, b])

            # Large Attentionの時、Num_hbond (Average for each pdbid
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            a = None if len(largeatten_cdr)==0 else (largeatten_cdr['bonding_tofrom'].apply(len)).mean()
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            b = None if len(smallatten_cdr)==0 else (smallatten_cdr['bonding_tofrom'].apply(len)).mean()
            vals_dict['num_HBondMean_per_residue (Ave over pdbid)'].append([a, b])

            # Large Attentionの時、Num_hbond (Sum for each pdbid
            # largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            # a = None if len(largeatten_cdr)==0 else (largeatten_cdr['bonding_tofrom'].apply(len).apply(int)).sum()
            # smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            # b = None if len(smallatten_cdr)==0 else (smallatten_cdr['bonding_tofrom'].apply(len).apply(int)).sum()
            # vals_dict['num_HBondSum_per_CDRofPdbid'].append([a, b])

            # Large Attentionの時、複数のHbond?
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            a = None if len(largeatten_cdr)==0 else (largeatten_cdr['bonding_tofrom'].apply(len)>1).sum() / len(largeatten_cdr)
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            b = None if len(smallatten_cdr)==0 else  (smallatten_cdr['bonding_tofrom'].apply(len)>1).sum() / len(smallatten_cdr)
            vals_dict['p_multiple_HBond'].append([a, b])

            # Large Attentionの時、More Than 2 のHbond?
            MORETHAN = 2
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            a = None if len(largeatten_cdr)==0 else (largeatten_cdr['bonding_tofrom'].apply(len)>MORETHAN).sum() / len(largeatten_cdr)
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            b = None if len(smallatten_cdr)==0 else  (smallatten_cdr['bonding_tofrom'].apply(len)>MORETHAN).sum() / len(smallatten_cdr)
            vals_dict[f'p_morethan{MORETHAN}_HBond'].append([a, b])

            # Large Attentionの時、More Than 3 のHbond?
            MORETHAN = 3
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            a = None if len(largeatten_cdr)==0 else (largeatten_cdr['bonding_tofrom'].apply(len)>MORETHAN).sum() / len(largeatten_cdr)
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            b = None if len(smallatten_cdr)==0 else  (smallatten_cdr['bonding_tofrom'].apply(len)>MORETHAN).sum() / len(smallatten_cdr)
            vals_dict[f'p_morethan{MORETHAN}_HBond'].append([a, b])

            # Large Attentionの時、More Than 4 のHbond?
            MORETHAN = 4
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            a = None if len(largeatten_cdr)==0 else (largeatten_cdr['bonding_tofrom'].apply(len)>MORETHAN).sum() / len(largeatten_cdr)
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            b = None if len(smallatten_cdr)==0 else  (smallatten_cdr['bonding_tofrom'].apply(len)>MORETHAN).sum() / len(smallatten_cdr)
            vals_dict[f'p_morethan{MORETHAN}_HBond'].append([a, b])

            # Large Attentionの時、根本 に?
            EDGENUM = 4
            large_atten_pos = largeatten_cdr['CDR'].str.split('_').apply(lambda x: x[1]).values.astype(int)
            pos_all = df_very_all[df_very_all['CDR']!='Not_CDR']['CDR'].str.split('_').apply(lambda x: x[1]).values.astype(int)
            pos_not_in_cdr = [i for i in range(len(pos_all)) if i not in pos_all][0]
            beginnings_endings_cdrs = list(range(EDGENUM)) + pos_all[-EDGENUM:].tolist() + \
                                      [pos_not_in_cdr-i-1 for i in range(EDGENUM)] + \
                                      [pos_not_in_cdr+i+1 for i in range(EDGENUM)]

            b = len(set(beginnings_endings_cdrs)) / (len(pos_all))
            if len(largeatten_cdr)>0:
                a = sum([l in beginnings_endings_cdrs for l in large_atten_pos]) / len(large_atten_pos)
            else:
                a = None
            vals_dict[f'p_having_edge_positions'].append([a, b])

            # Large Attentionの時、自分以外のTCR Chain?
            pepchain = chains[-1].split(', ')
            def count_to_other_cdr(row):
                tcrchain = row['4digits'][0]
                others_having_pep_tcr = [b[0] for b in row['bonding_tofrom']]
                others_having_pep_tcr = ', '.join(others_having_pep_tcr)
                for p in pepchain:
                    others_having_pep_tcr = others_having_pep_tcr.replace(p,'')
                others_having_tcr = others_having_pep_tcr
                others_having_tcr = others_having_tcr.replace(tcrchain, '')
                others_having_tcr = others_having_tcr.replace(' ','')
                others_having_tcr = others_having_tcr.split(',')
                return others_having_tcr

            def is_to_other_cdr(row):
                return len(count_to_other_cdr(row)) > 0

            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            smallatten_cdr['p_hbond_to_notself_TCRchain'] = smallatten_cdr.apply(is_to_other_cdr, axis=1)
            b = smallatten_cdr['p_hbond_to_notself_TCRchain'].sum() / len(smallatten_cdr)
            if len(largeatten_cdr)>0:
                largeatten_cdr['p_hbond_to_notself_TCRchain'] = largeatten_cdr.apply(is_to_other_cdr, axis=1)
                a = largeatten_cdr['p_hbond_to_notself_TCRchain'].sum() /len(largeatten_cdr)
                vals_dict['p_hbond_to_notself_TCRchain'].append([a, b])
            else:
                vals_dict['p_hbond_to_notself_TCRchain'].append([None, b])

            # Large Attentionの時、自分以外の TCRChain との数 mean？
            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            smallatten_cdr['count_bonds_to_notself_TCRchain'] = smallatten_cdr.apply(count_to_other_cdr, axis=1).apply(len)
            b = smallatten_cdr['count_bonds_to_notself_TCRchain'].mean()
            if len(largeatten_cdr) > 0:
                largeatten_cdr['count_bonds_to_notself_TCRchain'] = largeatten_cdr.apply(count_to_other_cdr, axis=1).apply(len)
                a = largeatten_cdr['count_bonds_to_notself_TCRchain'].mean()
                vals_dict['count_bonds_to_notself_TCRchain'].append([a, b])
            else:
                vals_dict['count_bonds_to_notself_TCRchain'].append([None, b])

            # Large Attentionの時、自分のTCR Chain?
            def p_to_myself_tcr(row):
                tcrchain = row['4digits'][0]
                others_having_pep_tcr = [b[0] for b in row['bonding_tofrom']]
                others_having_pep_tcr = ', '.join(others_having_pep_tcr)
                for p in pepchain:
                    others_having_pep_tcr = others_having_pep_tcr.replace(p,'')
                return tcrchain in others_having_pep_tcr

            largeatten_cdr = df_very_all[df_very_all['restype']==1].dropna(subset=['bonding_tofrom'])
            smallatten_cdr = df_very_all[df_very_all['restype']==2].dropna(subset=['bonding_tofrom'])
            smallatten_cdr['p_to_myself_tcr'] = smallatten_cdr.apply(p_to_myself_tcr, axis=1)
            small_count = smallatten_cdr['p_to_myself_tcr'].sum()
            b = None if len(smallatten_cdr)==0 else small_count / len(smallatten_cdr)
            if len(largeatten_cdr)>0:
                largeatten_cdr['p_to_myself_tcr'] = largeatten_cdr.apply(p_to_myself_tcr, axis=1)
                large_count = largeatten_cdr['p_to_myself_tcr'].sum()
                a = None if len(largeatten_cdr)==0 else large_count / len(largeatten_cdr)
                vals_dict['p_to_myself_tcr'].append([a, b])
            else:
                vals_dict['p_to_myself_tcr'].append([None, b])


        vis_df_copy = pd.DataFrame(vals_dict, index=unique_pdbid_list)
        for c in vis_df_copy.columns:
            vals = vis_df_copy[c]
            st.write(f'# {c}')
            st.write(f'### head {headi}')
            mean_large_attention = np.mean([float(v[0]) for v in vals if v[0] is not None])
            mean_small_attention = np.mean([float(v[1]) for v in vals if v[0] is not None])
            st.write(f'mean_large_attention {mean_large_attention:.4f}, mean_small_attention {mean_small_attention:.4f}')
            st.write(stats.ttest_rel([float(v[0]) for v in vals if v[0] is not None],
                                     [float(v[1]) for v in vals if v[0] is not None]))
            st.write('fill nan as 0')
            mean_large_attention = np.mean([float(v[0]) if v[0] is not None else 0 for v in vals])
            mean_small_attention = np.mean([float(v[1]) if v[0] is not None else 0 for v in vals])
            st.write(f'mean_large_attention {mean_large_attention:.4f}, mean_small_attention {mean_small_attention:.4f}')
            ttest_result = stats.ttest_rel(
                [float(v[0]) if v[0] is not None else 0 for v in vals],
                [float(v[1]) if v[0] is not None else 0 for v in vals])
            st.write(ttest_result)
            if ttest_result[1]<0.05:
                st.write('#### Significant!')
            else:
                st.write('#### not significant...')

        st.write("# Box Plot: head =",headi)
        N =3
        fig, axs = plt.subplots(N, 1+len(vis_df_copy.columns)//N , figsize=(5 * (len(vis_df_copy.columns)//N), 4*N))
        # fig.figure()
        vis_df_copy = vis_df_copy.rename(columns={'num_HBondMean_per_residue (Ave over pdbid)':'num_HBondMean'})
        for ax, col in zip(axs.ravel(), vis_df_copy.columns):
            ax.set_title(col)
            data = vis_df_copy[col].apply(pd.Series).dropna()
            ax.boxplot(data, whis=[5, 95], showmeans=True)
            ax.set_xticklabels(['large attn', 'small attn'])
        st.pyplot(fig)
        
        st.write("# DataFrame by PDB ID")
        st.write('# Statistics by pdbid')
        st.write('left: large attention, right: small attention')
        st.write("pdb ids = ", vis_df_copy.index.tolist())
        for c in vis_df_copy.columns:
            vis_df_copy[c] = vis_df_copy[c].apply(lambda tup: list([f'{tup[0]:.3f}' if tup[0] is not None else 'None',
                                                                    f'{tup[1]:.3f}' if tup[1] is not None else 'None'])
                                                  )
        st.dataframe(vis_df_copy)

