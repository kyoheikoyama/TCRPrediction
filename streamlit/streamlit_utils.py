import streamlit as st
import functools
import os, sys
import pickle
from tqdm import tqdm
import pandas as pd

sys.path.append("../analysis")
sys.path.append("../")
sys.path.append("./")
from analysis_util import calc_melt_df
from pdb_util import AACODES_DICT, AACODES_DICT_upper

AACODES_DICT.update(AACODES_DICT_upper)
AACODES_DICT_upper.update(AACODES_DICT)

##################################################################################
## Globals #######################################################################
STD_THH = 5.0
HEAD_COUNT = 4

print('loading....')


@functools.lru_cache(maxsize=200)
def pickleload(p): return pickle.load(open(p, "rb"))

# @st.cache(max_entries=200, allow_output_mutation=True)
# def pickleload(p): return pickle.load(open(p, "rb"))


DATETIME = '20220211_191954'

DICT_PDBID_2_Atten12 = pickleload(
    f"/Users/kyoheikoyama/workspace/tcrpred/analysis/DICT_PDB_Result/{DATETIME}_DICT_PDBID_2_Atten12.pickle"
)
DICT_PDBID_2_MELTDIST = pickleload(
    f"/Users/kyoheikoyama/workspace/tcrpred/analysis/DICT_PDB_Result/{DATETIME}_DICT_PDBID_2_MELTDIST.pickle"
)
DICT_PDBID_2_model_out = pickleload(
    f"/Users/kyoheikoyama/workspace/tcrpred/analysis/DICT_PDB_Result/{DATETIME}_DICT_PDBID_2_model_out.pickle"
)

DICT_PDBID_2_CHAINNAMES = pickleload(
    f"/Users/kyoheikoyama/workspace/tcrpred/analysis/DICT_PDB_Result/{DATETIME}_DICT_PDBID_2_CHAINNAMES.pickle"
)

DICT_PDBID_2_CDRS = pickleload(
    f"/Users/kyoheikoyama/workspace/tcrpred/analysis/DICT_PDB_Result/{DATETIME}_DICT_PDBID_2_CDRS.pickle"
)

DICT_PDBID_2_RESIDUES = pickleload(
    f"/Users/kyoheikoyama/workspace/tcrpred/analysis/DICT_PDB_Result/{DATETIME}_DICT_PDBID_2_RESIDUES.pickle"
)
POSITIVE_PRED_IDS = [
    k
    for k, v in DICT_PDBID_2_model_out.items()
    if DICT_PDBID_2_model_out[k][-1] > 0.5
]
ABP_2_NUMBER = {"alpha": 0, "beta": 1, "peptide": 2}
DIR_LIGPLOTTMP = "/Users/kyoheikoyama/workspace/ligplottmp"
HHB_COLUMNS = [
    "donor_res_code",
    "donor_chain_identifier",
    "donor_res_seq_num",
    "donor_res_atom_name",
    "acceptor_res_code",
    "acceptor_chain_identifier",
    "acceptor_res_seq_num",
    "acceptor_res_atom_name",
    "hydrogen_bond_distance",
    "plus_minus",
]

print('ending... loading....')


##################################################################################

# @functools.lru_cache(maxsize=4)
# # @st.cache
# @functools.lru_cache(maxsize=200)
def get_attention_and_hhb_relationship(strong_atten=False):
    givenPEP_distributedTCR_by_head = {}
    givenTCR_distributedPEP_by_head = {}
    for hi in tqdm(range(HEAD_COUNT)):
        givenTCR_distributedPEP_by_head[hi] = []
        givenPEP_distributedTCR_by_head[hi] = []
        for p, (a1_by_head, a2_by_head) in DICT_PDBID_2_Atten12.items():
            if p not in POSITIVE_PRED_IDS:
                continue
            a1, a2 = calc_melt_df(
                DICT_PDBID_2_MELTDIST[p], a1_by_head[hi], a2_by_head[hi]
            )
            a1["pdbid"] = p
            a2["pdbid"] = p
            if strong_atten == True:
                temp1 = a1[
                    (
                            a1["value_y"]
                            > (a1["value_y"].mean() + STD_THH * a1["value_y"].std())
                    )
                ].copy()
                temp2 = a2[
                    (
                            a2["value_y"]
                            > (a2["value_y"].mean() + STD_THH * a2["value_y"].std())
                    )
                ].copy()
            else:
                temp1 = a1[
                    (
                            a1["value_y"]
                            <= (a1["value_y"].mean() + STD_THH * a1["value_y"].std())
                    )
                ].copy()
                temp2 = a2[
                    (
                            a2["value_y"]
                            <= (a2["value_y"].mean() + STD_THH * a2["value_y"].std())
                    )
                ].copy()
            temp1["head"] = hi
            temp2["head"] = hi
            givenPEP_distributedTCR_by_head[hi] += [temp2]
            givenTCR_distributedPEP_by_head[hi] += [temp1]
    return givenPEP_distributedTCR_by_head, givenTCR_distributedPEP_by_head


def cleanup(s):
    s = s.replace("  ", " ").replace(" ", ",").replace(",,", ",").replace(",,", ",")
    return pd.Series(s.split(","))


def get_pos(pdbid, num, row, tseqnum="donor_res_seq_num", dict_pdbid_2_cdrresidues={}):
    if int(row[tseqnum]) in [
        c.get_id()[1] for c in dict_pdbid_2_cdrresidues[pdbid][num]
    ]:
        pos = [int(c.get_id()[1]) for c in dict_pdbid_2_cdrresidues[pdbid][num]].index(
            int(row[tseqnum])
        )
        if num == 1:  # beta needs another count for pos
            return pos + 1 + len(dict_pdbid_2_cdrresidues[pdbid][0])
        else:
            return pos
    else:
        return None

# @functools.lru_cache(maxsize=100)
# # @st.cache
# @functools.lru_cache(maxsize=200)
def get_df(pdbid):
    f = f"pdb{pdbid.lower()}"
    df = pd.read_table(os.path.join(DIR_LIGPLOTTMP, f, "ligplot.hhb"), skiprows=2)
    df = df.iloc[:, 0].apply(cleanup)
    df.columns = HHB_COLUMNS[: len(df.columns)]
    df["AA_code_donor"] = df["donor_res_code"].map(AACODES_DICT)
    df["AA_code_acceptor"] = df["acceptor_res_code"].map(AACODES_DICT)
    return df


def add_colname(df_hhb, pdbid, dict_pdbid_2_cdrresidues):
    alpha_len = len(dict_pdbid_2_cdrresidues[pdbid][0])
    colname_donor = []
    for i, row in df_hhb.iterrows():
        donor_num = ABP_2_NUMBER[row["abp_donor"]]
        donor_pos = get_pos(
            pdbid, donor_num, row, "donor_res_seq_num", dict_pdbid_2_cdrresidues
        )
        if donor_pos is None:  # the h-bond region is out of scope. (Not in CDR.)
            colname_donor.append("")
        else:
            donor_name = row["AA_code_donor"] + "_" + str(donor_pos)
            colname_donor.append(donor_name)
    colname_acceptor = []
    for i, row in df_hhb.iterrows():
        acceptor_num = ABP_2_NUMBER[row["abp_acceptor"]]
        acceptor_pos = get_pos(
            pdbid, acceptor_num, row, "acceptor_res_seq_num", dict_pdbid_2_cdrresidues
        )
        if acceptor_pos is None:  # the h-bond region is out of scope. (Not in CDR.)
            colname_acceptor.append(None)
        else:
            acceptor_name = row["AA_code_acceptor"] + "_" + str(acceptor_pos)
            colname_acceptor.append(acceptor_name)
    df_hhb["colname_donor"] = colname_donor
    df_hhb["colname_acceptor"] = colname_acceptor
    return df_hhb

# # @st.cache
# @functools.lru_cache()
def get_DICT_PDBID_2_dfhhb():
    DICT_PDBID_2_dfhhb = {}
    for iii, fname in tqdm(enumerate(sorted(os.listdir(DIR_LIGPLOTTMP)))):

        pdbid = fname.replace("pdb", "").upper()

        if "hbadd_hbplus_result" in fname:
            continue

        if pdbid=='5JZI' or pdbid=='4QRR' or pdbid=='5JHD' or pdbid=='2OL3':
            print(f'{pdbid} has only delta-chain and beta-chain' )
            continue

        df_hhb = get_df(pdbid)
        alphaname, betaname, peptidename = DICT_PDBID_2_CHAINNAMES[pdbid]
        alphaname_list, betaname_list, peptidename_list = (
            alphaname.split(", "),
            betaname.split(", "),
            peptidename.split(", "),
        )
        assert len(betaname_list) != 0

        chain2abp = {}
        for a in alphaname_list:
            chain2abp[a] = "alpha"
        for b in betaname_list:
            chain2abp[b] = "beta"
        for p in peptidename_list:
            chain2abp[p] = "peptide"

        df_hhb["abp_donor"] = df_hhb["donor_chain_identifier"].map(chain2abp)
        df_hhb["abp_acceptor"] = df_hhb["acceptor_chain_identifier"].map(chain2abp)

        df_hhb.dropna(subset=["abp_donor"], inplace=True)
        df_hhb.dropna(subset=["abp_acceptor"], inplace=True)
        df_hhb = add_colname(df_hhb, pdbid, DICT_PDBID_2_CDRS)
        df_hhb["peptide_col"] = df_hhb.loc[
            (df_hhb["abp_acceptor"] == "peptide"), "colname_acceptor"
        ]
        df_hhb.loc[
            ((df_hhb["abp_donor"] == "peptide") & df_hhb["peptide_col"].isna()),
            "peptide_col",
        ] = df_hhb["colname_donor"]
        df_hhb["beta_col"] = df_hhb.loc[
            (df_hhb["abp_acceptor"] == "beta"), "colname_acceptor"
        ]
        df_hhb.loc[
            ((df_hhb["abp_donor"] == "beta") & df_hhb["beta_col"].isna()), "beta_col"
        ] = df_hhb["colname_donor"]
        df_hhb["alpha_col"] = df_hhb.loc[
            (df_hhb["abp_acceptor"] == "alpha"), "colname_acceptor"
        ]
        df_hhb.loc[
            ((df_hhb["abp_donor"] == "alpha") & df_hhb["alpha_col"].isna()), "alpha_col"
        ] = df_hhb["colname_donor"]
        DICT_PDBID_2_dfhhb[pdbid] = df_hhb
    return DICT_PDBID_2_dfhhb


# @functools.lru_cache(maxsize=100)
def pdbid2residue(pdbid):
    DICT_PDBID_2_Atten12[pdbid]
    return
# # @st.cache

# @functools.lru_cache(maxsize=100)
def read_hhb_file(pdbid):
    root = '/Users/kyoheikoyama/workspace/ligplottmp/hbadd_hbplus_result/'
    return pd.read_table(f'{root}pdb{pdbid.lower()}.hhb', header=None)

def read_hhb_text(text):
    ress_atoms = text[:45].replace('  ',' ').replace('  ',' ').replace('  ',' ').replace('  ',' ').split(' ')
    donor_res = ress_atoms[0]
    donor_atom = ress_atoms[1]
    acceptor_res = ress_atoms[2]
    acceptor_atom = ress_atoms[3]
    da_dist = float(text[45:49])
    donor_mainorside = text[49:50]
    acceptor_mainorside = text[50:51]
    gap = int(text[53:55])
    da_dist_inCAatoms = float(text[56:60])
    Angle_inHydrogen = float(text[60:65])
    Dist_bw_H_n_Acceptor = float(text[66:70])
    smaller_angle_in_A = float(text[70:75])
    smaller_angle_in_D = float(text[76:81])
    count = int(text[81:])
    return {
        'donor_res':donor_res, 'donor_atom':donor_atom,
        'acceptor_res':acceptor_res,'acceptor_atom':acceptor_atom,
        'da_dist':da_dist, 'donor_mainorside':donor_mainorside,
        'acceptor_mainorside':acceptor_mainorside, 'gap':gap,
        'da_dist_inCAatoms':da_dist_inCAatoms, 'Angle_inHydrogen':Angle_inHydrogen,
        'Dist_bw_H_n_Acceptor':Dist_bw_H_n_Acceptor, 'smaller_angle_in_A':smaller_angle_in_A,
        'smaller_angle_in_D':smaller_angle_in_D,
        'count':count,
        'originaltext':text
    }

def process_hhb(df_hhb_inner):
    df_hhb_inner['AA_code_donor'] = df_hhb_inner.donor_res.apply(lambda x: x[-3:]).map(AACODES_DICT_upper)
    df_hhb_inner['AA_code_acceptor'] = df_hhb_inner.acceptor_res.apply(lambda x: x[-3:]).map(AACODES_DICT_upper)
    df_hhb_inner['donor_res_seq_num'] = df_hhb_inner.donor_res.apply(lambda x: x[-8:-4]).astype(int)
    df_hhb_inner['acceptor_res_seq_num'] = df_hhb_inner.acceptor_res.apply(lambda x: x[-8:-4]).astype(int)
    df_hhb_inner['donor_chain_identifier'] = df_hhb_inner.donor_res.apply(lambda x: x.split('/')[1][:1])
    df_hhb_inner['acceptor_chain_identifier'] = df_hhb_inner.acceptor_res.apply(lambda x: x.split('/')[1][:1])
    df_hhb_inner['donor_4digit'] = df_hhb_inner.donor_res.apply(lambda x: x.split('/')[1].split('-')[0])
    df_hhb_inner['acceptor_4digit'] = df_hhb_inner.acceptor_res.apply(lambda x: x.split('/')[1].split('-')[0])
    return df_hhb_inner

def get_mapping_2_4digits(donors_n_acceptors, pdbid):
    CDRname_2_4digits = {}
    Peptide_2_4digits = {}
    names = [n[0] for n in DICT_PDBID_2_CHAINNAMES[pdbid]]
    bond_list = []
    for n, i in zip(names, [0,1,2]):
        seqs = ''.join([f"{AACODES_DICT_upper[r.get_resname()]}" for r in DICT_PDBID_2_CDRS[pdbid][i]])
        queries = [f"{n}{res.get_id()[1]:004d}" for res in DICT_PDBID_2_CDRS[pdbid][i]]
        if i!=1:
            tokens = [f'{r}_{e}' for e, r in enumerate(seqs)]
            if i==0:
                CDRname_2_4digits.update({t: q for t, q in zip(tokens, queries)})
            else:
                Peptide_2_4digits.update({t: q for t, q in zip(tokens, queries)})
        if i==1:
            tokens = [f'{r}_{e+1+len(DICT_PDBID_2_CDRS[pdbid][0])}' for e, r in enumerate(seqs)]
            CDRname_2_4digits.update({t: q for t, q in zip(tokens, queries)})
        bond_list.append([text for text in donors_n_acceptors for q in queries if q in text])
    return CDRname_2_4digits, Peptide_2_4digits, bond_list


def vis_attentions(pdbid):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize = (40,10))
    for i in range(4):
        ax = axs[i]
        tcrs = DICT_PDBID_2_Atten12[pdbid][1][i].columns
        pep = DICT_PDBID_2_Atten12[pdbid][1][i].index
        ax.imshow(DICT_PDBID_2_Atten12[pdbid][1][i], vmin=0, vmax=.50)
        ax.set_xticks(range(len(tcrs)))
        ax.set_xticklabels(tcrs, fontdict=None, rotation=90)
        ax.set_yticks(range(len(pep)))
        ax.set_yticklabels(pep, fontdict=None, rotation=0)
    st.pyplot(fig)


def pdbid_to_donor_acceptor(pdbid, df_strong_temp, CDRname_2_4digits, Peptide_2_4digits, chains):
    df = read_hhb_file(pdbid)
    data_donors_n_acceptors = [read_hhb_text(row.item()) for i, row in df.iterrows()]
    data_donors_n_acceptors = [a for a in data_donors_n_acceptors if a is not None]
    df_donor_acceptor = pd.DataFrame(data_donors_n_acceptors)
    df_donor_acceptor = process_hhb(df_donor_acceptor)

    #
    df_donor_acceptor['donor_isin_largeAttention'] = df_donor_acceptor.donor_4digit.isin(df_strong_temp.tcr_residue)
    df_donor_acceptor['acceptor_isin_largeAttention'] = df_donor_acceptor.acceptor_4digit.isin(df_strong_temp.tcr_residue)
    df_donor_acceptor['CDR_isin_largeAttention'] = (df_donor_acceptor['donor_isin_largeAttention'] | df_donor_acceptor['acceptor_isin_largeAttention'])

    df_donor_acceptor['acceptor_isin_generalPeptide'] = df_donor_acceptor.acceptor_4digit.isin(Peptide_2_4digits.values())
    df_donor_acceptor['donor_isin_generalPeptide'] = df_donor_acceptor.donor_4digit.isin(Peptide_2_4digits.values())

    #
    df_donor_acceptor['donor_isin_generalCDR'] = df_donor_acceptor.donor_4digit.isin(CDRname_2_4digits.values())
    df_donor_acceptor['acceptor_isin_generalCDR'] = df_donor_acceptor.acceptor_4digit.isin(CDRname_2_4digits.values())
    df_donor_acceptor['isin_generalCDR'] = (df_donor_acceptor['donor_isin_generalCDR'] | df_donor_acceptor['acceptor_isin_generalCDR'])

    df_donor_acceptor['hbond_to_same_chain'] = df_donor_acceptor.donor_chain_identifier == df_donor_acceptor.acceptor_chain_identifier
    df_donor_acceptor['hbond_to_different_chain'] = df_donor_acceptor.donor_chain_identifier != df_donor_acceptor.acceptor_chain_identifier

    #
    df_donor_acceptor['hbond_relatedTo_AnyTCRs'] = (df_donor_acceptor.donor_chain_identifier.isin(list(''.join(chains[:2]))) |
                                                    df_donor_acceptor.acceptor_chain_identifier.isin(list(''.join(chains[:2])))
                                                    )
    df_donor_acceptor['hbond_relatedTo_Peptide'] = (df_donor_acceptor.donor_chain_identifier.isin(list(''.join(chains[-1]))) |
                                                    df_donor_acceptor.acceptor_chain_identifier.isin(list(''.join(chains[-1])))
                                                    )

    return df_donor_acceptor


(
    strong_givenPEP_distributedTCR_by_head,
    strong_givenTCR_distributedPEP_by_head,
) = get_attention_and_hhb_relationship(strong_atten=True)

(
    general_givenPEP_distributedTCR_by_head,
    general_givenTCR_distributedPEP_by_head,
) = get_attention_and_hhb_relationship(strong_atten=False)

DICT_PDBID_2_dfhhb = get_DICT_PDBID_2_dfhhb()

all_hbonds_list = []
general_interaction_list = []
for headi in range(4):
    for iii, _ in enumerate(strong_givenPEP_distributedTCR_by_head[headi]):
        pdbid = (
            general_givenPEP_distributedTCR_by_head[headi][iii]
                .pdbid.unique()
                .item()
        )
        df_hhb = DICT_PDBID_2_dfhhb[pdbid][["peptide_col", "beta_col", "alpha_col"]]
        df_hhb["beta_col"].fillna("", inplace=True)
        df_hhb["alpha_col"].fillna("", inplace=True)
        df_hhb = df_hhb.drop_duplicates(subset=["beta_col", "alpha_col"])
        df_hhb = df_hhb[(df_hhb["beta_col"] != "") | (df_hhb["alpha_col"] != "")]
        df_hhb["pdbid"] = pdbid
        if headi == 0:
            all_hbonds_list.append(df_hhb)
        df0 = pd.concat(
            [
                general_givenPEP_distributedTCR_by_head[headi][iii],
                strong_givenPEP_distributedTCR_by_head[headi][iii],
            ]
        )
        general_interaction_list.append(df0)
