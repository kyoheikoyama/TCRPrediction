#!/usr/bin/env python
# coding: utf-8

# # See the diff of (h-bond or non-h-bond) b/w strong and week attention by each head
#
# - hbond combination / Attention strong combination.
# - hbond combination / all combination.

import os, sys
import subprocess
import pickle, json
from tqdm import tqdm
import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind_from_stats
from matplotlib import pyplot as plt
import argparse

pd.options.display.float_format = "{:.5g}".format

pickleload = lambda p: pickle.load(open(p, "rb"))

HEAD_COUNT = 4



def read_hhb_file(pdbid):
    root = "/Users/kyoheikoyama/workspace/ligplottmp/hbadd_hbplus_result/"
    return pd.read_table(f"{root}pdb{pdbid.lower()}.hhb", header=None)


def read_hhb_text_and_find_chain(text, chainsletters):
    ress_atoms = (
        text[:45]
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .replace("  ", " ")
        .split(" ")
    )
    donor_res = ress_atoms[0]
    acceptor_res = ress_atoms[2]
    if (
        donor_res.split("/")[1][:1] in chainsletters
        or acceptor_res.split("/")[1][:1] in chainsletters
    ):
        return text


def add_4digits(r):
    res = r.get_full_id()[2]
    digits = r.get_full_id()[3][1]
    return f"{res}{digits:004d}"


def create_bond_info_by_pdb(pdbid):
    global DICT_bond_tuple_fromto
    DICT_bond_tuple_fromto = {}
    chains = DICT_PDBID_2_CHAINNAMES[pdbid.upper()]
    print(pdbid, chains)
    chainsletters = "".join(chains)

    dfhhb = read_hhb_file(pdbid)
    donors_n_acceptors = [
        read_hhb_text_and_find_chain(row.item(), chainsletters)
        for i, row in dfhhb.iterrows()
    ]
    donors_n_acceptors = [a for a in donors_n_acceptors if a is not None]

    bond_tuple_fromto = []
    for da in donors_n_acceptors:
        a = (
            da.replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .replace("  ", " ")
            .split(" ")
        )
        from_d = a[0].split("/")[1].split("-")[0][:5]
        to_d = a[2].split("/")[1].split("-")[0][:5]
        bond_tuple_fromto.append((from_d, to_d))

    DICT_bond_tuple_fromto[pdbid] = bond_tuple_fromto
    anyhbond_related_4digits = list(
        set(
            [tup[0] for tup in bond_tuple_fromto]
            + [tup[1] for tup in bond_tuple_fromto]
        )
    )

    CDRname_2_4digits, Peptide_2_4digits, bond_list = get_mapping_2_4digits(
        donors_n_acceptors, pdbid
    )
    dfc = pd.DataFrame(CDRname_2_4digits.items(), columns=["residue", "digit4"]).assign(
        **{"is_tcr": True, "pdbid": pdbid}
    )
    dfp = pd.DataFrame(Peptide_2_4digits.items(), columns=["residue", "digit4"]).assign(
        **{"is_tcr": False, "pdbid": pdbid}
    )
    df = pd.concat([dfc, dfp])
    df["has_bond"] = df["digit4"].isin(anyhbond_related_4digits)
    return df


def generate_alpha_beta_pep():
    alis, blis, clis = [], [], []
    for pdbid in NEEDED_PDBIDS:
        residues = DICT_PDBID_2_RESIDUES[pdbid]
        df_alpha = pd.DataFrame([residues[0]]).T.assign(pdbid=pdbid)
        df_alpha["digit4"] = df_alpha[0].apply(add_4digits)
        df_beta = pd.DataFrame([residues[1]]).T.assign(pdbid=pdbid)
        df_beta["digit4"] = df_beta[0].apply(add_4digits)
        df_pep = pd.DataFrame([residues[2]]).T.assign(pdbid=pdbid)
        df_pep["digit4"] = df_pep[0].apply(add_4digits)
        alis.append(df_alpha)
        blis.append(df_beta)
        clis.append(df_pep)
    return pd.concat(alis), pd.concat(blis), pd.concat(clis)


def load_AACODES_DICT():
    aacodepath = os.path.join(os.path.dirname(__file__), "../analysis", "aa_codes.csv")
    AACODES = pd.read_csv(aacodepath)
    AACODES_DICT = {
        row["Abbreviation"]: row["1 letter abbreviation"] for i, row in AACODES.iterrows()
    }
    AACODES_DICT_upper = {k.upper(): v for k, v in AACODES_DICT.items()}
    AACODES_DICT.update(AACODES_DICT_upper)
    AACODES_DICT_upper.update(AACODES_DICT)
    return AACODES_DICT_upper


def get_mapping_2_4digits(donors_n_acceptors, pdbid):
    CDRname_2_4digits = {}
    Peptide_2_4digits = {}
    AACODES_DICT_upper = load_AACODES_DICT()
    names = [n[0] for n in DICT_PDBID_2_CHAINNAMES[pdbid]]
    bond_list = []
    for n, i in zip(names, [0, 1, 2]):
        seqs = "".join(
            [
                f"{AACODES_DICT_upper.get(r.get_resname(), 'X')}"
                for r in DICT_PDBID_2_CDRS[pdbid][i]
            ]
        )
        queries = [f"{n}{res.get_id()[1]:004d}" for res in DICT_PDBID_2_CDRS[pdbid][i]]
        if i != 1:
            tokens = [f"{r}_{e}" for e, r in enumerate(seqs)]
            if i == 0:
                CDRname_2_4digits.update({t: q for t, q in zip(tokens, queries)})
            else:
                Peptide_2_4digits.update({t: q for t, q in zip(tokens, queries)})
        if i == 1:
            tokens = [
                f"{r}_{e+1+len(DICT_PDBID_2_CDRS[pdbid][0])}"
                for e, r in enumerate(seqs)
            ]
            CDRname_2_4digits.update({t: q for t, q in zip(tokens, queries)})
        bond_list.append(
            [text for text in donors_n_acceptors for q in queries if q in text]
        )
    return CDRname_2_4digits, Peptide_2_4digits, bond_list


def main(args):
    df_alpha, df_beta, df_pep = generate_alpha_beta_pep()
    df_bondinfo = pd.concat([create_bond_info_by_pdb(p) for p in NEEDED_PDBIDS])
    df_bond_tuple_fromto = pd.DataFrame(DICT_bond_tuple_fromto.items(), columns=["pdbid", "bonds"])

    df_bond_relations = []
    for i, row in df_bond_tuple_fromto.iterrows():
        list_of_tuple = row["bonds"]
        pdbid = row["pdbid"]
        bond_dict = defaultdict(list)
        for tup in list_of_tuple:
            bond_dict[tup[0]].append(tup[1])
            bond_dict[tup[1]].append(tup[0])
        t = pd.DataFrame(bond_dict.items(), columns=["from_res", "to_res_list"]).assign(
            pdbid=pdbid
        )
        df_bond_relations.append(t)
    df_bond_relations = pd.concat(df_bond_relations)
    df_bond_relations = df_bond_relations.rename(columns={"from_res": "digit4"})

    df_bondinfo = pd.merge(
        df_bondinfo,
        df_bond_relations,
        left_on=["pdbid", "digit4"],
        right_on=["pdbid", "digit4"],
        how="left",
    )

    df_bondinfo["to_res_list"] = df_bondinfo.to_res_list.apply(
        lambda xlis: [] if not isinstance(xlis, list) else xlis
    )

    # is_connecting_tcr
    df_bondinfo["is_connecting_to_cdr"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if isinstance(row.to_res_list, list):
            p = row.pdbid
            cdr_list = df_bondinfo.query("is_tcr==True and pdbid==@p")["digit4"].tolist()
            is_any_res_in_cdr = any([(r in cdr_list) for r in row.to_res_list])
            df_bondinfo.loc[[i], "is_connecting_to_cdr"] = is_any_res_in_cdr

    del cdr_list, is_any_res_in_cdr


    df_bondinfo["is_connecting_to_pep"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if isinstance(row.to_res_list, list):
            p = row.pdbid
            pep_list = df_bondinfo.query("is_tcr==False and pdbid==@p")["digit4"].tolist()
            is_any_res_in_pep = any([(r in pep_list) for r in row.to_res_list])
            df_bondinfo.loc[[i], "is_connecting_to_pep"] = is_any_res_in_pep
    del pep_list, is_any_res_in_pep

    s = df_bondinfo.query('is_tcr==True')['is_connecting_to_pep'].sum()
    m = df_bondinfo.query('is_tcr==True')['is_connecting_to_pep'].mean()
    print("df_bondinfo.query('is_tcr==True')['is_connecting_to_pep'].sum and mean = ", s, m)
    assert s!=0, "is_connecting_to_pep is not working properly."


    df_bondinfo["is_connecting_to_tcr"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if isinstance(row.to_res_list, list):
            p = row.pdbid
            tcr_list = [t for t in df_alpha.query("pdbid==@p")["digit4"].tolist()] + [
                t for t in df_beta.query("pdbid==@p")["digit4"].tolist()
            ]
            is_tcr_list = any([(r in tcr_list) for r in row.to_res_list])
            df_bondinfo.loc[[i], "is_connecting_to_tcr"] = is_tcr_list

    del tcr_list, is_tcr_list


    df_bondinfo["is_connecting_to_notCDR_tcr"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if isinstance(row.to_res_list, list):
            p = row.pdbid
            cdr_list = df_bondinfo.query("is_tcr==True and pdbid==@p")["digit4"].tolist()
            notcdr_tcr_list = [
                t
                for t in df_alpha.query("pdbid==@p")["digit4"].tolist()
                if t not in cdr_list
            ] + [
                t
                for t in df_beta.query("pdbid==@p")["digit4"].tolist()
                if t not in cdr_list
            ]
            is_notcdr_tcr_list = any([(r in notcdr_tcr_list) for r in row.to_res_list])
            df_bondinfo.loc[[i], "is_connecting_to_notCDR_tcr"] = is_notcdr_tcr_list
    del notcdr_tcr_list, is_notcdr_tcr_list

    s = df_bondinfo.query('is_tcr==True').is_connecting_to_notCDR_tcr.sum()
    m = df_bondinfo.query('is_tcr==True').is_connecting_to_notCDR_tcr.mean()
    print("df_bondinfo.query('is_tcr==True').is_connecting_to_notCDR_tcr.sum() = ", s, m)
    assert s!=0, "is_connecting_to_notCDR_tcr is not working properly."

    df_bondinfo["is_connecting_to_ownchain_tcr"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if row.is_tcr == False:
            continue
        is_connecting_to_ownchain_tcr = False
        p = row.pdbid
        alpha_digits_list = df_alpha.query("pdbid==@p")["digit4"].tolist()
        beta_digits_list = df_beta.query("pdbid==@p")["digit4"].tolist()
        is_in_alpha = row.digit4 in alpha_digits_list
        is_in_beta = row.digit4 in beta_digits_list
        assert (is_in_alpha + is_in_beta) > 0
        if is_in_alpha:
            is_connecting_to_ownchain_tcr = any(
                [(r in alpha_digits_list) for r in row.to_res_list]
            )
        if is_in_beta:
            is_connecting_to_ownchain_tcr = any(
                [(r in beta_digits_list) for r in row.to_res_list]
            )

        df_bondinfo.loc[
            [i], "is_connecting_to_ownchain_tcr"
        ] = is_connecting_to_ownchain_tcr


    df_bondinfo["is_connecting_to_ownchain_cdr"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if row.is_tcr == False:
            continue
        is_connecting_to_ownchain_cdr = False
        p = row.pdbid
        tcr_list = df_bondinfo.query("is_tcr==True and pdbid==@p")["digit4"].tolist()
        alpha_digits_list = df_alpha.query("pdbid==@p and (digit4 in @tcr_list)")[
            "digit4"
        ].tolist()
        beta_digits_list = df_beta.query("pdbid==@p and (digit4 in @tcr_list)")[
            "digit4"
        ].tolist()
        is_in_alpha = row.digit4 in alpha_digits_list
        is_in_beta = row.digit4 in beta_digits_list
        assert (is_in_alpha + is_in_beta) > 0
        if is_in_alpha:
            is_connecting_to_ownchain_cdr = any(
                [(r in alpha_digits_list) for r in row.to_res_list]
            )
        if is_in_beta:
            is_connecting_to_ownchain_cdr = any(
                [(r in beta_digits_list) for r in row.to_res_list]
            )

        df_bondinfo.loc[
            [i], "is_connecting_to_ownchain_cdr"
        ] = is_connecting_to_ownchain_cdr


    df_bondinfo["is_connecting_to_opposite_chain_tcr"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if row.is_tcr == False:
            continue
        is_opposite = False
        p = row.pdbid
        # tcr_list = df_bondinfo.query('is_tcr==True and pdbid==@p')['digit4'].tolist()
        alpha_digits_list = df_alpha.query("pdbid==@p ")["digit4"].tolist()
        beta_digits_list = df_beta.query("pdbid==@p ")["digit4"].tolist()
        is_in_alpha = row.digit4 in alpha_digits_list
        is_in_beta = row.digit4 in beta_digits_list
        assert (is_in_alpha + is_in_beta) > 0
        if is_in_alpha:
            is_opposite = any([(r in beta_digits_list) for r in row.to_res_list])
        if is_in_beta:
            is_opposite = any([(r in alpha_digits_list) for r in row.to_res_list])

        df_bondinfo.loc[[i], "is_connecting_to_opposite_chain_tcr"] = is_opposite


    df_bondinfo["is_connecting_to_opposite_chain_cdr"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        if row.is_tcr == False:
            continue
        is_opposite = False
        p = row.pdbid
        tcr_list = df_bondinfo.query("is_tcr==True and pdbid==@p")["digit4"].tolist()
        alpha_digits_list = df_alpha.query("pdbid==@p and (digit4 in @tcr_list)")[
            "digit4"
        ].tolist()
        beta_digits_list = df_beta.query("pdbid==@p and (digit4 in @tcr_list)")[
            "digit4"
        ].tolist()
        is_in_alpha = row.digit4 in alpha_digits_list
        is_in_beta = row.digit4 in beta_digits_list
        assert (is_in_alpha + is_in_beta) > 0
        if is_in_alpha:
            is_opposite = any([(r in beta_digits_list) for r in row.to_res_list])
        if is_in_beta:
            is_opposite = any([(r in alpha_digits_list) for r in row.to_res_list])

        df_bondinfo.loc[[i], "is_connecting_to_opposite_chain_cdr"] = is_opposite


    df_bondinfo["num_bonds"] = df_bondinfo["to_res_list"].apply(len)

    df_bondinfo["digit4_is_in_edge"] = False
    for i, row in tqdm(df_bondinfo.iterrows()):
        p = row.pdbid
        
        residue_list = df_bondinfo.query("is_tcr==True and pdbid==@p")["residue"].tolist()
        pos_all = (
            pd.Series(residue_list).str.split("_").apply(lambda x: x[1]).values.astype(int)
        )
        try:
            colon_pos = [i for i in range(len(pos_all)) if i not in pos_all][0]
            beginnings_endings_cdrs = (
                list(range(EDGENUM))
                + pos_all[-EDGENUM:].tolist()
                + [colon_pos - i - 1 for i in range(EDGENUM)]
                + [colon_pos + i + 1 for i in range(EDGENUM)]
            )

            df_bondinfo.loc[[i], "digit4_is_in_edge"] = (
                int(row.residue.split("_")[1]) in beginnings_endings_cdrs
            )
        except:
            print('Error in pdbid=', p)
            df_bondinfo.loc[[i], "digit4_is_in_edge"] = None


    
    print(df_bondinfo.head(3))

    df_bondinfo.to_parquet(f"../data/{datetime}__df_bondinfo.parquet")
    print(f"../data/{datetime}__df_bondinfo.parquet")


if __name__ == '__main__':
    """# example command
    python create_pdb_info.py \
        --dict_pdbid_2_chainnames ../data/DICT_PDBID_2_CHAINNAMES.json \
        --dict_pdbid_2_residues ../data/DICT_PDBID_2_RESIDUES.pickle \
        --dict_pdbid_2_cdrs ../data/DICT_PDBID_2_CDRS.pickle

    python create_pdb_info.py \
        --dict_pdbid_2_chainnames ../data/DICT_PDBID_2_CHAINNAMES.json \
        --dict_pdbid_2_residues ../data/20230816_045104__DICT_PDBID_2_RESIDUES.pickle \
        --dict_pdbid_2_cdrs ../data/20230816_045104__DICT_PDBID_2_CDRS.pickle \
        --residue_distances ../data/20230816_045104__residue_distances.parquet
    """

    # load paths with argument parser for DICT_PDBID_2_CHAINNAMES, DICT_PDBID_2_RESIDUES, and DICT_PDBID_2_CDRS
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_pdbid_2_chainnames", type=str, default="../data/DICT_PDBID_2_CHAINNAMES.json")
    parser.add_argument("--dict_pdbid_2_residues", type=str, default="../data/DICT_PDBID_2_RESIDUES.pickle")
    parser.add_argument("--dict_pdbid_2_cdrs", type=str, default="../data/DICT_PDBID_2_CDRS.pickle")
    parser.add_argument("--residue_distances", type=str, default="./../data/residue_distances.parquet")
    args = parser.parse_args()

    datetime = args.residue_distances.split("/")[-1].split("__")[0]

    # load json
    # DICT_PDBID_2_RESIDUES.pickle:  https://drive.google.com/file/d/1tTZB8ki1eHuYcXMJ1Vk8PIFUgUtJwuyW/view?usp=sharing
    # DICT_PDBID_2_CDRS.pickle:      https://drive.google.com/file/d/1S58FgfD8vKIQ1ChyfDJ2tbqdyx3WEeNZ/view?usp=sharing
    DICT_PDBID_2_CHAINNAMES = json.load(open(args.dict_pdbid_2_chainnames))
    DICT_PDBID_2_RESIDUES = pickleload(args.dict_pdbid_2_residues)
    DICT_PDBID_2_CDRS = pickleload(args.dict_pdbid_2_cdrs)
    df_distance = pd.read_parquet(args.residue_distances)
    NEEDED_PDBIDS = DICT_PDBID_2_CDRS.keys()
    # NEEDED_PDBIDS = ['2VLK', '5WKF', '3PQY', '4MJI', '4P2Q', '2YPL', '1J8H', '4P2R', '5MEN', '3MV8', '4OZF', '3VXR', '3VXS', '4OZG', '5TEZ', '2J8U', '6Q3S', '4JRX', '3VXU', '1U3H', '4JRY', '4Z7V', '4JFE', '4JFD', '3QIU', '2Z31', '2BNR', '3MBE', '4OZH', '2NX5', '5NHT', '4QOK', '5D2L', '1D9K', '4P2O', '5WKH', '6EQB', '2VLR', '6EQA']

    EDGENUM = 4

    main(args)

    