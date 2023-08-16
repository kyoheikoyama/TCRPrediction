#!/usr/bin/env python
# coding: utf-8

import os, sys, argparse, pickle, json
import pandas as pd
import numpy as np
import pandas as pd
import warnings
from tqdm import tqdm

from precompute_dict import tcr_a_or_b, get_cdrs_from_anarci
from pathos.threading import ThreadPool

from Bio.PDB import *

warnings.filterwarnings(action="once")

import warnings

warnings.filterwarnings("ignore")

import multiprocessing as mp
import json


parser = argparse.ArgumentParser()
parser.add_argument(
    "--pdbdir",
    type=str,
    default="../analysis/zipdata/pdb",
)
parser.add_argument(
    "--sceptre_result_csv", type=str, default="../data/sceptre_result_v2.csv"
)
parser.add_argument(
    "--cdrpath", type=str, default="../data/20230817_020156__DICT_PDBID_2_CDRS.pickle"
)
parser.add_argument(
    "--chainnamespath", type=str, default="../data/DICT_PDBID_2_CHAINNAMES.json"
)

parser.add_argument("--pdblist", type=str, default='../data/pdblist.csv')  # or ../data/pdblist.csv

args = parser.parse_args()

datetime = args.cdrpath.split("/")[-1].split("__")[0]
output_distance_path = f"../data/{datetime}__residue_distances.parquet"

AACODES_DICT = {
    row["Abbreviation"].upper(): row["1 letter abbreviation"]
    for i, row in pd.read_csv("../analysis/aa_codes.csv").iterrows()
}


p_list = [
    f"../.././external_data/ERGO-II/Samples/vdjdb_train_samples.pickle",
    f"../.././external_data/ERGO-II/Samples/mcpas_train_samples.pickle",
]

# pdb entries from the query result
# PDBENTRIES = [a.replace(".ent", "").replace("pdb", "").upper() for a in os.listdir("/Users/kyoheikoyama/workspace/tcrpred/analysis/pdb")]

PDBENTRIES = pd.read_csv(args.pdblist)["pdbid"].unique().tolist()

df_sceptre_result = pd.read_csv(args.sceptre_result_csv)
sceptre_pdbs = df_sceptre_result["pdb_id"].unique().tolist()
PDBENTRIES = sorted(list(set(sceptre_pdbs + PDBENTRIES)))

df_sceptre_result = df_sceptre_result[
    [
        "chain1_type",
        "chain2_type",
        "chain1_cdr3_seq_calculated",
        "chain2_cdr3_seq_calculated",
        "epitope_seq",
        "epitope_accession_IRI",
        "epitope_organism_IRI",
        "pdb_id",
        "tcr_c1_pdb_chain",
        "tcr_c2_pdb_chain",
        "mhc_c1_pdb_chain",
        "mhc_c2_pdb_chain",
        "e_pdb_chain",
        "pdb_cell_contact_area",
        "chain1_cdr3_pdb_pos",
        "chain2_cdr3_pdb_pos",
        "calc_e_residues",
        "calc_e_tcr_residues",
        "calc_e_mhc_residues",
        "calc_tcr_e_residues",
        "calc_tcr_mhc_residues",
        "calc_mhc_e_residues",
        "calc_mhc_tcr_residues",
        "calc_e_contact_area",
        "calc_cell_contact_area",
    ]
]
df_sceptre_result = pd.concat(
    [df_sceptre_result, pd.DataFrame(df_sceptre_result.apply(tcr_a_or_b, axis=1))],
    axis=1,
)

print(len(df_sceptre_result))

MAXLENGTH_A, MAXLENGTH_B, max_len_epitope = 28, 28, 25

PDBENTRIES = [
    s.replace("pdb", "").replace(".ent", "").upper() for s in os.listdir(args.pdbdir)
]

PDBENTRIES = sorted(
    list(set(df_sceptre_result["pdb_id"].unique().tolist() + PDBENTRIES))
)

print("len(PDBENTRIES)", len(PDBENTRIES))

pickleload = lambda p: pickle.load(open(p,"rb"))

if os.path.exists(args.cdrpath):
    DICT_PDBID_2_CDRS = pickleload(args.cdrpath)
else:
    DICT_PDBID_2_CDRS = {}

    with open(args.chainnamespath, "r") as json_file:
        DICT_PDBID_2_CHAINNAMES = json.load(json_file)

    for p, v in tqdm(DICT_PDBID_2_CHAINNAMES.items()):
        residues_chain_alpha, residues_chain_beta, epi = v
        residues_chain_cdr_alpha, residues_chain_cdr_beta = get_cdrs_from_anarci(
            p, residues_chain_alpha, residues_chain_beta
        )
        DICT_PDBID_2_CDRS[p] = (residues_chain_cdr_alpha, residues_chain_cdr_beta, epi)


def multithread_func(func, iterable):
    import os
    pool = ThreadPool(nodes=os.cpu_count() - 3)
    return pool.map(func, iterable)


def calc_dist_2(epitope_residues, beta_chain_residues):
    """
    This function returns the minimum distance of all the atoms (not CA atom)
    """
    distmat = np.empty(
        (len(epitope_residues), len(beta_chain_residues)),
    )
    distmat[:] = np.nan
    for ei, epres in enumerate(epitope_residues):
        if epres.get_resname() == "HOH":
            distmat[ei, :] = None
            continue
        for bi, res in enumerate(beta_chain_residues):
            if res.get_resname() == "HOH":
                distmat[ei, bi] = None
                continue

            diff = np.array(
                [
                    (eatom.get_coord() - batom.get_coord())
                    for eatom in epres
                    for batom in res
                ]
            )
            min_squared_dist = np.min((diff * diff).sum(axis=1))
            distmat[ei, bi] = np.sqrt(min_squared_dist)
    # assert np.sum(distmat==0) ==0, 'there are zeros'
    return distmat


def distance_mat_from_residues(
    residues_chain_cdr_alpha, residues_chain_cdr_beta, residues_chain_epitope
):
    cdr_alpha_no_hyphen = res_to_alphabetseq(residues_chain_cdr_alpha)
    cdr_beta_no_hyphen = res_to_alphabetseq(residues_chain_cdr_beta)
    seq_epitope = res_to_alphabetseq(residues_chain_epitope)
    distmat1 = calc_dist_2(residues_chain_epitope, residues_chain_cdr_beta)
    dist_separate = np.zeros((len(residues_chain_epitope), 1))
    distmat2 = calc_dist_2(residues_chain_epitope, residues_chain_cdr_alpha)
    distmat = np.c_[distmat2, dist_separate, distmat1]
    distmat = pd.DataFrame(
        distmat,
        index=list(make_xlabel(seq_epitope)),
        columns=list(make_xlabel(cdr_alpha_no_hyphen + ":" + cdr_beta_no_hyphen)),
    )
    return distmat


def res_to_alphabetseq(rlist):
    return "".join([AACODES_DICT.get(r.get_resname(), "X") for r in rlist])


def make_xlabel(seq):
    return [f"{s}_{i}" for i, s in enumerate(seq)]


DICT_PDBID_2_DISTANCE = {}
for p, v in tqdm(DICT_PDBID_2_CDRS.items()):
    a, b, e = v
    if any([vv is None or len(vv) == 0 for vv in v]):
        continue
    else:
        DICT_PDBID_2_DISTANCE[p] = distance_mat_from_residues(
            a,
            b,
            e,
        )


DICT_PDBID_2_MELTDIST = {}
for p, distmat in DICT_PDBID_2_DISTANCE.items():
    distmat_vis = (
        distmat.drop(columns=[c for c in distmat.columns if ":" in c])
        .melt(ignore_index=False)
        .reset_index()
        .rename(columns={"variable": "tcr", "index": "peptide"})
    )
    distmat_vis = distmat_vis.sort_values(by=["peptide", "tcr"])
    DICT_PDBID_2_MELTDIST[p] = distmat_vis


df_distance = pd.concat([df.assign(pdbid=k) for k, df in DICT_PDBID_2_MELTDIST.items()])
df_distance.to_parquet(output_distance_path)
