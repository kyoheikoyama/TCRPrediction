from Bio.PDB.PDBParser import PDBParser
import numpy as np
import pandas as pd
import os

aacodepath = os.path.join(os.path.dirname(__file__), './aa_codes.csv')
AACODES = pd.read_csv(aacodepath)
AACODES_DICT = {row['Abbreviation']:row['1 letter abbreviation'] for i, row in AACODES.iterrows()}
AACODES_DICT_upper = {k.upper():v for k,v in AACODES_DICT.items()}


# Create a list of epitope residues
def get_chain_list(structure, chain_name):
    residues_epitope = []
    for model in structure.get_list():
        for chain in model.get_list():
            if str(chain.id)==chain_name:
                for residue in chain.get_list():
                    residues_epitope.append(residue)
            else:
                continue
    return residues_epitope


# Calculate the distance between the alpha carbons for a pair of residues
def calc_dist(residues_epitope, residues_chain_beta):
    distmat = np.zeros((len(residues_epitope), len(residues_chain_beta)))
    for ei, epres in enumerate(residues_epitope):
        if epres.get_resname() == 'HOH': 
            distmat[ei, :] = None
            continue
        for bi, res in enumerate(residues_chain_beta):
            if res.get_resname() == 'HOH': 
                distmat[ei, bi] = None
                continue
            if 'CA' in [a.id for a in epres.get_atoms()] and 'CA' in [a.id for a in res.get_atoms()]:
                diff = epres['CA'].get_coord() - res["CA"].get_coord()
                distmat[ei, bi] = np.sqrt(np.sum(diff * diff))
            else:
                # diff = epres['C'].get_coord() - res["C"].get_coord()
                distmat[ei, bi] = None
    return distmat


def remove_HOH(residues):
    return [r for r in residues 
            if (r.get_resname() != 'HOH') and (r.get_resname() != 'IPA') and (r.get_resname()!='IOD')]

