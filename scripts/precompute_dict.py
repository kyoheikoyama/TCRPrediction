import  multiprocessing as mp
import pickle, os
import pandas as pd
from tqdm import tqdm
from Bio.PDB.PDBParser import PDBParser
import warnings
warnings.simplefilter("ignore")
import argparse
from datetime import datetime

def tcr_a_or_b(row):
    if row['chain1_type'] == 'alpha':
        return pd.Series(
            {'tcra_seq': row['chain1_cdr3_seq_calculated'], 
             'tcrb_seq': row['chain2_cdr3_seq_calculated']})
    else:
        return pd.Series(
            {'tcrb_seq': row['chain1_cdr3_seq_calculated'], 
             'tcra_seq': row['chain2_cdr3_seq_calculated']})

def remove_HOH(residues):
    return [r for r in residues 
            if (r.get_resname() != 'HOH') and (r.get_resname() != 'IPA') and (r.get_resname()!='IOD')]


def from_str_to_chain_names(structs):
    return [c.full_id[-1] for c in structs[0].get_chains()]

def res_to_alphabetseq(rlist):
    return ''.join([AACODES_DICT.get(r.get_resname(), 'X') for r in rlist])

def split_and_get_first(a):
    if ',' in a:
        a = a.split(',')[0]
    return a

def get_a_b_e_chains_from_sceptre(row):
    if row['chain1_type'] == 'alpha':
        return row.tcr_c1_pdb_chain, row.tcr_c2_pdb_chain, row.e_pdb_chain
    else:
        return row.tcr_c2_pdb_chain, row.tcr_c1_pdb_chain, row.e_pdb_chain

def get_a_b_e_chains_from_structure(structure):
    TCR_QUERIES = ['tcr', 't-cell receptor', 't cell receptor', 't-cell-receptor', 't cell-receptor', 'cdr', 'complementarity determining region']
    ALPHA_CHAIN_NAME, BETA_CHAIN_NAME, EPITOPE_CHAIN_NAME = None, None, None
    for k, v in structure.header['compound'].items():
        if any([t in str(v).lower() for t in TCR_QUERIES]):
            if ('alpha' in str(v['molecule']).lower()) and any([t in str(v['molecule']).lower() for t in TCR_QUERIES]):
                ALPHA_CHAIN_NAME = v['chain'].upper()
            if ('beta' in str(v['molecule']).lower()) and any([t in str(v['molecule']).lower() for t in TCR_QUERIES]):
                BETA_CHAIN_NAME = v['chain'].upper()
        if v['molecule'].count('-') > 3 or 'peptide' in str(v['molecule']).lower() or\
         'fragment' in str(v['molecule']).lower() or 'antigen' in str(v['molecule']).lower() or\
            'epitope' in str(v['molecule']).lower():
            chain = v['chain'].upper()
            if len(list(structure[0][chain[0]].get_residues())) < 30:
                EPITOPE_CHAIN_NAME = chain
            # print('*** TCR *** \n', k, v)
            # print(v['chain'].upper(), ' <- chain name?')
            # print()
            
    return ALPHA_CHAIN_NAME, BETA_CHAIN_NAME, EPITOPE_CHAIN_NAME


def run_anarci(seq_alpha, seq_beta, PDBID):
    outfile = f'{ANARCIROOT}/{PDBID}_anarci'
    anarci_command = f"ANARCI -i {seq_alpha} -o {outfile} --csv"
    print(anarci_command)
    os.system(anarci_command)

    outfile = f'{ANARCIROOT}/{PDBID}_anarci'
    anarci_command = f"ANARCI -i {seq_beta} -o {outfile} --csv"
    print(anarci_command)
    os.system(anarci_command)

    
def get_cdrs_from_anarci(pdbid, residues_chain_alpha, residues_chain_beta):
    _IMGT_CDR_POS = [str(i) for i in range(104, 119)] #
    seq_alpha = res_to_alphabetseq(residues_chain_alpha)
    seq_beta = res_to_alphabetseq(residues_chain_beta)
    
    if not os.path.exists(f'{ANARCIROOT}/{pdbid}_anarci_A.csv'):
        run_anarci(seq_alpha.replace("X",""), seq_beta.replace("X",""), pdbid)

    try:
        cdr_beta = ''.join(
            pd.read_csv(f'{ANARCIROOT}/{pdbid}_anarci_B.csv')[_IMGT_CDR_POS].values[0].tolist())
        cdr_alpha = ''.join(
            pd.read_csv(f'{ANARCIROOT}/{pdbid}_anarci_A.csv')[_IMGT_CDR_POS].values[0].tolist())
    except:
        print(f'{pdbid} is skipped because anarci failed')
        return None, None
    
    cdr_beta_no_hyphen = cdr_beta.replace('-','')
    cdr_alpha_no_hyphen = cdr_alpha.replace('-','')
    cdr_start_pos_alpha = seq_alpha.find(cdr_alpha_no_hyphen[:5])
    cdr_start_pos_beta = seq_beta.find(cdr_beta_no_hyphen[:5])
    residues_chain_cdr_alpha = residues_chain_alpha[cdr_start_pos_alpha:cdr_start_pos_alpha+len(cdr_alpha_no_hyphen)]
    residues_chain_cdr_beta = residues_chain_beta[cdr_start_pos_beta:cdr_start_pos_beta+len(cdr_beta_no_hyphen)]
    return residues_chain_cdr_alpha, residues_chain_cdr_beta
    

def get_structure_from_id(ent):
    # if file exists, load it, otherwise download it
    if not os.path.exists(f"{args.pdbdir}/pdb{ent}.ent"):
        from Bio.PDB import PDBList
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(ent, pdir=args.pdbdir, file_format='pdb')

    parser = PDBParser()
    structure = parser.get_structure(ent, f"{args.pdbdir}/pdb{ent}.ent")
    return structure


def get_residues_from_names(structure, chain_names):
    return [remove_HOH(list(structure[0][chain_name].get_residues())) 
            for chain_name in chain_names]


pickleload = lambda p: pickle.load(open(p,"rb"))


def main(args):
    # pdb entries from the query result
    if args.pdblist is not None:
        PDBENTRIES = pd.read_csv(args.pdblist).pdbid.tolist()
    else:
        PDBENTRIES =  [a.replace('.ent','').replace('pdb','').upper() 
                       for a in os.listdir(args.pdbdir)]
    
    assert '2YPL' in PDBENTRIES, '2YPL is not in PDBENTRIES'

    df_sceptre_result = pd.concat([
        DF_SCEPTRE_result,
        pd.DataFrame(DF_SCEPTRE_result.apply(tcr_a_or_b, axis=1))], axis=1)

    PDBENTRIES = sorted(list(set(df_sceptre_result['pdb_id'].unique().tolist() + PDBENTRIES)))

    assert '2YPL' in PDBENTRIES, '2YPL is not in PDBENTRIES'

    print('len(df_sceptre_result) =', len(df_sceptre_result))
    print('len(PDBENTRIES) =', len(PDBENTRIES), PDBENTRIES)
    structs = [get_structure_from_id(p) for p in tqdm(PDBENTRIES)]

    DICT_PDBID_2_STRUCTURE = {pdbid:s for pdbid, s in zip(PDBENTRIES, structs)}

    # sceptre is preferred. if not, use the structure from the query
    DICT_PDBID_2_CHAINNAMES = {row["pdb_id"]: get_a_b_e_chains_from_sceptre(row) for i, row in DF_SCEPTRE_result.iterrows()}
    DICT_PDBID_2_CHAINNAMES.update(
        {pdbid:get_a_b_e_chains_from_structure(s) for pdbid, s in DICT_PDBID_2_STRUCTURE.items() if pdbid not in DICT_PDBID_2_CHAINNAMES.keys()})

    import json
    with open('../data/DICT_PDBID_2_CHAINNAMES.json', "w") as json_file:
        json.dump(DICT_PDBID_2_CHAINNAMES, json_file)


    DICT_PDBID_2_RESIDUES = {}
    for p, v in tqdm(DICT_PDBID_2_CHAINNAMES.items()):
        if len(v) != 3 or None in v:
            print(p, 'is skipped because it has chains of',v)
            continue
        a, b, e = v
        a = split_and_get_first(a)
        b = split_and_get_first(b)
        e = split_and_get_first(e)
        s = DICT_PDBID_2_STRUCTURE[p]
        DICT_PDBID_2_RESIDUES[p]  = get_residues_from_names(s, [a, b, e])

    assert '5TEZ' in DICT_PDBID_2_RESIDUES, '5TEZ is not in DICT_PDBID_2_RESIDUES'
    assert '2YPL' in DICT_PDBID_2_RESIDUES, '2YPL is not in DICT_PDBID_2_RESIDUES'
    assert '1U3H' in DICT_PDBID_2_RESIDUES, '1U3H is not in DICT_PDBID_2_RESIDUES'
    assert '2Z31' in DICT_PDBID_2_RESIDUES, '2Z31 is not in DICT_PDBID_2_RESIDUES'

    ######## CDRs ########
    DICT_PDBID_2_CDRS = {}
    for pdbid, v in DICT_PDBID_2_RESIDUES.items():
        if pdbid=='5JZI' or pdbid=='4QRR' or pdbid=='5JHD' or pdbid=='2OL3':
            print(f'{pdbid} has only delta-chain and beta-chain' )
            continue
        residues_chain_alpha, residues_chain_beta, epi = v
        residues_chain_cdr_alpha, residues_chain_cdr_beta = \
            get_cdrs_from_anarci(pdbid, residues_chain_alpha, residues_chain_beta)
        if residues_chain_cdr_alpha is None or residues_chain_cdr_beta is None:
            continue
        else:
            DICT_PDBID_2_CDRS[pdbid] = (residues_chain_cdr_alpha, residues_chain_cdr_beta, epi)

    print('len(DICT_PDBID_2_CDRS) =', len(DICT_PDBID_2_CDRS), sorted(DICT_PDBID_2_CDRS.keys()))
    assert '5TEZ' in DICT_PDBID_2_CDRS, '5TEZ is not in DICT_PDBID_2_CDRS'
    
    assertion_list = []
    for key in DICT_PDBID_2_CDRS.keys():
        if 0 in [len(DICT_PDBID_2_CDRS[key][0]), len(DICT_PDBID_2_CDRS[key][1]), len(DICT_PDBID_2_CDRS[key][2])]:
            print("zero-lengths", key, len(DICT_PDBID_2_CDRS[key][0]), len(DICT_PDBID_2_CDRS[key][1]), len(DICT_PDBID_2_CDRS[key][2]))
            assertion_list.append(key)
    assert len(assertion_list) == 0, f'beta chain of ({assertion_list}) is not in DICT_PDBID_2_CDRS'

    datetimehash = datetime.now().strftime('%Y%m%d_%H%M%S')
    pickle.dump(DICT_PDBID_2_CDRS, open(f"../data/{datetimehash}__DICT_PDBID_2_CDRS.pickle", "wb"))
    pickle.dump(DICT_PDBID_2_RESIDUES, open(f"../data/{datetimehash}__DICT_PDBID_2_RESIDUES.pickle", "wb"))
    print('saved to', f"../data/{datetimehash}__DICT_PDBID_2_CDRS.pickle")
    print('saved to', f"../data/{datetimehash}__DICT_PDBID_2_RESIDUES.pickle")

if __name__ == '__main__':
    ## example usage:
    # python3 precompute_dict.py --pdbdir ../analysis/zipdata/pdb --sceptre_result_csv ../data/sceptre_result_v2.csv
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdblist", type=str, default='../data/pdblist.csv')  # or ../data/pdblist.csv
    parser.add_argument("--pdbdir", type=str, default='../analysis/zipdata/pdb')
    parser.add_argument("--anarciroot", type=str, default='/Users/kyoheikoyama/workspace/tcrpred/analysis/analysis1_alldata/anarci')
    parser.add_argument("--sceptre_result_csv", type=str, default='../data/sceptre_result_v2.csv')
    args = parser.parse_args()

    ANARCIROOT = args.anarciroot
    AACODES_DICT = {row['Abbreviation'].upper():row['1 letter abbreviation'] for i, row in pd.read_csv('../analysis/aa_codes.csv').iterrows()}

    DF_SCEPTRE_result = pd.read_csv(args.sceptre_result_csv)
    DF_SCEPTRE_result = DF_SCEPTRE_result[[
        'chain1_type', 'chain2_type', 'chain1_cdr3_seq_calculated', 'chain2_cdr3_seq_calculated', 
        'epitope_seq', 'epitope_accession_IRI', 'epitope_organism_IRI', 
        'pdb_id', 'tcr_c1_pdb_chain', 'tcr_c2_pdb_chain', 'mhc_c1_pdb_chain', 'mhc_c2_pdb_chain', 'e_pdb_chain', 'pdb_cell_contact_area', 'chain1_cdr3_pdb_pos', 'chain2_cdr3_pdb_pos',
        'calc_e_residues', 'calc_e_tcr_residues', 'calc_e_mhc_residues', 'calc_tcr_e_residues', 'calc_tcr_mhc_residues', 'calc_mhc_e_residues', 'calc_mhc_tcr_residues', 'calc_e_contact_area', 'calc_cell_contact_area'
                                        ]]

    main(args)