#!/usr/bin/env python
# coding: utf-8


# command = 'cd /Users/kyoheikoyama/workspace/ligplottmp/ && python /Users/kyoheikoyama/workspace/LigPlus/command.py'
# subprocess.check_call(command.split(' '), shell = False)


import os
import subprocess
import pickle
import json
from tqdm import tqdm


import subprocess
import os, pickle
from tqdm import tqdm

"""
"LigPlot command line" needs ResidueName and LigandPosition.

- Command mannual
https://www.ebi.ac.uk/thornton-srv/software/LIGPLOT/manual/
https://www.ebi.ac.uk/thornton-srv/software/LIGPLOT/manual/man2.html

- Output format
UI mannual
https://www.ebi.ac.uk/thornton-srv/software/LigPlus/manual2/manual.html
"""


def run_hbplus(filename):
	subprocess.check_call(['{}hbadd'.format(ligplot_plus), filename, components_cif, '-wkdir', f'{LIGPLOT_ROOT}/hbadd_hbplus_result/'], shell = False)
	file_prefix = filename.split('/')[-1].replace('.ent', '')

	if len(file_prefix)!=7:
		print(file_prefix, filename)
		assert False

	subprocess.check_call(['{}hbplus'.format(ligplot_plus), '-L', '-h', '2.90', '-d', '3.90', '-N', filename, '-wkdir', f'{LIGPLOT_ROOT}/hbadd_hbplus_result/'], shell = False)
	subprocess.check_call(['{}hbplus'.format(ligplot_plus), '-L', '-h', '2.70', '-d', '3.35', filename, '-wkdir', f'{LIGPLOT_ROOT}/hbadd_hbplus_result/'], shell = False)



# def run_all(filename, res1, res2, ligand_chain):
# 	"""Emulates running the LigPlot+ DIMPLOT algorithm. Rewriting as a CLI to allow for a batch mode."""

# 	file_prefix = filename.split('/')[-1].replace('.ent', '')
# 	if len(file_prefix)!=7:
# 		print(file_prefix, filename)
# 		assert False

# 	assert len(file_prefix)==7

# 	if os.path.exists(file_prefix):
# 		return

# 	# Run HBadd
# 	subprocess.check_call(['{}hbadd'.format(ligplot_plus), filename, components_cif, '-wkdir', f'{LIGPLOT_ROOT}/hbadd_hbplus_result/'], shell = False)


# 	# Run HBplus
# 	subprocess.check_call(['{}hbplus'.format(ligplot_plus), '-L', '-h', '2.90', '-d', '3.90', '-N', filename, '-wkdir', f'{LIGPLOT_ROOT}/hbadd_hbplus_result/'], shell = False)


# 	# Run HBplus again
# 	subprocess.check_call(['{}hbplus'.format(ligplot_plus), '-L', '-h', '2.70', '-d', '3.35', filename, '-wkdir', f'{LIGPLOT_ROOT}/hbadd_hbplus_result/'], shell = False)


# 	# Run ligplot
# 	# ligplot filename [residue1] [residue2] [chain_id] [-w] [-m].
# 	# [residue1] and [residue2] is the residue range for the ligand
# 	subprocess.check_call([
# 		'{}ligplot'.format(ligplot_plus), filename, str(res1), str(res2), ligand_chain, '-wkdir', f'{LIGPLOT_ROOT}/hbadd_hbplus_result/',
# 		'-prm', '/Users/kyoheikoyama/workspace/LigPlus/lib/params/ligplot.prm'],
# 		shell = False)

# 	files_to_rename = [_ for _ in os.listdir('.') if 'dimplot.' in _[0:8] or 'ligplot.' in _[0:8]]

# 	os.system(f'mkdir {file_prefix}')

# 	for file_to_rename in files_to_rename:
# 		subprocess.check_call([
# 			'mv', file_to_rename, f'./{file_prefix}/{file_to_rename}'])

# def main():
# 	"""Main function."""
# 	for pdbid in tqdm(list(DICT_PDBID_2_CDRS.keys())):
# 		alpha_chain, beta_chain, ligand_chain = DICT_PDBID_2_CHAINNAMES[pdbid]
# 		pdb_file = os.path.join(pdbdir, f"pdb{pdbid.lower()}.ent")
# 		r_0 = DICT_PDBID_2_CDRS[pdbid][-1][0]
# 		r_m1 = DICT_PDBID_2_CDRS[pdbid][-1][-1]
# 		res1, res2 = r_0.get_full_id()[-1][1], r_m1.get_full_id()[-1][1]
# 		print("*"*400)
# 		print("""pdb_file, res1, res2, ligand_chain""")
# 		print(pdb_file, res1, res2, ligand_chain)
# 		try:
# 			#run_hbplus(pdb_file, res1, res2, ligand_chain)
# 			run_all(pdb_file, res1, res2, ligand_chain)
# 		except:
# 			continue
# 		break
# 	print('done!')

def hbplus():
	DICT_PDBID_2_CDRS = pickle.load(open(args.dict_pdbid_2_cdrs, "rb"))
	DICT_PDBID_2_CHAINNAMES = pickle.load(open(args.dict_chainnames, "rb"))

	for pdbid in tqdm(list(DICT_PDBID_2_CHAINNAMES.keys())):
		alpha_chain, beta_chain, ligand_chain = DICT_PDBID_2_CHAINNAMES[pdbid]
		pdb_file = os.path.join(args.pdbdir, f"pdb{pdbid.lower()}.ent")
		r_0 = DICT_PDBID_2_CDRS[pdbid][-1][0]
		r_m1 = DICT_PDBID_2_CDRS[pdbid][-1][-1]
		res1, res2 = r_0.get_full_id()[-1][1], r_m1.get_full_id()[-1][1]

		print("*"*400)
		print("""pdb_file, res1, res2, ligand_chain""")
		print(pdb_file, res1, res2, ligand_chain)
		try:
			run_hbplus(pdb_file, res1, res2, ligand_chain)
		except:
			continue
	# quit()
	print('done!')
    
    


if __name__ == '__main__':
    """
    usage:
    python run_ligplot.py --dict_pdbid_2_cdrs ../data/20230817_020156__DICT_PDBID_2_CDRS.pickle 
	"""

    import argparse
    ### Define LigPlot+ environment here

    components_cif = '/Users/kyoheikoyama/workspace/LigPlus/lib/params/components.cif' # Location of components.cif
    ligplot_plus = '/Users/kyoheikoyama/workspace/LigPlus/lib/exe_mac/' # Location of your LigPlus executable folder
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dict_pdbid_2_chainnames", type=str, default="../data/DICT_PDBID_2_CHAINNAMES.json")
    parser.add_argument("--dict_pdbid_2_cdrs", type=str, default="../data/20230817_020156__DICT_PDBID_2_CDRS.pickle")
    parser.add_argument("--ligplot_root", type=str, default="/Users/kyoheikoyama/workspace/ligplottmp")
    parser.add_argument("--pdbdir", type=str, default='../analysis/zipdata/pdb')
    args = parser.parse_args()
    LIGPLOT_ROOT = args.ligplot_root
    os.system(f"mkdir -p {LIGPLOT_ROOT}/hbadd_hbplus_result/")
    DICT_PDBID_2_CDRS = pickle.load(open(args.dict_pdbid_2_cdrs, "rb"))
    DICT_PDBID_2_CHAINNAMES = json.load(open(args.dict_pdbid_2_chainnames))
    for pdbid in tqdm(list(DICT_PDBID_2_CDRS.keys())):
        lowerpdb = pdbid.lower()
        if os.path.exists(os.path.join(LIGPLOT_ROOT, f"hbadd_hbplus_result/pdb{lowerpdb}.hhb")) or\
			os.path.exists(os.path.join(LIGPLOT_ROOT, f"hbadd_hbplus_result/pdb{lowerpdb}.nnb")):
            print("already done", lowerpdb)
            continue
        run_hbplus(os.path.join(args.pdbdir, f"pdb{lowerpdb}.ent"))
    print('done!')
    
