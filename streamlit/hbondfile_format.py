import streamlit as st
import os, pickle, functools
import pandas as pd

from streamlit_utils import DICT_PDBID_2_Atten12, DICT_PDBID_2_MELTDIST, STD_THH, DICT_PDBID_2_model_out, DICT_PDBID_2_CHAINNAMES, DICT_PDBID_2_CDRS, DICT_PDBID_2_RESIDUES, POSITIVE_PRED_IDS
from streamlit_utils import get_attention_and_hhb_relationship, read_hhb_file, read_hhb_text


flatten = lambda xxlis: [l for lis in xxlis for l in lis]


@functools.lru_cache(maxsize=200)
def pickleload(p):
    return pickle.load(open(p, "rb"))

def hbondfile_vis():
    root = './ligplottmp/hbadd_hbplus_result/'
    filelist = sorted(os.listdir(root))
    filelist = [f for f in filelist if '.hhb' in f]
    st.write(f".hhb files are in `{root}` ")
    pdbid = st.text_input('PDB ID', '1D9K').upper()
    st.write(', '.join([f.replace('.hhb','').replace('pdb','').upper() for f in filelist]))

    ra = st.radio(
        "Choose original .hhb file or only cdr chain",
        ('Original', 'OnlyCDR/Peptide'))

    chains = DICT_PDBID_2_CHAINNAMES[pdbid.upper()]
    chainsletters = ''.join(chains)
    st.code(list(zip(["alphaname", "betaname", "peptidename"], chains)))
    
    chains_flatten = [c.split(', ') for c in chains]
    chains_flatten = flatten(chains_flatten)
    if ra == 'Original':
        st.write(f'hhb file of {pdbid} (Original)')
        with open(f'{root}pdb{pdbid.lower()}.hhb', 'r') as f:
            text = f.readlines()
    
        st.code(''.join(text))
    elif ra == 'OnlyCDR/Peptide':
        st.write(f'hhb file of {pdbid.upper()} (Only CDR chains or Peptide Chains)')
        st.write('chains =', chains)
        df = read_hhb_file(pdbid.upper())
        donors_n_acceptors = [read_hhb_text(row.item()) for i, row in df.iterrows() 
                              ]
        df_donors_n_acceptors = pd.DataFrame(donors_n_acceptors)

        df_donors_n_acceptors['donor_res_name'] = df_donors_n_acceptors['donor_res'].str.split('/').apply(lambda xx: xx[1][0])
        df_donors_n_acceptors['acceptor_res_name'] = df_donors_n_acceptors['acceptor_res'].str.split('/').apply(lambda xx: xx[1][0])
        
        df_donors_n_acceptors['donor_res_TF'] = df_donors_n_acceptors['donor_res_name'].isin(chains_flatten)
        df_donors_n_acceptors['acceptor_res_TF'] = df_donors_n_acceptors['acceptor_res_name'].isin(chains_flatten)
        df_donors_n_acceptors = df_donors_n_acceptors.query('donor_res_TF==True and acceptor_res_TF==True')
        st.dataframe(df_donors_n_acceptors)

    else:
        pass

    st.write('# File Format ')
    st.code("""
    ============================================================================
                             Table I: *.hb2 format
         https://github.com/mmravic314/bin/blob/master/HBplus/hbplus.man
    
    01-13 Donor Atom, including . . .
        01    Chain ID (defaults to '-')
        02-05 Residue Number 
        06    Insertion Code (defaults to '-')
        07-09 Amino Acid Three Letter Code
        10-13 Atom Type Four Letter Code
    
    15-27 Acceptor Atom, same format as Donor atom
    28-32 Donor - Acceptor distance, in Angstroms
    34-35 Atom Categories - M(ain-chain), S(ide-chain) or H(etatm) - of D & A
    37-39 Gap between donor and acceptor groups, in amino acids
          (-1 if not applicable)
    41-45 Distance between the CA atoms of the donor and acceptor residues
          (-1 if one of the two atoms is in a hetatm)
    47-51 Angle formed by the Donor and Acceptor at the hydrogen, in degrees.
          (-1 if the hydrogen is not defined)
    53-57 Distance between the hydrogen and the Acceptor, in Angstroms
          (-1 if the hydrogen is not defined)
    59-63 The smaller angle at the Acceptor formed by the hydrogen and an
          acceptor antecedent (-1 if the hydrogen, or the acceptor antecedent,
          is not defined)
    65-69 The smaller angle at the Acceptor formed by the donor and an acceptor
          antecedent (-1 if not applicable)
    71-75 Count of hydrogen bonds
    ============================================================================
    """)