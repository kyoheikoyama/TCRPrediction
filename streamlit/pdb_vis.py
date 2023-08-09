import streamlit as st
import pandas as pd
import numpy as np
import os

def pdb_vis():
    st.title('PyMol Commnad for PDB Visualization')

    pdbid = st.text_input('PDB ID', '1D9K')
    st.write('your pdbid is', pdbid.upper())

    files = os.listdir('./streamlit/pymolcommand_ribons_v2/')
    files = [f.replace('.pml','') for f in files]
    st.write(f"Available IDs ({len(files)} IDs):")
    st.write(', '.join(files))

    with open(f'./streamlit/pymolcommand_ribons_v2/{pdbid.upper()}.pml','r') as f:
        text = f.readlines()


    st.write('Copy and Paste this to PyMol')
    #st.write('\t Colors: Peptide=Yellow, CDR_Beta=Blue,  CDR_Alpha=Red')

    text = ''.join(text)
    text = f"""
    {text}
    """
    st.code(text)

