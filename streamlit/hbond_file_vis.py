import streamlit as st
import pandas as pd
import numpy as np
import os
from streamlit_utils import read_hhb_file, read_hhb_text


def hbond_file_vis():
    st.title('HBond File Visualization')

    files = os.listdir('./ligplottmp/hbadd_hbplus_result/')
    files = sorted([f.split('.')[0].replace('pdb','').upper() for f in files])

    pdbid = st.text_input('PDB ID', '1D9K')

    st.write('your pdbid is', pdbid.upper())
    st.write(f"Available IDs ({len(files)} IDs):")
    st.write(', '.join(files))

    textdata_by_bond =  read_hhb_file(pdbid)

    list_of_dict_donors_n_acceptors = [read_hhb_text(row.item()) for i, row in textdata_by_bond.iterrows()]
    df_data_donors_n_acceptors = pd.DataFrame(list_of_dict_donors_n_acceptors)


    st.write('## hbond data', pdbid)
    st.dataframe(df_data_donors_n_acceptors)


    st.write('## hbond text data', pdbid)

    normaltext = '\n'.join(textdata_by_bond.values.ravel())
    st.code(normaltext)



