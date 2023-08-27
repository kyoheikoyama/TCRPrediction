import pandas as pd

PropertyCOLS = ['head_all','num_bonds', 'is_connecting_to_cdr', 'is_connecting_to_pep', 'is_connecting_to_tcr',
               'is_connecting_to_notCDR_tcr', 'is_connecting_to_ownchain_tcr',
               'is_connecting_to_ownchain_cdr', 'is_connecting_to_opposite_chain_tcr',
               'is_connecting_to_opposite_chain_cdr', ]

def main(args):
    bondinfo = pd.read_parquet(args.bondinfo)
    df = pd.read_parquet(args.explained)
    

    df = pd.read_parquet("../data/mutation_study_entire_cross_newemb__explained.parquet")
    df['combined_all'] = df['pdbid'] + '__' +\
        df['tcra'] + '__' +\
        df['tcrb'] + '__' +\
        df['peptide']

    print(df['combined_all'].nunique())

    for col in PropertyCOLS:
        if col == 'num_bonds':
            bondinfo[col] = bondinfo[col].astype(str)
        if col == 'head_all':
            df[col] = df[col].astype(bool)
            df[col] = df[col].map({True:'T', False:'F'})
            continue
        bondinfo[col] = bondinfo[col].astype(bool)
        bondinfo[col] = bondinfo[col].map({True:'T', False:'F'})

    df = pd.merge(df, bondinfo, on=['pdbid', 'residue'], how='left')
    
    for col in PropertyCOLS:
        df[col] = df[col].fillna('N')
    
    result_list = []
    for i, coma in enumerate(df['combined_all'].unique()):
        item = []
        df_ = df[df['combined_all'] == coma].copy()
        a,b,c,p = df_[['tcra', 'tcrb', 'peptide', 'pdbid']].values[0]
        item += [a,b,c,p,coma]


        for col in PropertyCOLS:
            _dfcol = df_[['residue',col]]
            large_or_smmall = ''

            # print('_dfcol', _dfcol)

            for i, row in _dfcol.iterrows():
                r = row['residue']
                tf = row[col]
                if ':_' in r:
                    large_or_smmall+=':'
                else:
                    large_or_smmall += str(tf)
                    assert str(tf) in ['T', 'F', 'N']
                
            item += [large_or_smmall]
            
        result_list.append(item)

    df = pd.DataFrame(result_list, 
        columns=['tcra', 'tcrb', 'peptide', 'pdbid','combined_all'] + PropertyCOLS).rename(columns={'head_all':'is_large_atten'})
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bondinfo", type=str, default="../data/20230828_015709__df_bondinfo.parquet")
    parser.add_argument("--explained", type=str, default="../data/mutation_study_entire_cross_newemb__explained.parquet")

    # save file name
    parser.add_argument("--output", type=str, default="../data/mutation_study_result.parquet")

    args = parser.parse_args()
    df = main(args)

    df.to_parquet(args.output)
    print('saved to', args.output)

