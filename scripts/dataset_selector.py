import pickle
import pandas as pd
import pathlib
from recipes.dataset import TCRDataset, MCPASDataset
from sklearn.model_selection import KFold

def add_tcrspep(df, setting="onlyseq"):
    if setting=="noTCellType":
        for i, col in enumerate(["tcra","tcrb","peptide","va","ja","vb","ja","mhc"]):
            if i ==0: ser = df[col]
            else: ser += ":" + df[col]
        df["tcrs_pep"] = ser
    elif setting=="onlyseq":
        df["tcrs_pep"] = df["tcra"] + ":" + df["tcrb"] + "&"  + df["peptide"] 
    
    return df


def get_df(datapath):
    return pd.DataFrame(pickle.load(open(datapath, "rb")))


def get_df_from_path(p_list):
    return pd.concat([get_df(d) for d in p_list]).reset_index(drop=True)


def dataset_select(name, spbtarget=None, kfold=0):
    if name == "originalvdjdb":
        dataset_trainvalid = TCRDataset(
            f"{pathlib.Path(__file__).parent.absolute()}/../data/03.VDJdb.tsv",
            donors=["Donor1", "Donor2", "Donor3"],
            kfold=kfold,
        )
        dataset_test = TCRDataset(
            f"{pathlib.Path(__file__).parent.absolute()}/../data/03.VDJdb.tsv",
            donors=["Donor4"],
        )
        n_tok = 24  # NUM_VOCAB
        n_pos1 = 50  # MAX_LEN_AB (sum of maxlens of a and b)
        n_pos2 = 25  # MAX_LEN_Epitope
        n_seg = 5
    elif name == "mcpas":
        datapath = f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_train_samples.pickle"
        df_all = pd.DataFrame(pickle.load(open(datapath, "rb")))
        if not (spbtarget is None):
            df_all = df_all[df_all["peptide"] == spbtarget].reset_index(drop=True)
            df_all.drop(columns=["tcra"])
            df_all["tcra"] = "X"
            print(df_all["sign"].value_counts())
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        df_test = pd.DataFrame(pickle.load(
                    open(
                        f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_test_samples.pickle",
                        "rb",
                    )
                )
        )
        df_all = add_tcrspep(df_all)
        df_test = add_tcrspep(df_test)

        df_test = df_test[~df_test["tcrs_pep"].isin(df_all["tcrs_pep"])]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 52  # MAX_LEN_AB (sum of maxlens of a and b)
        n_pos2 = 28  # MAX_LEN_Epitope
        n_seg = 3
        if kfold==-1 or kfold=="-1":
            return (
                df_all,
                MCPASDataset(df_all), MCPASDataset(df_all),
                dataset_test,
                n_tok,
                n_pos1,
                n_pos2,
                n_seg,
            )
    elif name == "vdjdbno10x":
        datapath = f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_no10x_train_samples.pickle"
        df_all = pd.DataFrame(pickle.load(open(datapath, "rb")))
        if not (spbtarget is None):
            df_all = df_all[df_all["peptide"] == spbtarget].reset_index(drop=True)
            df_all.drop(columns=["tcra"])
            df_all["tcra"] = "X"
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        if kfold==-1 or kfold=="-1":
            kfold=0 # for temporary use
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        df_test = pd.DataFrame(
            pickle.load(
                open(
                    f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_no10x_test_samples.pickle",
                    "rb",
                )
            )
        )

        df_all = add_tcrspep(df_all)
        df_test = add_tcrspep(df_test)
        df_test = df_test[~df_test["tcrs_pep"].isin(df_all["tcrs_pep"])]

        if not (spbtarget is None):
            df_test = df_test[df_test["peptide"] == spbtarget].reset_index(drop=True)
            df_test.drop(columns=["tcra"])
            df_test["tcra"] = "X"
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 21  # MAX_LEN_Epitope
        n_seg = 3
        if kfold==-1 or kfold=="-1":
            return (
                df_all,
                MCPASDataset(df_all), MCPASDataset(df_all),
                dataset_test,
                n_tok,
                n_pos1,
                n_pos2,
                n_seg,
            )        
    elif name == "alltrain":
        p_list = [
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_train_samples.pickle",
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_train_samples.pickle",
        ]
        df_all = get_df_from_path(p_list)
        if not (spbtarget is None):
            df_all = df_all[df_all["peptide"] == spbtarget].reset_index(drop=True)
            df_all.drop(columns=["tcra"])
            df_all["tcra"] = "X"
            print(df_all["sign"].value_counts())

        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        p_list = [
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_test_samples.pickle",
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_test_samples.pickle",
        ]
        df_test = get_df_from_path(p_list)

        if not (spbtarget is None):
            df_test = df_test[df_test["peptide"] == spbtarget].reset_index(drop=True)
            df_test.drop(columns=["tcra"])
            df_test["tcra"] = "X"
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 26  # MAX_LEN_Epitope
        n_seg = 3
    elif name == "all":
        p_list = [
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_train_samples.pickle",
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_train_samples.pickle",
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_test_samples.pickle",
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_test_samples.pickle",
        ]
        df_all = get_df_from_path(p_list)
        if not (spbtarget is None):
            df_all = df_all[df_all["peptide"] == spbtarget].reset_index(drop=True)
            df_all.drop(columns=["tcra"])
            df_all["tcra"] = "X"
            print(df_all["sign"].value_counts())

        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        df_test = get_df_from_path(p_list)
        if not (spbtarget is None):
            df_test = df_test[df_test["peptide"] == spbtarget].reset_index(drop=True)
            df_test.drop(columns=["tcra"])
            df_test["tcra"] = "X"
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 26  # MAX_LEN_Epitope
        n_seg = 3
    elif name == "allwithtest":
        df_train = pd.read_parquet("../data/train_allDataTrainedModel.parquet")
        df_valid = pd.read_parquet("../data/valid_allDataTrainedModel.parquet")
        dataset_train = MCPASDataset(df_train)
        dataset_test = MCPASDataset(df_valid)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 26  # MAX_LEN_Epitope
        n_seg = 3        
        dataset_valid = dataset_train
        df_all = pd.concat([df_train, df_valid])

    elif name == "fake_v0":
        p_list = [
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_no10x_train_samples.pickle",
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_train_samples.pickle",
        ]
        df_all = pd.read_parquet(
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/fakedata_v0.parquet"
        )
        if not (spbtarget is None):
            df_all = df_all[df_all["peptide"] == spbtarget].reset_index(drop=True)
            df_all.drop(columns=["tcra"])
            df_all["tcra"] = "X"
            print(df_all["sign"].value_counts())

        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        df_test = pd.read_parquet(
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/fakedata_v0.parquet"
        )
        if not (spbtarget is None):
            df_test = df_test[df_test["peptide"] == spbtarget].reset_index(drop=True)
            df_test.drop(columns=["tcra"])
            df_test["tcra"] = "X"
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 26  # MAX_LEN_Epitope
        n_seg = 3
    elif name == "test":
        p_list = [
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/vdjdb_no10x_train_samples.pickle",
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/mcpas_train_samples.pickle",
        ]
        df_all = pd.read_parquet(
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/fakedata_v0.parquet"
        ).head(1000)
        if not (spbtarget is None):
            df_all = df_all[df_all["peptide"] == spbtarget].reset_index(drop=True)
            df_all.drop(columns=["tcra"])
            df_all["tcra"] = "X"
            print(df_all["sign"].value_counts())

        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        df_test = pd.read_parquet(
            f"{pathlib.Path(__file__).parent.absolute()}/../external_data/ERGO-II/Samples/fakedata_v0.parquet"
        ).head(1000)
        if not (spbtarget is None):
            df_test = df_test[df_test["peptide"] == spbtarget].reset_index(drop=True)
            df_test.drop(columns=["tcra"])
            df_test["tcra"] = "X"
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 26  # MAX_LEN_Epitope
        n_seg = 3
    else:
        assert False
    return (
        df_all,
        dataset_train,
        dataset_valid,
        dataset_test,
        n_tok,
        n_pos1,
        n_pos2,
        n_seg,
    )
