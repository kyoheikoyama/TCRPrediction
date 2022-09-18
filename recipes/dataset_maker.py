# %load ~/jupyter_notebook/user_work/tcrpred/recipes/dataset.py
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import sys, pickle
sys.path.append('../')
from recipes.utils import get_03_data, get_all_combination_and_fill_nagatives, get_infer_report, get_df_from_path

# from utils import get_03_data, get_all_combination_and_fill_nagatives, get_infer_report
import argparse, pickle, sys, os, json, datetime, pathlib
from torch import nn
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers.early_stopping import EarlyStopping
from ignite.metrics.metric import Metric
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

from recipes.metrics import ROC_AUC, PR_AUC
from copy import deepcopy
from recipes.dataset import TCRDataset, MCPASDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold



def make_dataset_and_params_from_args(args):
    if args.dataset=='originalvdjdb':
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

    elif args.dataset == "mcpas":
        datapath = f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/mcpas_train_samples.pickle"
        df_all = pd.DataFrame(pickle.load(open(datapath, "rb")))
        if not (args.spbtarget is None):
            df_all = df_all[df_all['peptide']==args.spbtarget].reset_index(drop=True)
            df_all.drop(columns=['tcra'])
            df_all['tcra'] = 'X'
            print(df_all['sign'].value_counts())
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][args.kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        dataset_test = MCPASDataset(
            pd.DataFrame(
                pickle.load(
                    open(
                        f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/mcpas_test_samples.pickle",
                        "rb",
                    )
                )
            )
        )
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 52  # MAX_LEN_AB (sum of maxlens of a and b)
        n_pos2 = 28  # MAX_LEN_Epitope
        n_seg = 3
    elif args.dataset == "vdjdbno10x":
        datapath = f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/vdjdb_no10x_train_samples.pickle"
        df_all = pd.DataFrame(pickle.load(open(datapath, "rb")))
        if not (args.spbtarget is None):
            df_all = df_all[df_all['peptide']==args.spbtarget].reset_index(drop=True)
            df_all.drop(columns=['tcra'])
            df_all['tcra'] = 'X'
        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [
            (train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][args.kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        df_test = pd.DataFrame(pickle.load(
                    open(
                f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/vdjdb_no10x_test_samples.pickle",
                    "rb",)))
        if not (args.spbtarget is None):
            df_test = df_test[df_test['peptide']==args.spbtarget].reset_index(drop=True)
            df_test.drop(columns=['tcra'])
            df_test['tcra'] = 'X'
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 21  # MAX_LEN_Epitope
        n_seg = 3

    elif args.dataset == "alltrain":
        p_list = [f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/vdjdb_no10x_train_samples.pickle",
                  f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/mcpas_train_samples.pickle",]
        df_all = get_df_from_path(p_list)
        if not (args.spbtarget is None):
            df_all = df_all[df_all['peptide']==args.spbtarget].reset_index(drop=True)
            df_all.drop(columns=['tcra'])
            df_all['tcra'] = 'X'
            print(df_all['sign'].value_counts())

        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [(train_index, valid_index) for train_index, valid_index in kf.split(df_all)
        ][args.kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        p_list = [f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/vdjdb_no10x_test_samples.pickle",
                  f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/mcpas_test_samples.pickle",]
        df_test = get_df_from_path(p_list)

        if not (args.spbtarget is None):
            df_test = df_test[df_test['peptide']==args.spbtarget].reset_index(drop=True)
            df_test.drop(columns=['tcra'])
            df_test['tcra'] = 'X'
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 26  # MAX_LEN_Epitope
        n_seg = 3
    elif args.dataset == "all_withNoTest":
        p_list = [f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/vdjdb_no10x_train_samples.pickle",
                  f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/mcpas_train_samples.pickle",
                 f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/vdjdb_no10x_test_samples.pickle",
                 f"{pathlib.Path(__file__).parent.absolute()}/../../external_data/ERGO-II/Samples/mcpas_test_samples.pickle",
                 ]
        df_all = get_df_from_path(p_list)
        if not (args.spbtarget is None):
            df_all = df_all[df_all['peptide']==args.spbtarget].reset_index(drop=True)
            df_all.drop(columns=['tcra'])
            df_all['tcra'] = 'X'
            print(df_all['sign'].value_counts())

        kf = KFold(n_splits=5, shuffle=True, random_state=2)
        train_index, valid_index = [(train_index, valid_index) for train_index, valid_index in kf.split(df_all)][args.kfold]
        df_train, df_valid = df_all.loc[train_index], df_all.loc[valid_index]
        dataset_train, dataset_valid = MCPASDataset(df_train), MCPASDataset(df_valid)
        df_test = get_df_from_path(p_list)
        if not (args.spbtarget is None):
            df_test = df_test[df_test['peptide']==args.spbtarget].reset_index(drop=True)
            df_test.drop(columns=['tcra'])
            df_test['tcra'] = 'X'
        dataset_test = MCPASDataset(df_test)
        n_tok = 29  # NUM_VOCAB
        n_pos1 = 62  # MAX_LEN_AB
        n_pos2 = 26  # MAX_LEN_Epitope
        n_seg = 3
    else:
        assert False
    
    return  dataset_train, dataset_valid, dataset_test, n_tok, n_pos1, n_pos2, n_seg