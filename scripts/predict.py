#!/usr/bin/env python
# coding: utf-8

import os
import sys
import pandas as pd
import numpy as np
import pandas as pd
import pickle, sys
import pathlib
import json
import torch
from attention_extractor import TCRModel
from tqdm import tqdm
sys.path.append("../analysis/")
# import plotly.express as px

sys.path.append("../")
sys.path.append("../..")
from recipes.dataset import MCPASDataset

# Ignite
from ignite.engine import create_supervised_trainer
from ignite.handlers import Checkpoint
from matplotlib import pyplot as plt

MAXLENGTH_A, MAXLENGTH_B, max_len_epitope = 28, 28, 25
KFOLD_I = 3

SEED = 9
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


def main(ckptpath, dt, output_filepath, args):
    dfinput = pd.read_parquet(args.input_filepath).reset_index(drop=True)
    if args.input_filepath == "../data/panpep_zeroshot.parquet":
        dfinput["sign"] = dfinput["Label"]
        dfinput["peptide"] = dfinput["Peptide"]
        dfinput["tcrb"] = dfinput["TCR"]
        dfinput["tcra"] = "CAASETSYDKVIF"

    # Load model
    with open(f"../hpo_params/best.json", "r") as fp:
        hparams = json.load(fp)

    # device = "cpu"  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # fix random seeds
    d_model, d_ff, n_head, n_local_encoder, n_global_encoder = (
        hparams["d_model"],
        hparams["d_ff"],
        hparams["n_head"],
        hparams["n_local_encoder"],
        hparams["n_global_encoder"],
    )
    dropout = hparams["dropout"]
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]

    n_tok = 29  # NUM_VOCAB
    n_pos1 = 62  # MAX_LEN_AB (sum of maxlens of a and b)
    n_pos2 = 26  # MAX_LEN_Epitope
    n_seg = 3

    """Model, optim, trainer"""
    model = TCRModel(
        d_model=d_model,
        d_ff=d_ff,
        n_head=n_head,
        n_local_encoder=n_local_encoder,
        n_global_encoder=n_global_encoder,
        dropout=dropout,
        scope=4,
        n_tok=n_tok,
        n_pos1=n_pos1,
        n_pos2=n_pos2,
        n_seg=n_seg,
    )

    # Optimizer
    model.to(device)
    model = model.eval()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 15.0], device=device))

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        device=device,
    )

    checkpoint = torch.load(ckptpath, map_location=torch.device(device))
    checkpoint["trainer"]["seed"] = SEED

    ## Get the prediction and the best model
    to_load = {"model": model, "optimizer": optim, "trainer": trainer}

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

    ret = use_model_on_df(dfinput, model, batch_size, device=device)
    retdf = pd.DataFrame(ret, columns=["pred0", "pred1"])
    dfinput = pd.concat([dfinput, retdf], axis=1)
    dfinput.to_parquet(output_filepath)


    from scipy.special import softmax
    from sklearn.metrics import roc_auc_score, average_precision_score
    p1 = retdf.apply(softmax, axis=1).apply(lambda x: x[1])

    print("AUC: ", roc_auc_score(dfinput["sign"], p1))
    print("AP: ", average_precision_score(dfinput["sign"], p1))

    

def use_model_on_df(df, model, batch_size=128, device="cpu"):
    torch_dataset = MCPASDataset(df)
    loader = torch.utils.data.DataLoader(torch_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    model = model.eval()

    result_list = []

    with torch.no_grad():
        for i, (xx, yy) in tqdm(enumerate(loader)):
            result_tuple = model([x.to(device) for x in xx])
            result_list.append(result_tuple)

    concat_result = torch.cat(result_list, dim=0)
    concat_result = concat_result.cpu().numpy()
    return concat_result


if __name__ == "__main__":
    """
    python predict.py --model_key entire_crossatten --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet
        AUC:  0.5362159470265514
        AP:  0.18550580020451382

    python predict.py --model_key entire_cross --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet

    python predict.py --model_key entire_crossatten --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/panpep_zeroshot.parquet
        AUC:  0.5158217865029484
        AP:  0.5272058104597115
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="entire_crossatten")
    parser.add_argument("--checkpointsjson", type=str, default="../hpo_params/checkpoints.json")
    parser.add_argument(
        "--input_filepath", type=str, default="../data/recent_data_test.parquet"
    )

    args = parser.parse_args()
    output_filepath = args.input_filepath.replace(".parquet","") + f"_{args.model_key}.parquet"

    with open(args.checkpointsjson, "r") as fp:
        checkpointsjson = json.load(fp)

    dt = checkpointsjson[args.model_key]
    torch_ckptdir = f"/media/kyohei/forAI/tcrpred/checkpoint/{dt}/"
    ckptpath = torch_ckptdir + checkpointsjson[args.model_key + "_ckpt"]

    main(ckptpath, dt, output_filepath, args)
