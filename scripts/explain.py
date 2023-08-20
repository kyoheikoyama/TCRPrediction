#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import pandas as pd
import pickle, sys
# import boto3
import pathlib, json
import torch
import warnings
warnings.filterwarnings("ignore")

sys.path.append('../scripts/')
from attention_extractor import TCRModel, Explain_TCRModel
sys.path.append('../analysis/')

sys.path.append('../')
sys.path.append('../..')
from sklearn.model_selection import KFold
from recipes.dataset import MCPASDataset
from recipes.model import SelfOnAll

# Ignite
sys.path.append('../scripts/')
from analysis_util import get_mat_from_result_tuple
from ignite.engine import Events, create_supervised_trainer
from ignite.handlers import Checkpoint
from matplotlib import pyplot as plt

import sys
sys.path.append('./../streamlit/')


# ＃ーーーーーーーーーーーーーーーーーーーーーー---------------------------------------------------------------------------------------------------------------------------------
# ＃ーーーーーーーーーーーーーーーーーーーーーー---------------------------------------------------------------------------------------------------------------------------------
# ＃ーーーーーーーーーーーーーーーーーーーーーー---------------------------------------------------------------------------------------------------------------------------------
# ＃ーーーーーーーーーーーーーーーーーーーーーー---------------------------------------------------------------------------------------------------------------------------------
# ＃ーーーーーーーーーーーーーーーーーーーーーー---------------------------------------------------------------------------------------------------------------------------------

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



def _get_tuple_result(alpha, beta, pep, model, explain_model, device='cpu'):
    df = pd.DataFrame({'sign':1, 'tcra':alpha,'tcrb':beta, 'peptide':pep}, 
                      index=[0])

    torch_dataset = MCPASDataset(df)
    analysis_loader = torch.utils.data.DataLoader(torch_dataset)

    with torch.no_grad():
        for i, (xx,yy) in enumerate(analysis_loader):
            result_tuple = get_attention_weights([x.to(device) for x in xx], 
                                           model, explain_model=explain_model, device=device)
            
    
    return result_tuple
    

def get_attention_weights(__xx, model, explain_model, device='cpu'):
    from scipy.special import expit, softmax
    #print(__xx)
    explain_model.load_state_dict(model.state_dict(), strict=False)
    model.to(device)
    explain_model = explain_model.to(device)
    model = model.eval()
    explain_model = explain_model.eval()
    
    with torch.no_grad():
        tcrab, epitope, src_kwargs, tgt_kwargs = explain_model(__xx)  #  (L,B,E)
        attn_output1, attn_output_weights1 = model.enc.cgr.mha.fg.f.mha(
            query=tcrab,  # target  27
            key=epitope,   # source  10  (source is usually the output of encoder)
            value=epitope,   # source  10  
            **{'key_padding_mask': tgt_kwargs['memory_key_padding_mask']}
        )

        attn_output2, attn_output_weights2 = model.enc.cgr.mha.fg.g.f.mha(
            query=epitope, 
            key=tcrab, 
            value=tcrab,
            **{'key_padding_mask': src_kwargs['memory_key_padding_mask']}
        )
        
        ypred = model(__xx)
    
    return (attn_output_weights1[0].cpu().numpy(), 
                             attn_output_weights2[0].cpu().numpy(),  
                            float(softmax(ypred.cpu().numpy())[0][1]))


def get_attended_self(alpha, beta, pep, model, explain_model, device='cpu', ONLY_HEAD_ZERO=False):
    df = pd.DataFrame({'sign':1, 'tcra':alpha,'tcrb':beta, 'peptide':pep}, 
                      index=[0])

    torch_dataset = MCPASDataset(df)
    analysis_loader = torch.utils.data.DataLoader(torch_dataset)
    with torch.no_grad():
        for i, (xx,yy) in enumerate(analysis_loader):
            print("[x.shape for x in xx]", [x.shape for x in xx])
            ypred = model([x.to(device) for x in xx])
            attention_values = model.atten.att
    attention_values = attention_values[0].cpu().numpy()
    # print(attention_values[1].sum(axis=1))  # this is the all one vector
    # print(attention_values[head_number_i].sum(axis=1))  # this is the all one vector

    attention_tcr4 = [(attention_values[i] > attention_values[i].ravel().mean() \
                       + 4.5 * attention_values[i].ravel().std())
                        for i in range(4)]
    attended_tcrs = pd.concat([pd.Series(a.any(axis=0)).to_frame().T for a in attention_tcr4]) #.any()
    attended_tcrs.index = [f"head_{i}" for i in range(len(attended_tcrs))]
    attended_tcrs = attended_tcrs.T
    attended_tcrs["head_all"] = attended_tcrs["head_0"] | attended_tcrs["head_1"] | attended_tcrs["head_2"] | attended_tcrs["head_3"]
    return attended_tcrs, None


    
def get_attended_cross(alpha, beta, pep, model, explain_model, device='cpu', ONLY_HEAD_ZERO=False):
    result_tuple = _get_tuple_result(alpha, beta, pep, model, explain_model, device='cpu')
    attention_pep4_tcr4 = get_mat_from_result_tuple(result_tuple, alpha,beta,pep)
    
    attention_tcr4 = [(attention_pep4_tcr4[1][i] > attention_pep4_tcr4[1][i].values.ravel().mean() \
                       + 4.5 * attention_pep4_tcr4[1][i].values.ravel().std())
                        for i in range(4)]
    attention_pep4 = [(attention_pep4_tcr4[0][i] > attention_pep4_tcr4[0][i].values.ravel().mean() \
                       + 5.5 * attention_pep4_tcr4[0][i].values.ravel().std())
                    for i in range(4)]
    attended_tcrs = pd.concat([a.any(axis=0).to_frame().T for a in attention_tcr4]) #.any()
    attended_tcrs.index = [f"head_{i}" for i in range(len(attended_tcrs))]
    attended_tcrs = attended_tcrs.T
    attended_tcrs["head_all"] = attended_tcrs["head_0"] | attended_tcrs["head_1"] | attended_tcrs["head_2"] | attended_tcrs["head_3"]
    attended_peps = pd.concat([a.any(axis=0).to_frame().T for a in attention_pep4]) #.any()
    attended_peps.index = [f"head_{i}" for i in range(len(attended_peps))]
    attended_peps = attended_peps.T
    attended_peps["head_all"] = attended_peps["head_0"] | attended_peps["head_1"] | attended_peps["head_2"] | attended_peps["head_3"]
    return attended_tcrs, attended_peps


def main(ckptpath, dt, output_filepath, args):
    dfinput = pd.read_parquet(args.input_filepath).reset_index(drop=True)
    dfinput = dfinput[dfinput.peptide.apply(len) <= 25]

    # Load model
    with open(f"../hpo_params/best.json", "r") as fp:
        hparams = json.load(fp)

    device = "cpu"  
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    if args.model_key == "entire_self":
        print("entire_self")
        model = SelfOnAll(
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
        ).to(device)

    else:
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
        ).to(device)

        explain_model = Explain_TCRModel(d_model=d_model, d_ff=d_ff, n_head=n_head, n_local_encoder=n_local_encoder, 
                                    n_global_encoder=n_global_encoder, dropout=dropout, scope=4, 
                                    n_tok=n_tok, n_pos1=n_pos1, n_pos2=n_pos2, n_seg=n_seg)


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

    if args.model_key == "entire_self":
        get_attended = get_attended_self
        explain_model = None
        pass
    else:
        # Load the best model on explain model
        get_attended = get_attended_cross
        explain_model.load_state_dict(model.state_dict(), strict=False)
        explain_model = explain_model.to(device)
        explain_model = explain_model.eval()

    dlist = []
    for i, row in tqdm(dfinput.iterrows()):
        pdbid = row["pdbid"]
        cdr_alpha = row["tcra"]
        cdr_beta = row["tcrb"]
        peptide = row["peptide"]

        attended_tcrs, attended_peps = get_attended(cdr_alpha, cdr_beta, peptide, model, explain_model, device)
        
        if args.model_key == "entire_self":
            temp_tcr = attended_tcrs.reset_index(drop=False).rename(columns={0:'is_attention_large', })
            temp_tcr["type"] = ["alpha"] * len(cdr_alpha) + [None] * (28-len(cdr_alpha)) +\
                  ["beta"] *len(cdr_beta) + [None] *(28-len(cdr_beta)) + ["peptide"] * len(peptide) + [None] * (25-len(peptide))
            temp_tcr["residue"] = res_to_number(cdr_alpha) + [None] * (28-len(cdr_alpha)) + res_to_number(cdr_beta, len(cdr_alpha)) + [None] *(28-len(cdr_beta)) + res_to_number(peptide) + [None] * (25-len(peptide))
            temp_tcr["pdbid"] = pdbid
            temp_tcr.dropna(subset=["residue"], inplace=True)
        else:
            attended_tcrs, attended_peps = get_attended(cdr_alpha, cdr_beta, peptide, model, explain_model, device)
            attended_tcrs, _ = get_attended(cdr_alpha, cdr_beta, peptide, model, explain_model, device)
            temp_tcr = attended_tcrs.reset_index(drop=False).rename(columns={0:'is_attention_large', 'index':'residue'})
            temp_pep = attended_peps.reset_index(drop=False).rename(columns={0:'is_attention_large', 'index':'residue'})
            temp_tcr["type"] = "tcr"
            temp_tcr["pdbid"] = pdbid
            temp_pep["type"] = "peptide"
            temp_pep["pdbid"] = pdbid


        dlist.append(temp_tcr)

    df = pd.concat(dlist).reset_index(drop=False).rename(columns={"index":'position'})
    print(df.sample(10))
    df.to_parquet(output_filepath)
    print("saved to ", output_filepath)

def res_to_number(res, plus=0):
    return [f"{r}_{i+plus}" for i, r in enumerate(res)]

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
    python explain.py --model_key entire_crossatten --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/pdb_complex_sequences.parquet
    python explain.py --model_key entire_self --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/pdb_complex_sequencesV2.parquet

    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_key", type=str, default="entire_crossatten")
    parser.add_argument("--checkpointsjson", type=str, default="../hpo_params/checkpoints.json")
    parser.add_argument(
        "--input_filepath", type=str, default="../data/pdb_complex_sequences.parquet"
    )

    args = parser.parse_args()
    output_filepath = args.input_filepath.replace(".parquet","") + f"_{args.model_key}__explained.parquet"

    with open(args.checkpointsjson, "r") as fp:
        checkpointsjson = json.load(fp)

    dt = checkpointsjson[args.model_key]
    torch_ckptdir = f"/media/kyohei/forAI/tcrpred/checkpoint/{dt}/"
    ckptpath = torch_ckptdir + checkpointsjson[args.model_key + "_ckpt"]

    main(ckptpath, dt, output_filepath, args)
