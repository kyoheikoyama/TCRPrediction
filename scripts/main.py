import argparse, pickle, sys, os, json, datetime, pathlib
from torch import nn
import torch
import time
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers.early_stopping import EarlyStopping
from ignite.metrics.metric import Metric
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

sys.path.append("../")
sys.path.append("./")
from recipes.metrics import ROC_AUC, PR_AUC
from copy import deepcopy
from recipes.dataset import TCRDataset, MCPASDataset
from recipes.model import TCRModel, SPBModel, SelfOnAll
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from attention_extractor import *
from dataset_selector import dataset_select

yymmddhhmmss = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
print("yymmddhhmmss", yymmddhhmmss)

device = "cuda" if torch.cuda.is_available() else "cpu"

# fix random seeds
seed = 9
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# for logging
GLOBAL_Validation_score_on_each_epoch = list()


def to_ignite_metric(m: nn.Module):
    """The common task-independent converter. DO NOT use this for loss."""

    class IgniteMetric(Metric):
        def __init__(self, output_transform=lambda x: x):
            super().__init__(output_transform=output_transform)
            self._hh = torch.tensor([]).float()
            self._yy = torch.tensor([]).float()
            self._m = deepcopy(m)

        def reset(self):
            self._hh = torch.tensor([]).float()
            self._yy = torch.tensor([]).float()

        def update(self, output):
            hh, yy = output
            self._hh = torch.cat([self._hh, hh.float().detach().cpu()], dim=0)
            self._yy = torch.cat([self._yy, yy.float().detach().cpu()], dim=0)

        def compute(self):
            val = self._m(self._hh, self._yy)
            return val

    return IgniteMetric


def main(args):
    # parameters
    with open(
        f"{pathlib.Path(__file__).parent.absolute()}/../hpo_params/{args.params}", "r"
    ) as fp:
        hparams = json.load(fp)

    VALID_LOG_KEY = hparams["VALID_LOG_KEY"]  # = early_stopping_target
    LOGKEY_MINBETTER = bool(hparams["LOGKEY_MINBETTER"])

    d_model = hparams["d_model"]
    d_ff = hparams["d_ff"]
    n_head = hparams["n_head"]
    n_local_encoder = hparams["n_local_encoder"]
    n_global_encoder = hparams["n_global_encoder"]
    dropout = hparams["dropout"]
    max_tolerance, max_epochs = 10, 300
    batch_size = hparams["batch_size"]
    lr = hparams["lr"]

    d_model, d_ff, n_head, n_local_encoder, n_global_encoder = (
        hparams["d_model"],
        hparams["d_ff"],
        hparams["n_head"],
        hparams["n_local_encoder"],
        hparams["n_global_encoder"],
    )
    dropout = hparams["dropout"]
    kfold = int(hparams["kfold"]) if args.kfold is None else int(args.kfold)
    (
        df_all,
        dataset_train,
        dataset_valid,
        dataset_test,
        n_tok,
        n_pos1,
        n_pos2,
        n_seg,
    ) = dataset_select(args.dataset, args.spbtarget, kfold)

    if args.dev:
        max_tolerance, max_epochs = 2, 2

    kwargs = {"num_workers": 8, "pin_memory": True} if device == "cuda" else {}

    for name, dset in zip(
        ["tra", "val", "tes"], [dataset_train, dataset_valid, dataset_test]
    ):
        print("len(dset)", len(dset), name)

    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True, **kwargs
    )
    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid, batch_size=batch_size, shuffle=False, **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False, **kwargs
    )

    if args.modeltype == "self_on_all":
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

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0], device=device))

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optim,
        loss_fn=loss_fn,
        device=device,
    )

    val_metrics_dict = {
        "accuracy": Accuracy(),
        "xent": Loss(loss_fn),
        "roc_auc": to_ignite_metric(ROC_AUC())(),
        "pr_auc_on_one": to_ignite_metric(PR_AUC(1))(),
        "pr_auc_on_zero": to_ignite_metric(PR_AUC(0))(),
    }

    evaluator = create_supervised_evaluator(
        model=model,
        metrics=val_metrics_dict,
        device=device,
        # prepare_batch=prepare_batch,
    )

    @trainer.on(Events.ITERATION_COMPLETED(every=20))
    def log_training_loss(trainer):
        print(
            "  -*-*-* Epoch[{}] Loss: {:.4f}".format(
                trainer.state.epoch, trainer.state.output
            )
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        text = f"Train Results - Epoch: {trainer.state.epoch}. "
        for k, v in metrics.items():
            text += f"Avg {k}: {v:.3f} "
        print(text)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(valid_loader)
        metrics = evaluator.state.metrics
        text = f"Validation Results - Epoch: {trainer.state.epoch}. "
        GLOBAL_Validation_score_on_each_epoch.append(metrics[VALID_LOG_KEY])
        for k, v in metrics.items():
            text += f"Avg {k}: {v:.3f} "
        print(text)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_test_results(trainer):
        evaluator.run(test_loader)
        metrics = evaluator.state.metrics
        text = f"Test Results - Epoch: {trainer.state.epoch}. "
        for k, v in metrics.items():
            text += f"Avg {k}: {v:.4f} "
        print(text)
    
    # def apply_non_negative_constraint(engine):
    #     for param in model.parameters():
    #         param.data.clamp_(min=0)  # Ensure non-negative values
    # trainer.add_event_handler(Events.ITERATION_COMPLETED, apply_non_negative_constraint)

    evaluator.add_event_handler(
        Events.COMPLETED,
        EarlyStopping(
            patience=max_tolerance,
            score_function=lambda engine: -engine.state.metrics[VALID_LOG_KEY]
            if LOGKEY_MINBETTER
            else engine.state.metrics[VALID_LOG_KEY],
            trainer=trainer,
        ),
    )

    if os.path.exists("/media/kyohei/forAI/tcrpred"):
        CHECK_POINT_DIR = f"/media/kyohei/forAI/tcrpred/checkpoint/{yymmddhhmmss}"
        LOGDIR = "/media/kyohei/forAI/tcrpred/hhyylog"
    else:
        LOGDIR = f"{pathlib.Path(__file__).parent.absolute()}/../hhyylog"

        CHECK_POINT_DIR = f"{pathlib.Path(__file__).parent.absolute()}/../checkpoint/{yymmddhhmmss}"

    os.system(f"mkdir -p {LOGDIR}")
    os.system(f"mkdir -p {CHECK_POINT_DIR}")
    print("checkpoint:", CHECK_POINT_DIR, "will be written as an ignite checkpoint")

    to_save = {"model": model, "optimizer": optim, "trainer": trainer}
    handler = Checkpoint(
        to_save,
        DiskSaver(CHECK_POINT_DIR, create_dir=True),
        n_saved=5,
        filename_prefix=f"best_{yymmddhhmmss}",
        score_function=lambda engine: -engine.state.metrics[VALID_LOG_KEY]
        if LOGKEY_MINBETTER
        else engine.state.metrics[VALID_LOG_KEY],
        score_name=VALID_LOG_KEY,
        global_step_transform=global_step_from_engine(trainer),
    )
    evaluator.add_event_handler(Events.COMPLETED, handler)

    ####################################################
    ## Ignite the loop
    trainer.run(train_loader, max_epochs=max_epochs)
    ####################################################

    ## Get the prediction and the best model
    to_load = {"model": model, "optimizer": optim, "trainer": trainer}

    ## Resume engine’s run from a state. User can load a state_dict and run engine starting from the state
    if LOGKEY_MINBETTER:
        best_epoch = 1 + GLOBAL_Validation_score_on_each_epoch.index(
            min(GLOBAL_Validation_score_on_each_epoch)
        )
    else:
        best_epoch = 1 + GLOBAL_Validation_score_on_each_epoch.index(
            max(GLOBAL_Validation_score_on_each_epoch)
        )
    trainer.load_state_dict({
        "seed": seed,
        "epoch": best_epoch,
        "max_epochs": max_epochs,
        "epoch_length": len(train_loader),
    })

    print("last checkpoint_fp:", handler.last_checkpoint)
    points = [
        float(x.replace(".pt", "").split("=")[-1]) for x in os.listdir(CHECK_POINT_DIR)
    ]
    points_index = points.index(max(points))
    best_checkpoint_fp = os.path.join(
        CHECK_POINT_DIR, os.listdir(CHECK_POINT_DIR)[points_index]
    )
    print("best_checkpoint_fp:", best_checkpoint_fp)
    Checkpoint.load_objects(to_load=to_load, checkpoint=torch.load(best_checkpoint_fp))

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        hhlist, yylist = [], []
        for xx, yy in test_loader:
            __xx = [x.to(device) for x in xx]
            __hh = model(__xx)
            hhlist.append(pd.DataFrame(__hh.cpu().detach().numpy()))
            yylist.append(pd.DataFrame(yy.cpu().detach().numpy()))

    dfh = pd.concat(hhlist).rename(columns={0: "ypred0", 1: "ypred1",})
    dfy = pd.concat(yylist).rename(columns={0: "ytrue"})
    dfhhyy = pd.concat([dfh, dfy], axis=1)
    datasettest = dataset_test.data
    for col in ["ypred0", "ypred1", "ytrue"]:
        datasettest[col] = dfhhyy[col].values

    datasettest[["ytrue", "ypred0","ypred1","tcra","tcrb","peptide",]]\
        .to_parquet(f"{LOGDIR}/{yymmddhhmmss}_k{kfold}_datasettest.parquet")


    print("saved: ", f"{LOGDIR}/{yymmddhhmmss}_k{kfold}_datasettest.parquet")

    #######################################################
    # save Attention
    #######################################################

    if args.save_attention:
        explain_model = Explain_TCRModel(
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

        for name, dset in zip(
            ["tra", "val", "tes"], [dataset_train, dataset_valid, dataset_test]
        ):
            print("len", name, len(dset))
            results_attention = []
            analysis_loader = torch.utils.data.DataLoader(
                dataset=dset, batch_size=1, shuffle=False
            )
            for i, (xx, yy) in enumerate(analysis_loader):
                result_tuple = get_attention_weights(
                    [x.to(device) for x in xx],
                    model,
                    explain_model=explain_model,
                    device=device,
                )
                result = result_tuple + tuple(yy.cpu().numpy())
                results_attention.append(result)
            pickle.dump(
                results_attention,
                open(f"{LOGDIR}/{yymmddhhmmss}_results_attention_{name}.pickle", "wb"),
            )
            print("saved: ", f"{LOGDIR}/{yymmddhhmmss}_results_attention_{name}.pickle")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    parser.add_argument("--withcomet", action="store_true")
    parser.add_argument("--params")
    parser.add_argument("--dataset")
    parser.add_argument("--kfold", type=int)
    parser.add_argument("--spbtarget", default=None)
    parser.add_argument("--save_attention", action="store_true", default=False)
    parser.add_argument("--modeltype", default="cross", choices=["self_on_all", "cross"]) #self_on_all, cross

    args = parser.parse_args()

    if args.spbtarget == "None":
        args.spbtarget = None

    withcomet = args.withcomet
    time.sleep(1)

    print("args.dataset", args.dataset)
    print("spbtarget", args.spbtarget)
    print("torch.cuda.is_available()", torch.cuda.is_available())
    print("torch.cuda.device_count()", torch.cuda.device_count())

    main(args)
