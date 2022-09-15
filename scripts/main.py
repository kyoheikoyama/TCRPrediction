import argparse, pickle, sys, os, json, datetime, pathlib

parser = argparse.ArgumentParser()
parser.add_argument("--dev", action="store_true")
parser.add_argument("--withcomet", action="store_true")
parser.add_argument("--params")
parser.add_argument("--dataset")
parser.add_argument("--kfold", type=int, default=0)
parser.add_argument("--spbtarget", default=None)


args = parser.parse_args()

if args.spbtarget=="None": args.spbtarget=None

kfold = int(args.kfold)
withcomet = args.withcomet
if withcomet: from comet_ml import Experiment
from torch import nn
import torch
import time
time.sleep(1)

print("kfold =", kfold, "devmode =", args.dev)
print("args.dataset", args.dataset)
print("spbtarget", args.spbtarget)
print("torch.cuda.is_available()", torch.cuda.is_available())
print("torch.cuda.device_count()", torch.cuda.device_count())
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
from recipes.model import TCRModel, SPBModel
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from attention_extractor import *

time.sleep(np.random.randint(0,40))

yymmddhhmmss = datetime.datetime.today().strftime("%Y%m%d_%H%M%S")
print("yymmddhhmmss", yymmddhhmmss)

# parameters
with open(
    f"{pathlib.Path(__file__).parent.absolute()}/../hpo_params/{args.params}", "r"
) as fp:
    hparams = json.load(fp)

if withcomet:
    experiment = Experiment(
        api_key="lzKnG53WCE1I4U4ReMx9dec7L",
        project_name="general",
        workspace="kkoyama",
    )
    experiment.log_parameters(hparams)

device = "cuda" if torch.cuda.is_available() else "cpu"

d_model = hparams["d_model"]
d_ff = hparams["d_ff"]
n_head = hparams["n_head"]
n_local_encoder = hparams["n_local_encoder"]
n_global_encoder = hparams["n_global_encoder"]
dropout = hparams["dropout"]
max_tolerance, max_epochs = 30, 100
early_stopping_target = "pr_auc_on_one"
batch_size = hparams["batch_size"]
lr = hparams["lr"]

# fix random seeds
seed = 9
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# for logging
global_validation_score_on_each_epoch = list()
valid_log_key = "pr_auc_on_one"

def get_df(datapath):
    return pd.DataFrame(pickle.load(open(datapath, "rb")))

def get_df_from_path(p_list):
    return pd.concat([get_df(d) for d in p_list]).reset_index(drop=True)

from dataset_selector import dataset_select
df_all,  dataset_train, dataset_valid, dataset_test, n_tok, n_pos1, n_pos2, n_seg = \
    dataset_select(args.dataset, args.spbtarget, args.kfold)

if 'test' in args.dataset:
    max_tolerance, max_epochs = 2, 2

kwargs = {"num_workers": 8, "pin_memory": True} if device == "cuda" else {}

for name, dset in zip(['tra','val','tes'], [dataset_train, dataset_valid, dataset_test]):
    print('len(dset)', len(dset), name)
    
train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train, batch_size=batch_size, shuffle=True, **kwargs
)
valid_loader = torch.utils.data.DataLoader(
    dataset=dataset_valid, batch_size=batch_size, shuffle=False, **kwargs
)
test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test, batch_size=batch_size, shuffle=False, **kwargs
)

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
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 15.0], device=device))


trainer = create_supervised_trainer(
    model=model,
    optimizer=optim,
    loss_fn=loss_fn,
    device=device,
)


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
        if withcomet:
            experiment.log_metric(f"train:{k}", v, epoch=trainer.state.epoch)
    print(text)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(valid_loader)
    metrics = evaluator.state.metrics
    text = f"Validation Results - Epoch: {trainer.state.epoch}. "
    global_validation_score_on_each_epoch.append(metrics[valid_log_key])
    for k, v in metrics.items():
        text += f"Avg {k}: {v:.3f} "
        if withcomet:
            experiment.log_metric(f"valid:{k}", v, epoch=trainer.state.epoch)
    print(text)


@trainer.on(Events.EPOCH_COMPLETED)
def log_test_results(trainer):
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    text = f"Test Results - Epoch: {trainer.state.epoch}. "
    for k, v in metrics.items():
        text += f"Avg {k}: {v:.4f} "
        if withcomet:
            experiment.log_metric(f"test:{k}", v, epoch=trainer.state.epoch)
    print(text)


evaluator.add_event_handler(
    Events.COMPLETED,
    EarlyStopping(
        patience=max_tolerance,
        score_function=lambda engine: engine.state.metrics[early_stopping_target],
        trainer=trainer,
    ),
)

CHECK_POINT_DIR = (
    f"{pathlib.Path(__file__).parent.absolute()}/../../checkpoint/{yymmddhhmmss}"
)
os.system(f"mkdir -p {CHECK_POINT_DIR}")
print("checkpoint:", CHECK_POINT_DIR, "will be written as an ignite checkpoint")
to_save = {"model": model, "optimizer": optim, "trainer": trainer}
handler = Checkpoint(
    to_save,
    DiskSaver(CHECK_POINT_DIR, create_dir=True),
    n_saved=2,
    filename_prefix=f"best_{yymmddhhmmss}",
    score_function=lambda engine: engine.state.metrics[valid_log_key],
    score_name=valid_log_key,
    global_step_transform=global_step_from_engine(trainer),
)
evaluator.add_event_handler(Events.COMPLETED, handler)

####################################################
## Ignite the loop
trainer.run(train_loader, max_epochs=max_epochs)
####################################################

## Get the prediction and the best model
to_load = {"model": model, "optimizer": optim, "trainer": trainer}
checkpoint_fp = f"{CHECK_POINT_DIR}/{handler.last_checkpoint}"
checkpoint = torch.load(checkpoint_fp)
Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

model = model.to(device)
model.eval()
with torch.no_grad():
    hhlist, yylist = [], []
    for xx, yy in test_loader:
        __xx = [x.to(device) for x in xx]
        __hh = model(__xx)
        hhlist.append(pd.DataFrame(__hh.cpu().detach().numpy()))
        yylist.append(pd.DataFrame(yy.cpu().detach().numpy()))

LOGDIR = f"{pathlib.Path(__file__).parent.absolute()}/../../hhyylog"
os.system(f"mkdir -p {LOGDIR}")
pd.concat(hhlist).to_pickle(f"{LOGDIR}/{yymmddhhmmss}_k{kfold}_test_hh.pickle")
pd.concat(yylist).to_pickle(f"{LOGDIR}/{yymmddhhmmss}_k{kfold}_test_yy.pickle")
dataset_test.data.to_pickle(f"{LOGDIR}/{yymmddhhmmss}_k{kfold}_datasettest.pickle")

print("saved: ", f"{LOGDIR}/{yymmddhhmmss}_k{kfold}_test_hh.pickle")
print("saved: ", f"{LOGDIR}/{yymmddhhmmss}_k{kfold}_test_yy.pickle")

#######################################################
# save Attention
#######################################################

d_model, d_ff, n_head,n_local_encoder,n_global_encoder = hparams['d_model'], hparams['d_ff'], hparams['n_head'], hparams['n_local_encoder'], hparams['n_global_encoder']
dropout=hparams['dropout']

explain_model = Explain_TCRModel(d_model=d_model, d_ff=d_ff, n_head=n_head, n_local_encoder=n_local_encoder, 
                                 n_global_encoder=n_global_encoder, dropout=dropout, scope=4, 
                                 n_tok=n_tok, n_pos1=n_pos1, n_pos2=n_pos2, n_seg=n_seg)

for name, dset in zip(['tra','val','tes'], [dataset_train, dataset_valid, dataset_test]):
    print('len', name, len(dset))
    results_attention = []
    analysis_loader = torch.utils.data.DataLoader(dataset=dset, batch_size=1, shuffle=False)
    for i, (xx,yy) in enumerate(analysis_loader):
        result_tuple = get_attention_weights([x.to(device) for x in xx], 
                                       model, explain_model=explain_model, device=device)
        result = result_tuple + tuple(yy.cpu().numpy())
        results_attention.append(result)
    pickle.dump(results_attention, open(f'{LOGDIR}/{yymmddhhmmss}_results_attention_{name}.pickle', 'wb'))
    print('saved: ', f'{LOGDIR}/{yymmddhhmmss}_results_attention_{name}.pickle')
