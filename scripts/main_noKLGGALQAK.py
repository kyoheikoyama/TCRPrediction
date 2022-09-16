from torch import nn
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers.early_stopping import EarlyStopping
from ignite.metrics.metric import Metric
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from recipes.metrics import ROC_AUC, PR_AUC
from copy import deepcopy
from recipes.dataset import TCRDataset
from recipes.model import TCRModel
import pandas as pd
import numpy as np
import os, json
import datetime
import pathlib


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dev', action='store_true')
parser.add_argument('--params')
args = parser.parse_args()

yymmddhhmmss = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
device = torch.device("cuda:1") if torch.cuda.is_available() else "cpu"


# parameters
with open(f'./hpo_params/{args.params}', 'r') as fp:
    hparams = json.load(fp)

d_model=hparams['d_model']
d_ff=hparams['d_ff']
n_head=hparams['n_head']
n_local_encoder=hparams['n_local_encoder']
n_global_encoder=hparams['n_global_encoder']
dropout=hparams['dropout']
max_tolerance = 20
max_epochs = 20
early_stopping_target = "pr_auc_on_one"
batch_size = hparams['batch_size']
lr=hparams['lr']

# fix random seeds
seed = 9
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# for logging
global_validation_score_on_each_epoch = list()
valid_log_key = 'pr_auc_on_one'

# Data
if args.dev:
    max_epochs = 2
    dataset_trainvalid = TCRDataset(datapath='./data/03.VDJdb_noKLGGALQAK.tsv', donors=['Donor1', 'Donor2', 'Donor3'], size='small')
    dataset_test = TCRDataset(datapath='./data/03.VDJdb_noKLGGALQAK.tsv', donors=['Donor4'], size='small')
else:
    dataset_trainvalid = TCRDataset(datapath='./data/03.VDJdb_noKLGGALQAK.tsv', donors=['Donor1', 'Donor2', 'Donor3'])
    dataset_test = TCRDataset(datapath='./data/03.VDJdb_noKLGGALQAK.tsv', donors=['Donor4'])


dataset_train, dataset_valid = torch.utils.data.random_split(
    dataset_trainvalid,
    [int(len(dataset_trainvalid) * 0.8), len(dataset_trainvalid) - int(len(dataset_trainvalid) * 0.8)]
    # generator=torch.Generator().manual_seed(42),
)


train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    dataset=dataset_valid, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test, batch_size=batch_size, shuffle=False)


model = TCRModel(d_model=d_model, d_ff=d_ff, n_head=n_head, n_local_encoder=n_local_encoder, 
                 n_global_encoder=n_global_encoder, dropout=dropout, scope=4)

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=lr)

# Loss
loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 15.0], device=device))


trainer = create_supervised_trainer(
    model=model, optimizer=optim, loss_fn=loss_fn, device=device,
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
    print("---Epoch[{}] Loss: {:.4f}".format(trainer.state.epoch, trainer.state.output))


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(trainer):
    evaluator.run(train_loader)
    metrics = evaluator.state.metrics
    text = f'Train Results - Epoch: {trainer.state.epoch}. '
    for k, v in metrics.items():
        text += f'Avg {k}: {v:.3f} '
    print(text)


@trainer.on(Events.EPOCH_COMPLETED)
def log_validation_results(trainer):
    evaluator.run(valid_loader)
    metrics = evaluator.state.metrics
    text = f'Validation Results - Epoch: {trainer.state.epoch}. '
    global_validation_score_on_each_epoch.append(metrics[valid_log_key])
    for k, v in metrics.items():
        text += f'Avg {k}: {v:.3f} '
    print(text)


@trainer.on(Events.EPOCH_COMPLETED)
def log_test_results(trainer):
    evaluator.run(test_loader)
    metrics = evaluator.state.metrics
    text = f'Test Results - Epoch: {trainer.state.epoch}. '
    for k, v in metrics.items():
        text += f'Avg {k}: {v:.4f} '
    print(text)


evaluator.add_event_handler(
    Events.COMPLETED,
    EarlyStopping(
        patience=max_tolerance,
        score_function=lambda engine: engine.state.metrics[early_stopping_target],
        trainer=trainer,
    ),
)

CHECK_POINT_DIR = f'{pathlib.Path(__file__).parent.absolute()}/checkpoint/{yymmddhhmmss}'
to_save = {'model': model, 'optimizer': optim, 'trainer': trainer}
handler = Checkpoint(to_save, 
                     DiskSaver(CHECK_POINT_DIR, create_dir=True), 
                     n_saved=2,
                     filename_prefix=f'best_{yymmddhhmmss}', 
                     score_function=lambda engine: engine.state.metrics[valid_log_key], 
                     score_name=valid_log_key,
                     global_step_transform=global_step_from_engine(trainer))

evaluator.add_event_handler(Events.COMPLETED, handler)

## Ignite the loop
trainer.run(train_loader, max_epochs=max_epochs)

## Get the prediction and the best model
to_load = {'model': model, 'optimizer': optim, 'trainer': trainer}
checkpoint_fp = f"{CHECK_POINT_DIR}/{handler.last_checkpoint}"
checkpoint = torch.load(checkpoint_fp)
Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)

## Resume engine’s run from a state. User can load a state_dict and run engine starting from the state
# print('best score: ', max(global_validation_score_on_each_epoch))
# best_epoch = 1 + global_validation_score_on_each_epoch.index(max(global_validation_score_on_each_epoch))
# state_dict = {"seed": seed, "epoch": best_epoch, "max_epochs": max_epochs*2, "epoch_length": len(train_loader)}
# trainer.load_state_dict(state_dict)

model = model.to(device)
model.eval()
with torch.no_grad():
    hhlist, yylist = [], []
    for xx, yy in test_loader:
        __xx = [x.to(device) for x in xx]
        __hh = model(__xx)
        hhlist.append(pd.DataFrame(__hh.cpu().detach().numpy()))
        yylist.append(pd.DataFrame(yy.cpu().detach().numpy()))

pd.concat(hhlist).to_csv(f'./log/{yymmddhhmmss}_hh.csv', index=None)
pd.concat(yylist).to_csv(f'./log/{yymmddhhmmss}_yy.csv', index=None)
dataset_test.data.to_csv('dataset_test.csv', index=None)