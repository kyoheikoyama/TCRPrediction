from torch import nn
import torch
import optuna
from copy import deepcopy
import json, time, random

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers.early_stopping import EarlyStopping
from ignite.metrics.metric import Metric
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine

from recipes.metrics import ROC_AUC, PR_AUC
from recipes.dataset import TCRDataset
from recipes.model import TCRModel
import pandas as pd
import numpy as np
import os
import datetime
import pathlib

device = "cuda" if torch.cuda.is_available() else "cpu"

# fix random seeds
seed = 9
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def main(args, params):
    yymmddhhmmss = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    max_epochs = args.maxepochs
    early_stopping_target = "pr_auc_on_one"
    valid_log_key = 'pr_auc_on_one'
    
    # for logging
    global_validation_score_on_each_epoch = list()    

    # Data
    if args.dev:
        max_epochs = 2
        dataset_trainvalid = TCRDataset(datadir='./data', donors=['Donor1', 'Donor2', 'Donor3'], size='small')
        dataset_test = TCRDataset(datadir='./data', donors=['Donor4'], size='small')
    elif args.datasize4hpo:
        dataset_trainvalid = TCRDataset(datadir='./data', donors=['Donor1', 'Donor2', 'Donor3'], size='datasize4hpo')
        dataset_test = TCRDataset(datadir='./data', donors=['Donor4'], size='datasize4hpo')
    else:
        dataset_trainvalid = TCRDataset(datadir='./data', donors=['Donor1', 'Donor2', 'Donor3'])
        dataset_test =TCRDataset(datadir='./data', donors=['Donor4'])


    dataset_train, dataset_valid = torch.utils.data.random_split(
        dataset_trainvalid,
        [int(len(dataset_trainvalid) * 0.8), len(dataset_trainvalid) - int(len(dataset_trainvalid) * 0.8)]
        # generator=torch.Generator().manual_seed(42),
    )

    batch_size = params['batch_size']
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test, batch_size=batch_size, shuffle=False)

    model = TCRModel(d_model=params['d_model'], d_ff=params['d_ff'], n_head=params['n_head'], n_local_encoder=params['n_local_encoder'], 
                     n_global_encoder=params['n_global_encoder'], dropout=params['dropout'], scope=4)

    # Optimizer
    optim = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # Loss
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 6.0], device=device))

    # for debug
    # for xx, yy in test_loader: break
    # ypred = model(xx)
    # loss = loss_fn(ypred, yy)

    # TODO: function to send data into GPU for ignite engine
    # def prepare_batch(batch, device=None, non_blocking=False):
    #     return send2dev(batch)

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


    @trainer.on(Events.COMPLETED)
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
            patience=args.maxtolerance,
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
    print('final max(global_validation_score_on_each_epoch)', max(global_validation_score_on_each_epoch))
    return max(global_validation_score_on_each_epoch)


def suggest(trial, name, target, x):
    fs = {
        "cat": trial.suggest_categorical,
        "int": trial.suggest_int,
        "cun": trial.suggest_uniform,
        "lun": trial.suggest_loguniform,
        "dun": trial.suggest_discrete_uniform,
    }
    if type(x) is list and target=='cat':
        return fs[target](name, x)
    elif type(x) is list and target!='cat':
        return fs[target](name, *x)
    else:
        return fs[target](name, *x)


def suggest_params(trial, args):
    dict_params = json.load(open(f"./hpo_params/{args.hpoparams}"))
    params = {}
    for k in dict_params.keys():
        params[k] = suggest(
            trial,
            k,
            list(dict_params[k].keys())[0],
            list(dict_params[k].values())[0],
        )
    return params


def objective_fn(trial, job_name, args):
    params = suggest_params(trial, args)  # TODO
    import numpy as np
    import torch
    try:
        return main(args, params)
    except:
        torch.cuda.empty_cache()
        return np.nan

if __name__ == "__main__":
    import argparse
    import functools
    yymmddhhmmss_for_optuna = datetime.datetime.today().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev', action='store_true')
    parser.add_argument('--datasize4hpo', action='store_true')
    parser.add_argument('--hpoparams')
    parser.add_argument('--njobs', type=int, default=1)
    parser.add_argument('--ntrials', type=int, default=60)
    parser.add_argument('--maxepochs', type=int, default=50)
    parser.add_argument('--maxtolerance', type=int, default=20)
    parser.add_argument('--onetimejob', action='store_true')
    args = parser.parse_args()
    
    if args.onetimejob:
        params = json.load(open(f"./hpo_params/{args.hpoparams}"))
        main(args, params)
    
    job_name = f'tcrpred-{yymmddhhmmss_for_optuna}'
    objective = functools.partial(
        objective_fn,
        job_name=job_name,
        args=args,
    )
    study = optuna.create_study(
        direction='maximize',
        study_name='tcrpred',
        storage=f"sqlite:///optuna-{job_name}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.ntrials, n_jobs=args.njobs)
