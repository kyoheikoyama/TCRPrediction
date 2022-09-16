#!/bin/bash
#PBS -q DBG
#PBS --group=K2107
#PBS -l elapstim_req=0:10:00
#PBS -l cpunum_job=38
#PBS -l gpunum_job=1
export SINGULARITY_BIND="/sqfs/work/,/sqfs2/cmc/0/work/"
echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
echo $kfold
echo $spbtarget
source /sqfs/work/K2107/u6b233/tcrpred_gpucontainer/bin/activate

#singularity run  --home ~/jupyter_notebook/user_work/tcrpred/scripts/ --nv /sqfs/work/K2107/u6b233/nvidia_sandbox.sif /usr/bin/python main.py --dev --params optuna_best.json --dataset mcpas 

singularity run  --home ~/jupyter_notebook/user_work/tcrpred/:/root/jupyter_notebook/user_work/tcrpred/ --nv /sqfs/work/K2107/u6b233/nvidia_sandbox.sif /usr/bin/python scripts/main.py --params optuna_best.json --dataset vdjdbno10x --kfold $kfold --spbtarget $spbtarget
