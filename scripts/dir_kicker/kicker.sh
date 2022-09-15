#!/bin/bash
#PBS -q SQUID
#PBS --group=K2107
#PBS -l elapstim_req=6:00:00
#PBS -l cpunum_job=8
#PBS -l gpunum_job=1
export SINGULARITY_BIND="/sqfs/work/,/sqfs2/cmc/0/work/"

kfold=$kfold
echo $PBS_O_WORKDIR
cd $PBS_O_WORKDIR
echo $kfold
echo $spbtarget
echo $dataset
source /sqfs/work/K2107/u6b233/tcrpred_gpucontainer/bin/activate
#singularity run  --home ~/jupyter_notebook/user_work/tcrpred/scripts/ --nv /sqfs/work/K2107/u6b233/nvidia_sandbox.sif /usr/bin/python main.py --dev --params optuna_best.json --dataset mcpas 

if [ -z "${PARAMS+xxx}" ]; 
	then echo "PARAMS is not set at all. So it is now optuna_best.json "
	PARAMS=optuna_best.json;
	else
		echo "PARAMS is not set"
		echo $PARAMS
	fi
echo $PARAMS

singularity run  \
	--bind /sqfs/work/K2107/u6b233:/root/jupyter_notebook/user_work \
        --bind /sqfs/work/K2107/u6b233/checkpoint:/root/jupyter_notebook/user_work/checkpoint \
        --bind ~/jupyter_notebook/user_work/tcrpred:/root/jupyter_notebook/user_work/tcrpred \
        --home ~/jupyter_notebook/user_work/:/root/jupyter_notebook/user_work/ \
        --pwd /root/jupyter_notebook/user_work/tcrpred/ \
        --workdir /root/jupyter_notebook/user_work/ \
	--nv /sqfs/work/K2107/u6b233/nvidia_sandbox.sif /usr/bin/python scripts/main.py --params $PARAMS --dataset $dataset --kfold $kfold --spbtarget $spbtarget
