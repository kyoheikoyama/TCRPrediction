# tcrpred
tcrpred

## Setup env

1. make init.conda
2. bash 
3. conda activate tcrpred
4. source ~/SageMaker/bash_ssh.sh && source ~/SageMaker/alias.sh && make env.conda



## command history

#### debug

python main.py --dev --params optuna_best.json

python main.py --dev --params optuna_best.json --dataset mcpas

python main.py 

#### hpo

python main_hpo.py --dev --hpoparams hpo.json

python main_hpo.py --datasize4hpo --hpoparams hpo.json
   
python main_hpo.py --onetimejob --hpoparams optuna_best.json

python main.py --params best_20210218_062936_json.json


###### 20210317
python main_hpo.py --datasize4hpo --hpoparams hpo.json

python main_noKLGGALQAK.py --params best_20210218_062936_json.json


###### 20210803 
python main.py --params optuna_best.json --dataset mcpas