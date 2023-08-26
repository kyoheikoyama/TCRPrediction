# tcrpred
tcrpred for the paper (https://www.biorxiv.org/content/10.1101/2023.02.16.528799v3)

## Setup env

1. make init.conda
2. conda activate tcrpred
3. make env.conda

## Commands

The main folder for the scripts is `./scripts/`

- main.py: This script is used for training the model.
- prediction.py: This script is used for making predictions using the trained model. The only output is the score of the prediction.
- explain.py: This script is used for explaining, adding, and acquiring Attention in a trained model for an arbitrary array. The output is the Attention, not just the acquisition of the score of the prediction.
- precompute_dict.py, calc_distances_pdb.py, run_ligplot.py, create_pdb_info.py: These scripts use PDB files as input to determine CDR regions, calculate remaining period distances, and obtain hydrogen bond information using Ligprot. The `create_pdb_info.py` script provides information for each amino acid residue, such as peptide-contacts or other information.



## How to use for adhoc inputs
- Create your own datasets.csv: for instance, ../data/sample_train.csv
- then pass them in the `main_from_csv.py`

`cd scripts && python main_from_csv.py --traincsv ../data/sample_train.csv  --testcsv ../data/sample_test.csv `



## 20230626 for sequence model

#### "../data/20230627_110913_k-1_datasettest.parquet"
python main.py --params best.json --dataset vdjdbno10x --modeltype cross 

#### "../data/20230627_111332_k-1_datasettest.parquet"
python main.py --params best.json --dataset mcpas --modeltype cross


##  20230703 attention on all

#### /media/kyohei/forAI/tcrpred/hhyylog/20230704_221459_k-1_datasettest.parquet
python main.py --params best.json --dataset allwithtest --modeltype cross 

#### /media/kyohei/forAI/tcrpred/hhyylog/20230704_232054_k-1_datasettest.parquet
python main.py --params best.json --dataset allwithtest --modeltype self_on_all


#### /media/kyohei/forAI/tcrpred/hhyylog/20230819_081045_k-1_datasettest.parquet
python main.py --params best.json --dataset all --modeltype self_on_all


#### /media/kyohei/forAI/tcrpred/hhyylog/20230819_070510_k-1_datasettest.parquet
python main.py --params best.json --dataset all --modeltype cross

### /media/kyohei/forAI/tcrpred/hhyylog/20230823_213832_k-1_datasettest.parquet
python main.py --params best.json --dataset all --modeltype cross 

### /media/kyohei/forAI/tcrpred/hhyylog/20230823_202306_k-1_datasettest.parquet
python main.py --params best.json --dataset all --modeltype self_on_all

## experiment for neurips

#### /media/kyohei/forAI/tcrpred/hhyylog/20230825_210148_k-1_datasettest.parquet
python main.py --params best.json --dataset vdjdbno10x --modeltype self_on_all
  - Test Results - Epoch: 26. Avg accuracy: 0.8676 Avg xent: 0.3589 Avg roc_auc: 0.9499 Avg pr_auc_on_one: 0.7539 Avg pr_auc_on_zero: 0.9908

#### /media/kyohei/forAI/tcrpred/hhyylog/20230825_205356_k-1_datasettest.parquet
python main.py --params best.json --dataset mcpas --modeltype self_on_all
  - Test Results - Epoch: 26. Avg accuracy: 0.8638 Avg xent: 0.7556 Avg roc_auc: 0.9234 Avg pr_auc_on_one: 0.6299 Avg pr_auc_on_zero: 0.9857 

## experiment for neurips
#### /media/kyohei/forAI/tcrpred/hhyylog/20230826_031920_k-1_datasettest.parquet
python main.py --params best.json --dataset vdjdbno10x --modeltype cross
  - Test Results - Epoch: 35. Avg accuracy: 0.9057 Avg xent: 0.6189 Avg roc_auc: 0.9459 Avg pr_auc_on_one: 0.7760 Avg pr_auc_on_zero: 0.9890  

#### /media/kyohei/forAI/tcrpred/hhyylog/20230826_032916_k-1_datasettest.parquet
python main.py --params best.json --dataset vdjdbno10x --modeltype self_on_all
  - Test Results - Epoch: 27. Avg accuracy: 0.9077 Avg xent: 0.5786 Avg roc_auc: 0.9513 Avg pr_auc_on_one: 0.7690 Avg pr_auc_on_zero: 0.9906 

## /media/kyohei/forAI/tcrpred/hhyylog/20230826_033421_k-1_datasettest.parquet
python main.py --params best.json --dataset mcpas --modeltype cross
  - Test Results - Epoch: 23. Avg accuracy: 0.8738 Avg xent: 0.6872 Avg roc_auc: 0.9187 Avg pr_auc_on_one: 0.6349 Avg pr_auc_on_zero: 0.9845 

## /media/kyohei/forAI/tcrpred/hhyylog/20230826_033957_k-1_datasettest.parquet
python main.py --params best.json --dataset mcpas --modeltype self_on_all
  - Test Results - Epoch: 31. Avg accuracy: 0.8824 Avg xent: 1.0388 Avg roc_auc: 0.9206 Avg pr_auc_on_one: 0.6123 Avg pr_auc_on_zero: 0.9852 

## /media/kyohei/forAI/tcrpred/hhyylog/20230826_035401_k-1_datasettest.parquet
python main.py --params best.json --dataset entire --modeltype cross
  - Test Results - Epoch: 47. Avg accuracy: 0.7253 Avg xent: 2.7443 Avg roc_auc: 0.5519 Avg pr_auc_on_one: 0.2052 Avg pr_auc_on_zero: 0.8788


## /media/kyohei/forAI/tcrpred/hhyylog/20230826_044148_k-1_datasettest.parquet
python main.py --params best.json --dataset entire --modeltype self_on_all
  - Test Results - Epoch: 32. Avg accuracy: 0.7141 Avg xent: 2.5865 Avg roc_auc: 0.5357 Avg pr_auc_on_one: 0.1889 Avg pr_auc_on_zero: 0.8698 



## Prediction
- prediction.py
  - `python predict.py --model_key entire_crossatten --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet`

  - python predict.py --model_key entire_self_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet
  - python predict.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet  # 
  - python predict.py --model_key entire_cross_stoppingByAP --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet 

## Explain
  - python explain.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/pdb_complex_sequencesV2.parquet
  - python explain.py --model_key entire_self_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/pdb_complex_sequencesV2.parquet


# How to annotate pdb

- precompute_dict.py
  - python precompute_dict.py --pdbdir ../analysis/zipdata/pdb --sceptre_result_csv ../data/sceptre_result_v2.csv

- calc_distances_pdb.py
  - python calc_distances_pdb.py --cdrpath ../data/20230817_060411__DICT_PDBID_2_CDRS.pickle

- run_ligplot.py 
  - python run_ligplot.py --dict_pdbid_2_cdrs ../data/20230817_060411__DICT_PDBID_2_CDRS.pickle 

- create_pdb_info.py 
  - `python create_pdb_info.py \
    --dict_pdbid_2_chainnames ../data/DICT_PDBID_2_CHAINNAMES.json \
    --dict_pdbid_2_residues ../data/20230817_060411__DICT_PDBID_2_RESIDUES.pickle \
    --dict_pdbid_2_cdrs ../data/20230817_060411__DICT_PDBID_2_CDRS.pickle \
    --residue_distances ../data/20230817_060411__residue_distances.parquet`

