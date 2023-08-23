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

#### ../../hhyylog/20230703_185102_k-1_datasettest.parquet
python main.py --params best.json --dataset vdjdbno10x --modeltype self_on_all

#### ../../hhyylog/20230703_215045_k-1_datasettest.parquet
python main.py --params best.json --dataset mcpas --modeltype self_on_all

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

## Prediction
- prediction.py
  - `python predict.py --model_key entire_crossatten --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet`

  - python predict.py --model_key entire_self_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet
  - python predict.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet  # 
  - python predict.py --model_key entire_cross_stoppingByAP --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet 


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

