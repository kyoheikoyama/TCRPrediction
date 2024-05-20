# tcrpred
TCR-pMHC (CDR3ab-pMHC) prediction programme for the paper (https://www.biorxiv.org/content/10.1101/2023.02.16.528799v3)

# How to setup or install environment

## Base
1. conda create -n tcrpred python=3.8.3 -y
2. conda activate tcrpred
3. (Then, you are in the conda env. Now run the following command in the conda env. Change torch versions according to your own env)

## CPU version
```
conda install -y pytorch==1.7.0 cpuonly -c pytorch
python -m pip install pip==20.2 && pip install -r ./requirements.txt --use-feature=2020-resolver
conda install ignite -c pytorch
conda install mkl==2024.0
```

## GPU version
```
conda install -y pytorch==1.7.0 cudatoolkit=10.2 -c pytorch
python -m pip install pip==20.2 && pip install -r ./requirements.txt --use-feature=2020-resolver
conda install ignite -c pytorch
```

## Error handling
If you face the error below, run `conda install mkl==2024.0`.

```
>>> import torch
Traceback (most recent call last):
 File "<stdin>", line 1, in <module>
 File "/home/XXXXX/miniconda3/envs/tcrpred/lib/python3.8/site-packages/torch/__init__.py", line 190, in <module>
  from torch._C import *
ImportError: /home/XXXXX/miniconda3/envs/tcrpred/lib/python3.8/site-packages/torch/lib/libtorch_cpu.so: undefined symbol: iJIT_IsProfilingActive
```


# How to predict (Use the best model)

## Example command
If you want to run the prediction program, in the scripts directory, run `predict.py` . 
The command is as follows:
`python predict.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet`

This will use ./data/recent_data_test.parquet for the prediction.

## About the schema for new data
The data content has the following 4 Columns, the model needs peptide tcra tcrb for the prediction, and sign for the score calculation.

`peptide tcra tcrb sign`

If you want to enter new data and make a prediction, just enter the above four pairs and the program should work.
If the sign is unknown, just set all cases to 0, 1, or np.nan and it should work without error.
Also, please note the maximum lengths of the CDR and Peptide array are fixed as follows:
`MAXLENGTH_A, MAXLENGTH_B, max_len_epitope = 28, 28, 25`

## How to test the code for adhoc inputs
- Create your own datasets.csv: for instance, ../data/sample_adhoc.csv
`python predict.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/sample_adhoc.csv`

- The output file will be saved at `../data/sample_adhoc_entire_cross_newemb.csv`
  - The output has additional columns of ['pred0','pred1'], which are the softmax result of the network output. 
  - We can look at the pred1 to judge the prediction results for positive or negative. 
  - If the value of pred1 is more than 0.5, the prediction result is positive for the input.



## How to train the code for adhoc inputs
- Create your own datasets.csv: for instance, ../data/sample_train.csv
- then pass them in the `main_from_csv.py`

`cd scripts && python main_from_csv.py --traincsv ../data/sample_train.csv  --testcsv ../data/sample_test.csv `


## Commands

The main folder for the scripts is `./scripts/`

- main.py: This script is used for training the model.
- prediction.py: This script is used for making predictions using the trained model. The only output is the score of the prediction.
- explain.py: This script is used for explaining, adding, and acquiring Attention in a trained model for an arbitrary array. The output is the Attention, not just the acquisition of the score of the prediction.
- precompute_dict.py, calc_distances_pdb.py, run_ligplot.py, create_pdb_info.py: These scripts use PDB files as input to determine CDR regions, calculate remaining period distances, and obtain hydrogen bond information using Ligprot. The `create_pdb_info.py` script provides information for each amino acid residue, such as peptide-contacts or other information.



## Prediction

after making sure you have data to predict (`../data/recent_data_test.parquet`) and checkpoint 

- prediction.py
  - `python predict.py --model_key entire_crossatten --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet`

  - python predict.py --model_key entire_self_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet
  - python predict.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet  # 
  - python predict.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/covid19_9mer.parquet
  - python predict.py --model_key benchmark_cross_newemb_vdjdbno10x --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet
  - python predict.py --model_key benchmark_cross_mcpas --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet

  - python predict.py --model_key entire_cross_stoppingByAP --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/recent_data_test.parquet 




## Explain
  - python explain.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/pdb_complex_sequencesV2.parquet
  - python explain.py --model_key entire_self_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/pdb_complex_sequencesV2.parquet
  - python explain.py --model_key entire_cross_newemb --checkpointsjson ../hpo_params/checkpoints.json --input_filepath ../data/mutation_study.parquet

## stats_test
  - python stats_test.py --seqfile ../data/pdb_complex_sequencesV2.parquet \
        --checkpointsjson ../hpo_params/checkpoints.json \
            --input_filepath ../data/pdb_complex_sequencesV2_entire_cross_newemb__explained.parquet --datetimehash 20230828_015709

    python stats_test.py --seqfile ../data/pdb_complex_sequencesV2.parquet \
        --checkpointsjson ../hpo_params/checkpoints.json \
            --input_filepath ../data/pdb_complex_sequencesV2_entire_self_newemb__explained.parquet --datetimehash 20230828_015709
            
  - python mutation_analysis.py --bondinfo ../data/20230828_015709__df_bondinfo.parquet --explained ../data/mutation_study_entire_cross_newemb__explained.parquet

  - python mutation_analysis.py --bondinfo ../data/20230828_015709__df_bondinfo.parquet --explained ../data/mutation_study_entire_self_newemb__explained.parquet


# How to annotate pdb

- precompute_dict.py
  - python precompute_dict.py --pdbdir ../analysis/zipdata/pdb --sceptre_result_csv ../data/sceptre_result_v2.csv

- calc_distances_pdb.py
  - python calc_distances_pdb.py --cdrpath ../data/20230828_015709__DICT_PDBID_2_CDRS.pickle

- run_ligplot.py 
  - python run_ligplot.py --dict_pdbid_2_cdrs ../data/20230828_015709__DICT_PDBID_2_CDRS.pickle 

- create_pdb_info.py 
  - `python create_pdb_info.py \
    --dict_pdbid_2_chainnames ../data/DICT_PDBID_2_CHAINNAMES.json \
    --dict_pdbid_2_residues ../data/20230828_015709__DICT_PDBID_2_RESIDUES.pickle \
    --dict_pdbid_2_cdrs ../data/20230828_015709__DICT_PDBID_2_CDRS.pickle \
    --residue_distances ../data/20230828_015709__residue_distances.parquet`

