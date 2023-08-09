# tcrpred
tcrpred for the paper (https://www.biorxiv.org/content/10.1101/2023.02.16.528799v2)

## Setup env

1. make init.conda
2. conda activate tcrpred
3. make env.conda


## How to use
- Create your own datasets.csv: for instance, ../data/sample_train.csv
- then pass them in the `main_from_csv.py`

`cd scripts && python main_from_csv.py --traincsv ../data/sample_train.csv  --testcsv ../data/sample_test.csv `



## 20230626

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