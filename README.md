# nar-cardest
Cardinality estimation by non-autoregressive (masked) model.


## Prerequisites
All experiments can be run in a docker container.

* Docker
* GPU/cuda environment (for Training)


## Getting Started
### Setup
Dependencies are automatically installed while building a docker image.

```bash
# on host
git clone https://github.com/OnizukaLab/nar-cardest.git
cd nar-cardest
docker build -t cardest .
docker run --rm --gpus all -v `pwd`:/workspaces/nar-cardest -it cardest bash

# in container
poetry shell

# in poetry env in container
./scripts/dowload_dmv.sh
./scripts/dowload_imdb.sh
```

<details>
<summary>(Additional info) Alternative to join-sampled IMDb csv files</summary>

In order to run IMDb-related experiments, need join-sampled csv files.
Use files from `./scripts/download_imdb.sh` or the following extra process.

1. Run `job-light` (re-training) of [neurocard](https://github.com/neurocard/neurocard), get a 10M-join-sampled csv file, and put it as a file `datasets/imdb-job-light.csv`.
    * To get 10M samples, `bs` and `max_steps` should be `2048` and `5120`, respectively.
    * To get a join-sampled file, `_save_samples` option may be required.
    * Actual re-training is not necessary. Just a join-sampled file is needed.
2. In the same way as above, get a 10M-join-sampled csv file of full IMDb dataset and put it as a file `datasets/imdb.csv`.
    * In addition, need to adjust target tables and columns in [neurocard/experiments.py](https://github.com/neurocard/neurocard/blob/master/neurocard/experiments.py) to match [nar-cardest/cardest/datasets.py#Loader.load_imdb](https://github.com/OnizukaLab/nar-cardest/blob/master/cardest/datasets.py)
3. Finally, transform some columns' name in `datasets/{imdb.csv,imdb-job-light.csv}`.
    * Replace `(.+):(.+)` with `\1.\2`
    * Replace `__in_(.+)` with `__in__:\1`
    * Replace `__fanout_(.+)` with `__fanout__:\1`
</details>


### Examples
#### Training
Choose hyperparameter search by optuna or manually specified parameters.
```bash
# train w/ hyperparameter search
python cardest/run.py --train -d=imdb -t=mlp --n-trials=10 -e=20

# train w/o hyperparameter search
python cardest/run.py --train -d=imdb -t=mlp -e=20 --d-word=64 --d-ff=256 --n-ff=4 --lr=5e-4
```


#### Evaluation
```bash
# evaluation
python cardest/run.py --eval -d=imdb -b=job-light -t=mlp -m=models/mlp-ur-imdb/nar-mlp-imdb-imdb-universal.pt
```
You can find results in `results/<benchmark_name>` after trial.


### Options
#### Common Options
* `-d/--dataset`: Dataset name
* `-t/--model-type`: Internal model type (`mlp` for MLP or `trm` for Transformer)
* `--seed`: Random seed (Default: `1234`)
* `--n-blocks`: The number of blocks (for Transformer)
* `--n-heads`: The number of heads (for Transformer)
* `--d-word`: Embedding dimension
* `--d-ff`: Width of feedforward networks
* `--n-ff`: The number of feedforward networks (for MLP)
* `--fact-threshold`: [Column factorization](https://speakerdeck.com/zongheng/neurocard-one-cardinality-estimator-for-all-tables?slide=27) threshold (Default: `2000`)
* `--fact-bits`: [Column factorization](https://speakerdeck.com/zongheng/neurocard-one-cardinality-estimator-for-all-tables?slide=27) bit (Default: `10`)

#### Options for Training
* `-e/--epochs`: Training epoch
* `--batch-size`: Batch size (Default: `1024`)

(w/ hyperparameter search)
* `--n-trials`: The number of trials for hyperparameter search

(w/ specified parameters)
* `--lr`: Learning rate
* `--warmups`: Warm-up epoch (for Transformer) (`lr` and `warmups` are exclusive)

#### Options for Evaluation
* `-m/--model`: Path to model
* `-b/--benchmark`: Benchmark name
* `--eval-sample-size`: Sample size for evaluation

#### Choices
* Datasets
    * DMV
        * `dmv`: All data of DMV
    * IMDb
        * `imdb`: (almost) All data of IMDb
        * `imdb-job-light`: Subset of IMDb for JOB-light benchmark
* Benchmarks
    * DMV
        * `synthetic2000-1234`: Synthetic 2000 queries (random seed = 1234)
    * IMDb
        * `job-light`: Real-world 70 queries
* Models
    * `mlp`: Masked MLP-based non-autoregressive model
    * `trm`: Masked Transformer-based non-autoregressive model


## Pretrained models
* `mlp-ur-dmv`: MLP-based model for DMV dataset (Set `--fact-threshold=3000`)
* `trm-ur-dmv`: Transformer-based model for DMV dataset (Set `--fact-threshold=3000`)
* `mlp-ur-imdb`: MLP-based model for IMDb dataset
* `trm-ur-imdb`: Transformer-based model for IMDb dataset
* `mlp-ur-imdb-jl`: MLP-based model for IMDb dataset (subset for only JOB-light benchmark)
* `trm-ur-imdb-jl`: Transformer-based model for IMDb dataset (subset for only JOB-light benchmark)


## Reference
```bib
@InProceedings{nar_cardest,
    author = {Ito, Ryuichi and Xiao, Chuan and Onizuka, Makoto},
    title = {{Robust Cardinality Estimator by Non-autoregressive Model}},
    booktitle = {Software Foundations for Data Interoperability},
    year = {2022},
    pages = {55--61},
    isbn = {978-3-030-93849-9}
}
```


## Acknowledgement
Some source codes are based on [naru](https://github.com/naru-project/naru)/[neurocard](https://github.com/neurocard/neurocard)
