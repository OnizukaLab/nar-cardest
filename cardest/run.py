import os
import glob
import json
import random
import shutil
import logging
import argparse
from datetime import datetime
from typing import Any, List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import torch
import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor
import ray
from ray import tune
from ray.tune.logger import LoggerCallback
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.wandb import WandbLoggerCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import sqlparse as sp
from wandb import wandb

import cardest.estimators
import cardest.datasets
import cardest.common
import cardest.parse
import cardest.models


def calc_entropy(name, data, bases=None):
    import scipy.stats

    s = "Entropy of {}:".format(name)
    ret = []
    for base in bases:
        assert base == 2 or base == "e" or base is None
        e = scipy.stats.entropy(data, base=base if base != "e" else None)
        ret.append(e)
        unit = "nats" if (base == "e" or base is None) else "bits"
        s += " {:.4f} {}".format(e, unit)
    print(s)
    return ret


def calc_q_err(est_card, true_card):
    if true_card == 0 and est_card != 0:
        return est_card
    if true_card != 0 and est_card == 0:
        return true_card
    if true_card == 0 and est_card == 0:
        return 1.0
    return max(est_card / true_card, true_card / est_card)


def query(
    est,
    query,
    true_card,
    i=None,
):
    cols, ops, vals, tbls = query

    conds = [f""""{c}" {o} '{str(v)}'""" for c, o, v in zip(cols, ops, vals)]
    pseudo_sql = (
        f"query {i}: select count(1) from {','.join(tbls)} where {' and '.join(conds)}"
    )
    print(f"\n{pseudo_sql}")

    est_card, elapsed_time_ms = est.query(query, i)
    err = calc_q_err(est_card, true_card)
    est.add_err(err, est_card, true_card)
    print(f"--  actual {true_card} {str(est)} {est_card} (err={err:.3f})")

    wandb.log(
        {
            "query_no": i,
            "q_err": err,
            "true_card": true_card,
            "est_card": est_card,
            "elapsed_time_ms": elapsed_time_ms,
            "sql": "(TODO)",
            "pseudo sql": pseudo_sql,
        }
    )  # in term of wandb, `i` is used for `step` as well
    return est_card


default_model = {}


def main():
    logging.basicConfig(level="INFO")

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--cache-data", action="store_true")

    parser.add_argument("--params-dir", type=str)
    parser.add_argument(
        "--model-type", "-t", type=str, choices=["trm", "mlp"], required=True
    )
    parser.add_argument("--dataset", "-d", type=str, required=True)

    def nullable_int(val: str):
        if val.isdigit():
            return int(val)
        return None

    parser.add_argument("--n-trials", type=nullable_int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmups", type=Optional[int])
    parser.add_argument("--n-blocks", type=int)
    parser.add_argument("--n-heads", type=int)
    parser.add_argument("--d-word", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--n-ff", type=int)  # for mlp
    parser.add_argument("--without-pos-emb", action="store_true")
    parser.add_argument("--fact-threshold", type=int, default=2000)
    parser.add_argument("--fact-bits", type=int, default=10)

    parser.add_argument("--epochs", "-e", type=int)
    parser.add_argument("--batch-size", type=int, default=1024)

    parser.add_argument("--model", "-m", type=str)
    parser.add_argument("--benchmark", "-b", type=str)
    parser.add_argument("--eval-sample-size", type=int, default=1000)

    parser.add_argument("--cont-fanout", action="store_true")

    parser.add_argument("--wandb-project", type=str)
    parser.add_argument("--wandb-entity", type=str)

    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    local_mode: bool = args.local
    do_train: bool = args.train
    do_eval: bool = args.eval
    cache_data: bool = args.cache_data

    device: str = args.device
    assert device != "cuda" or torch.cuda.is_available()

    params_dir: str = args.params_dir
    model_type: str = args.model_type
    relation_type: str = "ur"
    dataset_name: str = args.dataset
    n_trials: Optional[int] = args.n_trials  # if not None, do hyperparam search
    warmups: Optional[int] = args.warmups

    assert not (
        params_dir is not None and n_trials is not None
    ), "params files can be used as config base only if not use hyperparam search"

    # to specify searchable params
    # will be used w/ `n_traial is None`
    d_word: Optional[int] = args.d_word
    n_blocks: Optional[int] = args.n_blocks
    d_ff: Optional[int] = args.d_ff
    n_ff: Optional[int] = args.n_ff
    n_heads: Optional[int] = args.n_heads
    batch_size: Optional[int] = args.batch_size
    lr: Optional[float] = args.lr

    # non-searchable params
    n_epochs: int = args.epochs
    fact_threshold: int = args.fact_threshold
    fact_bits: int = args.fact_bits
    cont_fanout: bool = args.cont_fanout

    model_path: str = args.model
    if model_path is None and do_eval:
        model_path = default_model[dataset_name][model_type][relation_type]
    benchmark_name: str = args.benchmark
    eval_sample_size: int = args.eval_sample_size

    wandb_project: Optional[str] = args.wandb_project
    wandb_entity: Optional[str] = args.wandb_entity

    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    ray.init(local_mode=local_mode)

    # initialize
    loader = cardest.datasets.Loader(fact_threshold, fact_bits, device)
    db = loader.load(dataset_name, relation_type)

    table_names = [f"{db.name}-universal"]
    project_dir = os.getcwd()
    cache_dir_root = os.path.join(project_dir, ".cache")
    timestamp = datetime.now().strftime("%Y%m%d%H%M")

    # set params
    if do_train and n_trials:  # train w/ hyperparam search
        params_dict = {}
        for table_name in table_names:
            params_dict[table_name] = {
                "static": {
                    "device": device,
                    "dataset_name": dataset_name,
                    "model_type": model_type,
                    "relation_type": relation_type,
                    "cont_fanout": cont_fanout,
                    "table_name": table_name,
                    "n_epochs": n_epochs,  # behavior like max epoch
                    "fact_threshold": fact_threshold,
                    "fact_bits": fact_bits,
                    "id": timestamp,
                    "n_trials": n_trials,
                    "project_dir": project_dir,
                    "cache_dir_root": cache_dir_root,
                    "wandb_project": wandb_project,
                    "wandb_entity": wandb_entity,
                    "seed": random_seed,
                },
                "d_word": tune.choice([32, 64]),
                "batch_size": 1024,
                # "batch_size": 2,  # hack for tiny dataset
                "lr": tune.loguniform(1e-4, 5e-3) if warmups is None else None,
                "warmups": warmups,
                "act_func_name": "gelu",
            }

            table = db.vtable
            input_bins: List[Tuple[str, int]] = (
                [
                    (c.name, (c.dist_size if not c.is_fanout() else 1))
                    for c in table.scols
                ]
                if cont_fanout
                else [(c.name, c.dist_size) for c in table.scols]
            )
            params_dict[table_name]["static"]["input_bins"] = input_bins

            if model_type == "trm":
                params_dict[table_name].update(
                    {
                        "n_blocks": tune.choice([1, 2]),
                        "d_ff": tune.choice([128, 256]),
                        "n_heads": tune.choice([1, 2]),
                    }
                )
            elif model_type == "mlp":
                params_dict[table_name].update(
                    {"d_ff": tune.choice([128, 256]), "n_ff": tune.choice([4, 8])}
                )
    elif do_train and not n_trials:  # train w/ specified hyperparams
        params_dict = {}
        for table_name in table_names:
            if params_dict is not None:
                found = False
                for params_file_path in glob.glob(os.path.join(params_dir, "*.json")):
                    if table_name in params_file_path:
                        with open(params_file_path) as f:
                            params_dict[table_name] = json.load(f)["config"]
                            params_dict[table_name][
                                "base_params_file_path"
                            ] = params_file_path
                            params_dict[table_name]["base_params_id"] = params_dict[
                                table_name
                            ]["static"]["id"]
                            found = True
                            logging.info(
                                f"Found config file for {table_name} ({params_file_path})"
                            )
                            break
                if not found:
                    params_dict[table_name] = {}
            else:
                params_dict[table_name] = {}

            def set_if_not_none(dic: Dict[str, Any], key: str, val: Any, force=False):
                if val is None:  # NOTE: not support to set None even if force is True
                    return
                if (key not in dic or dic[key] is None) or force:
                    if force:
                        logging.info(f"Overriding config {table_name}[{key}] = {val}")
                    dic[key] = val

            params_dict_t = params_dict[table_name]
            set_if_not_none(params_dict_t, "d_word", d_word)
            set_if_not_none(params_dict_t, "batch_size", batch_size)
            set_if_not_none(params_dict_t, "lr", lr if warmups is None else None)
            set_if_not_none(params_dict_t, "warmups", warmups)
            set_if_not_none(params_dict_t, "act_func_name", "gelu")

            if "static" not in params_dict[table_name]:
                params_dict[table_name]["static"] = {}

            params_static = params_dict[table_name]["static"]
            set_if_not_none(params_static, "device", device, True)
            set_if_not_none(params_static, "dataset_name", dataset_name)
            set_if_not_none(params_static, "model_type", model_type)
            set_if_not_none(params_static, "relation_type", relation_type)
            set_if_not_none(params_static, "cont_fanout", cont_fanout)
            set_if_not_none(params_static, "table_name", table_name)
            set_if_not_none(params_static, "n_epochs", n_epochs, True)
            set_if_not_none(params_static, "fact_threshold", fact_threshold)
            set_if_not_none(params_static, "fact_bits", fact_bits)
            set_if_not_none(params_static, "id", timestamp, True)
            set_if_not_none(params_static, "project_dir", project_dir, True)
            set_if_not_none(params_static, "cache_dir_root", cache_dir_root, True)
            set_if_not_none(params_static, "wandb_project", wandb_project, True)
            set_if_not_none(params_static, "wandb_entity", wandb_entity, True)
            set_if_not_none(params_static, "seed", random_seed, True)

            table = db.vtable
            input_bins: List[Tuple[str, int]] = (
                [
                    (c.name, (c.dist_size if not c.is_fanout() else 1))
                    for c in table.scols
                ]
                if cont_fanout
                else [(c.name, c.dist_size) for c in table.scols]
            )
            set_if_not_none(params_static, "input_bins", input_bins)

            if model_type == "trm":
                set_if_not_none(params_dict_t, "n_blocks", n_blocks, True)
                set_if_not_none(params_dict_t, "d_ff", d_ff, True)
                set_if_not_none(params_dict_t, "n_heads", n_heads, True)
            elif model_type == "mlp":
                set_if_not_none(params_dict_t, "d_ff", d_ff, True)
                set_if_not_none(params_dict_t, "n_ff", n_ff, True)

            # always None for avoiding hyperparam search
            params_static["n_trials"] = None
    elif do_eval:
        params_file_path = model_path.replace(".pt", ".json")

        with open(params_file_path) as f:
            verbose_params = json.load(f)
            params = verbose_params["config"]
        table_name = params["static"]["table_name"]

        # update contextual params (different from train phase)
        params["static"]["device"] = device
        params["static"]["wandb_project"] = wandb_project
        params["static"]["wandb_entity"] = wandb_entity

        # add eval-specific params
        params["static"]["model_path"] = model_path
        params["static"]["benchmark_name"] = benchmark_name
        params["static"]["eval_sample_size"] = eval_sample_size

        params_dict = {table_name: params}

    # run
    if do_train:
        dataModule = cardest.common.DataModule(
            db,
            cache_dir_root=list(params_dict.values())[0]["static"]["cache_dir_root"],
            cont_fanout=cont_fanout,
        )
        train(params_dict, dataModule)
    elif do_eval:
        eval_batch(params_dict, db)
    elif cache_data:
        for table_name in table_names:
            cardest.common.DBDataset(
                db, cont_fanout, table_name=table_name, cache_dir_root=cache_dir_root
            )


def create_model(
    params: Dict[str, Union[str, int, Dict[str, Union[str, int]]]]
) -> pl.LightningModule:
    input_bins: List[Tuple[str, int]] = params["static"]["input_bins"]
    model_type: str = params["static"]["model_type"]
    table_name = params["static"]["table_name"]

    if model_type == "trm":
        model = cardest.models.NARTransformer(
            params=params,
            table_name=table_name,
            input_bins=input_bins,
        )
    elif model_type == "mlp":
        model = cardest.models.NARMLP(
            params=params,
            table_name=table_name,
            input_bins=input_bins,
        )
    else:
        raise ValueError(f"Unexpected model type: {model_type}")

    return model.to(params["static"]["device"])


def train_table_trial(
    params: Dict[str, Union[str, int, Dict[str, Union[str, int]]]],
    dataModule: pl.LightningDataModule,
) -> None:
    logger = pl_loggers.WandbLogger(
        name=f"{tune.get_trial_name()}_step",
        project=params["static"]["wandb_project"],
        entity=params["static"]["wandb_entity"],
        reinit=True,
        settings=wandb.Settings(start_method="fork"),
        mode=(
            "online"
            if params["static"]["wandb_project"] is not None
            and params["static"]["wandb_entity"] is not None
            else "offline"
        ),
    )  # for per-step logging
    logger._experiment = wandb.init(
        **logger._wandb_init
    )  # manually init for parallel run

    table_name = params["static"]["table_name"]
    model = create_model(params)
    trainer = pl.Trainer(
        max_epochs=params["static"]["n_epochs"],
        callbacks=[
            TuneReportCheckpointCallback(
                metrics=[
                    "epoch",  # for ASHA scheduler (e.g., 1, 2, ..., n_epochs)
                    f"{table_name}/tra_loss",
                    f"{table_name}/val_loss",
                ],
                filename=f"{model.name}.pt",
                on="validation_end",
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        gpus=1,
        num_sanity_val_steps=0,
        # related to logging
        # if set val_check_interval != 1.0,
        # it confuses ASHA scheduler w/ default time_attr (regard n steps as 1 epoch)
        val_check_interval=0.25,
        log_every_n_steps=1,
        logger=logger,
    )
    dataModule.set_table(params["static"]["table_name"], params["batch_size"])
    trainer.fit(model, dataModule)
    logger._experiment.finish()


def train_table(
    params: Dict[str, Union[str, int, Dict[str, Union[str, int]]]],
    dataModule: pl.LightningDataModule,
    n_trials: Optional[int],  # `None` represents run w/ specified hyperparams
    loggerCallback: LoggerCallback = None,
) -> None:
    table_name = params["static"]["table_name"]
    scheduler = ray.tune.schedulers.ASHAScheduler(
        time_attr="epoch",
        max_t=params["static"]["n_epochs"],
        # If set 1, 1st validation of 1st ep will be recognized as a representative metric of 1st ep
        # As a result, the scheduler may cut off a trial at a too early step
        # On the other hand, if set 2, run full 1ep (+alpha) at least
        grace_period=max(params["static"]["n_epochs"] // 3, 2),
        reduction_factor=2,
    )
    search_alg = OptunaSearch(metric=f"{table_name}/val_loss", mode="min")
    search_alg = ConcurrencyLimiter(search_alg, max_concurrent=1)

    if params["static"]["n_trials"] is None:
        scheduler = None
        search_alg = None

    analysis = tune.run(
        tune.with_parameters(train_table_trial, dataModule=dataModule),
        resources_per_trial=tune.PlacementGroupFactory(
            [{"CPU": 6, "GPU": 0.5}, {"CPU": 6, "GPU": 0.5}]
        ),
        local_dir=os.path.join(params["static"]["project_dir"], "runs"),
        num_samples=n_trials if n_trials is not None else 1,
        search_alg=search_alg,
        metric=f"{table_name}/val_loss",
        mode="min",
        config=params,
        scheduler=scheduler,
        keep_checkpoints_num=5,
        checkpoint_score_attr=f"min-{table_name}/val_loss",
        reuse_actors=True,  # NOTE: no change w/ functional trainable
        callbacks=[loggerCallback],
    )

    print(f"best parameters: {analysis.best_config}")
    model_dir = os.path.join(
        params["static"]["project_dir"], "models", params["static"]["id"]
    )
    os.makedirs(model_dir, exist_ok=True)
    best_model_path = glob.glob(f"{analysis.best_checkpoint}*.pt")[0]
    best_model_name = os.path.splitext(os.path.basename(best_model_path))[0]
    shutil.copy2(best_model_path, model_dir)
    with open(os.path.join(model_dir, f"{best_model_name}.json"), "w") as f:
        json.dump(
            {
                **analysis.best_result,
                "best_logdir": analysis.best_logdir,
                "best_checkpointdir": analysis.best_checkpoint.local_path,
            },
            f,
            indent=2,
        )


def train(params_dict: Dict[str, Union[str, int]], dataModule: pl.LightningDataModule):
    for table_name, params in params_dict.items():
        loggerCallback = WandbLoggerCallback(
            project=params["static"]["wandb_project"],
            entity=params["static"]["wandb_entity"],
            api_key=(
                None
                if params["static"]["wandb_project"] is not None
                and params["static"]["wandb_entity"] is not None
                else "dummy"
            ),
            mode=(
                "online"
                if params["static"]["wandb_project"] is not None
                and params["static"]["wandb_entity"] is not None
                else "offline"
            ),
        )  # NOTE: ray-tune LoggerCallback cannot report logs per-step

        train_table(params, dataModule, params["static"]["n_trials"], loggerCallback)


def eval_batch(
    params_dict: Dict[str, Union[str, int]],
    db: cardest.common.DB,
):
    common_params = list(params_dict.values())[0]["static"]
    dataset_name = common_params["dataset_name"]
    model_type = common_params["model_type"]
    relation_type = common_params["relation_type"]
    benchmark_name = common_params["benchmark_name"]

    wandb.init(
        name=f"{model_type}-{relation_type}/{dataset_name}-{benchmark_name}({common_params['id']})",
        project=common_params["wandb_project"],
        entity=common_params["wandb_entity"],
        config=params_dict,
        mode=(
            "online"
            if common_params["wandb_project"] is not None
            and common_params["wandb_entity"] is not None
            else "offline"
        ),
    )

    assert len(params_dict) == 1
    params = list(params_dict.values())[0]
    model = create_model(params)
    model.load_state_dict(torch.load(params["static"]["model_path"])["state_dict"])
    model.eval()

    master_dataset_name = "imdb" if dataset_name.startswith("imdb") else dataset_name
    benchmark = pa.csv.read_csv(
        f"benchmark/{master_dataset_name}-{benchmark_name}.csv"
    ).to_pandas()
    true_cards = benchmark["true_cardinality"].values
    conds_list = [cardest.parse.parse_to_conds(sql) for sql in benchmark["sql"].values]
    # TODO: refactoring
    queries = []
    for conds in conds_list:
        cols = []
        ops = []
        vals = []
        tbls = set()
        for cond in conds:
            col_name = ""
            for t in cond[1].tokens:
                if t.ttype == sp.tokens.Name:
                    col_name = col_name + t.value
                elif t.ttype == sp.tokens.Literal.String.Symbol:
                    if t.value.startswith('"'):
                        t.value = t.value[1:]
                    if t.value.endswith('"'):
                        t.value = t.value[:-1]
                    col_name = col_name + t.value
                elif t.ttype == sp.tokens.Punctuation:
                    tbls.add(col_name)
                    col_name = col_name + "."
                else:
                    assert False, f"col_name contains invalid tokens: {cond[1]}"
            is_join = cond[3]
            val = ""
            if not is_join:

                def _parse_val(t):
                    if t is None:
                        return None
                    elif isinstance(t, list):
                        return [_parse_val(x) for x in t]
                    elif t.ttype == sp.tokens.Literal.Number.Integer:
                        return int(t.value)
                    elif t.ttype == sp.tokens.Literal.Number.Float:
                        return float(t.value)
                    elif t.ttype == sp.tokens.Literal.String.Single:
                        if t.value.startswith("'"):
                            t.value = t.value[1:]
                        if t.value.endswith("'"):
                            t.value = t.value[:-1]

                        if col_name == "Reg Valid Date":
                            return np.datetime64(t.value)
                        else:
                            return str(t.value)
                    elif t.ttype == sp.tokens.Keyword and t.value.upper() == "NULL":
                        return None
                    else:
                        assert False, f"Not supported type: {t}"

                val = _parse_val(cond[2])
            else:
                for t in cond[2].tokens:
                    if t.ttype == sp.tokens.Name:
                        val = val + t.value
                    elif t.ttype == sp.tokens.Punctuation:
                        tbls.add(val)
                        val = val + "."
                    else:
                        assert False, f"val contains invalid tokens: {cond[2]}"

            if not is_join:
                cols.append(col_name)
                ops.append(cond[0].value.upper())
                vals.append(val)
            else:
                pass
        queries.append(
            (
                np.array(cols),
                np.array(ops, dtype=np.str),
                np.array(vals, dtype=np.object),
                np.array(list(tbls), dtype=np.str),
            )
        )

    assert len(queries) == len(true_cards)
    n_queries = len(queries)

    if not isinstance(model, dict):
        estimator = cardest.estimators.ProgressiveSampling(
            model,
            db,
            params_dict,
        )
    for i in range(n_queries):
        query(
            estimator,
            queries[i],
            true_cards[i],
            i=i,
        )
    result = {
        "estimator": [estimator.name] * n_queries,
        "q_err": estimator.errs,
        "estimated_cardinality": estimator.est_cards,
        "true_cardinality": estimator.true_cards,
        "elapsed_time_ms": estimator.elapsed_time_ms,
    }
    result_dir_path = os.path.join("results", benchmark_name)
    result_file_name = (
        f"{dataset_name}_{model_type}_{relation_type}"
        + f"_{common_params['id']}"
        + f"_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
    )
    result_file_path = os.path.join(result_dir_path, result_file_name)
    result = pd.DataFrame(result)
    xs = result["q_err"].quantile(q=[0.5, 0.9, 0.95, 0.99, 1.0]).tolist()
    xs.append(result["elapsed_time_ms"].mean())
    print("[" + ", ".join([f"{x:.3f}" for x in xs]) + "]")
    os.makedirs(result_dir_path, exist_ok=True)
    result.to_csv(result_file_path, index=False)
    with open(result_file_path.replace(".csv", ".json"), "w") as f:
        json.dump(params_dict, f, indent=2)
    print(result_file_name)


if __name__ == "__main__":
    main()
