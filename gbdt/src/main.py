from preprocess import preprocess
from solver_lightgbm import Solver

import mlflow
import hydra
import optuna.integration.lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import mlflow

import os
import random
import json
from pathlib import Path


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


@hydra.main('../config/config.yaml')
def main(cfg):
    # kwargs = {"g_n_comp": 50, "c_n_comp":15, "threshold":0.5}
    train, test= preprocess(
            "/kaggle/input/lish-moa/train_features.csv",
            "/kaggle/input/lish-moa/test_features.csv",
            cfg.preprocess.function_name,
            cfg.preprocess.kwargs)

    train_targets = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
    sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

    train_targets = train_targets.drop("sig_id", axis=1)

    solver = Solver(features=train, targets=train_targets, test_features=test, sub=sub, **cfg.train)
    solver.train_pred()

    with open("res.json", "w") as f:
        json.dump(res_dict, f)

    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.ex_name)

    mlflow.log_artifact(Path.cwd() / "res.json")
    mlflow.log_params(cfg.preprocess)
    mlflow.log_params(cfg.train.params)
    mlflow.log_metric("CV score", score)


if __name__ == "__main__":
    main()
