from preprocess import preprocess
from solver_nn import Solver

import mlflow
import hydra
import omegaconf
import optuna.integration.lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

import os
import random
import json
from pathlib import Path
import warnings

warnings.simplefilter('ignore')


SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def log_params(dic, name=None):
    for key, values in dic.items():
        if type(values) == omegaconf.dictconfig.DictConfig:
            if name is not None:
                key = name + "." + key
            log_params(values, key)
        else:
            mlflow.log_param(key, values)
            print(key,":", values)


@hydra.main('../config/config.yaml')
def main(cfg):

    mlflow.set_tracking_uri('file://' + hydra.utils.get_original_cwd() + '/mlruns')
    mlflow.set_experiment(cfg.ex_name)

    with mlflow.start_run() as run:
        log_params(cfg)

        train, test= preprocess(
                "/kaggle/input/lish-moa/train_features.csv",
                "/kaggle/input/lish-moa/test_features.csv",
                cfg.preprocess.function_name,
                cfg.preprocess.kwargs)

        train_targets = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
        sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")

        train_targets = train_targets.drop("sig_id", axis=1)

        solver = Solver(features=train, targets=train_targets, test_features=test, sub=sub, **cfg.train)
        score = solver.train_pred()

        # with open("res.json", "w") as f:
            # json.dump(res_dict, f)

        mlflow.log_metric("CV score", score)


if __name__ == "__main__":
    main()
