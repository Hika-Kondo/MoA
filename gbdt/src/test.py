from preprocess import preprocess
from solver import hole_train, hole_train_pred

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


SEED = 1234
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)


def main():
    kwargs = {"g_n_comp": 50, "c_n_comp":15, "threshold":0.5}
    train, test= preprocess(
            "/kaggle/input/lish-moa/train_features.csv",
            "/kaggle/input/lish-moa/test_features.csv",
            "gene_cell_split_pca_use_low_drop_row_variace",
            kwargs)

    train_targets = pd.read_csv("/kaggle/input/lish-moa/train_targets_scored.csv")
    sub = pd.read_csv("/kaggle/input/lish-moa/sample_submission.csv")
    # train = train.merge(train_targets, on="sig_id")

    # train = train[train["cp_type"] != "ctl_vehicle"].reset_index(drop=True)
    # test = test[test["cp_type"] != "ctl_vehicle"].reset_index(drop=True)

    train = train.drop("cp_type", axis=1)
    test = test.drop("cp_type", axis=1)

    train = train.drop("sig_id", axis=1)
    test = test.drop("sig_id", axis=1)
    train_targets = train_targets.drop("sig_id", axis=1)

    params = {
                "objective": "regression",
                "metric": "binary_logloss",
                "random_seed":SEED,
                "device": "gpu",
                "gpu_use_dp": False,
                "boosting_type": "gbdt",
            }

    # res_params, res_score = hole_train(train[feature_cols], train[target], params)
    res_dict, score = hole_train_pred(train, train_targets, test, params, sub)
    with open("res.json", "w") as f:
        json.dump(res_dict, f)



if __name__ == "__main__":
    main()
