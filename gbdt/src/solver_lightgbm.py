import numpy as np
import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss

from random import randint
from collections import defaultdict
from utils import after_process


class Solver:
    '''
    solver class for lightgbm
    '''

    def __init__(self, params, features, targets, test_features, num_folds, num_ensemble, is_under, sub):
        self.params = dict(params)
        self.features = features
        self.targets = targets
        self.test_features = test_features
        self.num_folds = num_folds
        self.num_ensemble = num_ensemble
        self.is_under = is_under
        self.sub = sub

        self.score = 0
        self.num_add = 0
        self.cv = KFold(n_splits=num_folds, shuffle=True,)
        self.cv_res = defaultdict(list)

        # self.pred_np = []
        # self.ans_np = []

    def return_columns(self):
        return targets.columns

    def _train_pred(self, column, SEED=None):
        '''
        One training session.
        self.train_features are features to be used for training.
        self.train_targets[column] are the answers to the training questions.
        self.val_features are features to be used for validate
        self.val_targets[column] is the answer to validate.
        '''
        # sigle train and pred
        if SEED is not None:
            self.params["random_seed"] = SEED

        X_train, X_val, Y_train, Y_val = train_test_split(
                    self.train_features,
                    self.train_targets[column]
                )
        train_data = lgbm.Dataset(X_train, Y_train)
        val_data = lgbm.Dataset(X_val, Y_val)

        model = lgbm.train(
                    self.params,
                    train_data,
                    valid_sets=[train_data, val_data],
                    num_boost_round=1000,
                    early_stopping_rounds=10,
                    verbose_eval=10,
                )

        y_pred = model.predict(self.test_features, num_iteration=model.best_iteration)
        val_pred = model.predict(self.val_features, num_iteration=model.best_iteration)

        # self.pred_np.append(val_pred)
        # self.ans_np.append(np.asarray(self.val_targets[column]))

        score = log_loss(y_true=self.val_targets[column], y_pred=val_pred, labels=[0,1])
        self.score += score
        self.num_add += 1
        return y_pred, score, val_pred, np.asarray(self.val_targets[column])

    def _under_sampling(self, column):
        '''
        do under sampling
        '''
        features_col = self.train_features.columns
        targets_col = self.train_targets.columns

        df = pd.concat([self.train_features, self.train_targets], axis=1)
        value_counts = df[column].value_counts().sort_values(ascending=True)
        df_each_classes = []
        for i in range(len(value_counts)):
            df_one_class = df[df[column] == value_counts.index[i]]
            if i != 0:
                df_one_class = df_one_class.sample(n=len(df_each_classes[0]))
            df_each_classes.append(df_one_class)

        df_balanced = pd.concat(df_each_classes, axis=0)
        df_balanced = df_balanced.reset_index(drop=True)

        return df_balanced[features_col], df_balanced[targets_col]

    def train_pred(self):
        '''
        train and predict dataframe
        For a single column in lightgbm, we predict the following.
            1. do a CV fold (training and prediction in that fold)
            2. 1. do num_ensemble times.
        '''

        pred_np = []; ans_np = []

        # all columns roop
        for column in self.targets.columns:
            self.sub[column] = 0
            # ensamble roop
            self.cv_res["sig_id"].append(column)
            for i in range(self.num_ensemble):
                # CV roop
                for fold_id, (train_index, valid_index) in enumerate(self.cv.split(self.features)):
                    self.train_features = self.features.loc[train_index, :]
                    self.val_features = self.features.loc[valid_index, :]

                    self.train_targets = self.targets.loc[train_index, :]
                    self.val_targets = self.targets.loc[valid_index, :]

                    if self.is_under:
                        self.train_feature, self.train_target = self._under_sampling(column)

                    y_pred, score, pred, ans = self._train_pred(SEED=randint(0,10000), column=column)
                    y_pred /= (self.num_folds * self.num_ensemble)
                    self.sub[column] += y_pred
                    self.cv_res["num_ensemble_{}_num_fold{}".format(i,fold_id)].append(score)
                    pred_np.append(pred); ans_np.append(ans)

        self.sub.to_csv("submission.csv")
        cv_res = pd.DataFrame(self.cv_res)
        cv_res.to_csv("CV_res.csv")

        pred_np = np.concatenate(pred_np)
        ans_np = np.concatenate(ans_np)
        after_process(pred=pred_np, ans=ans_np, name="res.png")
        return self.score / self.num_add
