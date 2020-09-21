import lightgbm as lgbm
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from multiprocessing import Pool
from random import randint


class Solver:
    '''
    solver class for lightgbm
    '''

    def __init__(self, params, features, targets, test_features, num_folds, num_ensemble, sub):
        self.params = dict(params)
        self.features = features
        self.targets = targets
        self.test_features = test_features
        self.num_folds = num_folds
        self.num_ensemble = num_ensemble
        self.sub = sub

        self.score = 0
        self.cv = KFold(n_splits=num_folds, shuffle=True,)

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

        train_data = lgbm.Dataset(self.train_features, self.train_targets[column])

        model = lgbm.train(
                self.params,
                train_data,)

        y_pred = model.predict(self.test_features)
        val_pred = model.predict(self.val_features)
        self.score += log_loss(y_true=self.val_targets[column], y_pred=val_pred, labels=[0,1])
        return y_pred

    def train_pred(self):
        '''
        train and predict dataframe
        For a single column in lightgbm, we predict the following.
            1. do a CV fold (training and prediction in that fold)
            2. 1. do num_ensemble times.
        '''

        # all columns roop
        for column in self.targets.columns:
            # ensamble roop
            for i in range(self.num_ensemble):
                # CV roop
                for fold_id, (train_index, valid_index) in enumerate(self.cv.split(self.features)):
                    self.train_features = self.features.loc[train_index, :]
                    self.val_features = self.features.loc[valid_index, :]

                    self.train_targets = self.targets.loc[train_index, :]
                    self.val_targets = self.targets.loc[valid_index, :]

                    y_pred = self._train_pred(SEED=randint(0,100000), column=column)

            y_pred /= (self.num_folds * self.num_ensemble)
            self.sub[column] = y_pred

        self.sub.to_csv("submission.csv")
        return self.score / len(self.targets.columns) * self.num_ensemble * self.num_folds
