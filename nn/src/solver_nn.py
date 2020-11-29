import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split, StratifiedKFold
from sklearn.metrics import log_loss
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn

from logging import getLogger
from random import randint
from collections import defaultdict
import gc

from utils import after_process
from model import SimpleNet
from model_train import train_fn, valid_fn, inference_fn
from dataset import TrainDataset


class Solver:
    '''
    solver class for lightgbm
    '''

    def __init__(self, features, targets, test_features, is_under, mode, is_lda, sub, batch_size,
            num_hidden_layers, dropout_rate, hidden_size, epochs, device, learning_rate):
        self.features = features
        self.targets = targets
        self.test_features = test_features
        self.is_under = is_under
        self.is_lda = is_lda
        self.sub = sub
        self.mode = mode
        self.num_hidden_layers = num_hidden_layers
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.device = device
        self.learning_rate = learning_rate

        self.score = 0
        self.num_add = 0

        self.logger = getLogger("SOLVER")
        torch.backends.cudnn.benchmark = True

    def return_columns(self):
        return targets.columns

    def _train_pred(self, column, test_features, SEED=None):
        '''
        One training session.
        self.train_features are features to be used for training.
        self.train_targets[column] are the answers to the training questions.
        self.val_features are features to be used for validate
        self.val_targets[column] is the answer to validate.
        '''

        x_train, x_val, y_train, y_val = train_test_split(
                    self.features,
                    self.targets[column],
                    test_size=0.1,
                    shuffle=True,
                    # stratify=True,
                )

        model = SimpleNet(
                self.num_hidden_layers,
                self.dropout_rate,
                len(x_train.columns),
                self.hidden_size,
                1,
                )

        model.to(self.device)
        train_dataset = TrainDataset(x_train, y_train)
        trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, )

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-3)

        best_loss = np.inf
        i = torch.tensor(self.test_features.values, dtype=torch.float)

        for epoch in range(1, self.epochs):
            train_loss = train_fn(model, optimizer, nn.BCEWithLogitsLoss(), trainloader, self.device)
            valid_loss = valid_fn(model, nn.BCEWithLogitsLoss(), x_val, y_val, self.device)
            self.logger.info('Epoch:{}, train_loss:{:.5f}, valid_loss:{:.5f}'
                    .format(epoch, train_loss, valid_loss))

            if valid_loss < best_loss:
                not_update_epoch = 0
                best_loss = valid_loss
                torch.save(model.state_dict(), 'best_model_{}.pth'.format(column))
            else:
                not_update_epoch += 1
            # if early_stopping_epoch == not_update_epoch:
            #     print('early stopping')
            #     torch.save(model.state_dict(), 'best_model_{}.pth'.format(column))
            #     break

        self.score += best_loss
        self.num_add += 1
        self.logger.info("column:{} validation loss {}".format(column, best_loss))
        gc.collect()
        y_pred = inference_fn(model, self.test_features, self.device)
        return y_pred, best_loss

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
        score = 0
        now = 0

        # all columns roop
        for column in self.targets.columns:
            self.logger.info("{}/{}\t{}".format(now, len(self.targets.columns),column)); now+=1
            self.sub[column] = 0

            if column in ["atp-sensitive_potassium_channel_antagonist", "erbb2_inhibitor",]:
                self.sub[column] = 1e-5
                continue

            train_features = self.features
            test_features = self.test_features

            if self.is_lda:
                train_features, test_features = self._lda(column)

            y_pred, column_score = self._train_pred(column, test_features)
            self.sub[column] = y_pred
            # self.cv_res[column].append(score)
            # pred_np.append(pred); ans_np.append(ans)
            # score += column_score

        self.sub.to_csv("submission.csv")
        # cv_res = pd.DataFrame(self.cv_res)
        # cv_res.to_csv("CV_res.csv")

        pred_np = np.concatenate(pred_np)
        ans_np = np.concatenate(ans_np)
        after_process(pred=pred_np, ans=ans_np, name="res.png")
        return self.score / self.num_add

    def _lda(self, column):
        '''
        lda
        '''
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.features, self.targets[column])

        features = self.features
        test_features = self.test_features

        features["lda"] = clf.transform(self.features)
        test_features["lda"] = clf.transform(self.test_features)

        return features, test_features
