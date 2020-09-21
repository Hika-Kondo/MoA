import lightgbm as lgb_original
import optuna.integration.lightgbm as lgb
import pandas as pd
from sklearn.model_selection import KFold

from collections import defaultdict
from multiprocessing import Pool


def hole_train_lgbm(features, targets, params):
    '''
    train hole data
    '''
    res_params = {}
    res_score = {}
    targets_columns = targets.columns
    for column in targets_columns:
        res, score = _single_train(features, targets[column], params)
        res_params[column] = res
        res_score[column] = score
    return res_params, res_score


def hole_train_pred_lgbm(features, targets, test, params, sub):
    params = dict(params)
    res_dict= []
    targets_columns = targets.columns
    sum_score = 0

    for column in targets_columns:
        res, score = _single_train(features, targets[column], params)
        res_dict.append({"column": column, "res": res})
        sum_score += score

        pred = _single_pred(features, targets[column], test, res)
        sub[column] = pred
        return res_dict, sum_score / len(targets_columns)


def _single_train(features, targets, params):
    '''
    train single column of target
    '''

    trainval = lgb.Dataset(features, targets)
    tuner = lgb.LightGBMTunerCV(
                params,
                trainval,
                verbose_eval=100,
                early_stopping_rounds=100,
                folds=KFold(n_splits=3),
            )
    tuner.run()
    return tuner.best_params, tuner.best_score


def _single_pred(features, targets, test, params):
    '''
    single column predict
    '''
    print(features.shape, targets.shape)
    train_val = lgb_original.Dataset(features, targets)
    train_session = lgb_original.train(
                params,
                train_val,
                verbose_eval=100,
            )

    pred = train_session.predict(test, num_iteration=train_session.best_iteration)
    return pred
