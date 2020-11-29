from .utils import split_df
import preprocess.functions

import pandas as pd


def preprocess(train, test, function_name, kwargs):
    '''
    preprocess of input features.
    Execute preprocessing functions along the lines of cfg
    return preprecessed train and test df
    args:
        train: train df
        test:  test df
        cfg:   preprocess functions
    '''
    train = pd.read_csv(train); test = pd.read_csv(test)
    train_size = train.shape[0]; test_size = test.shape[0]
    concat_df = pd.concat([train, test])
    concat_df = getattr(functions, function_name)(concat_df, **kwargs)

    concat_df = pd.get_dummies(concat_df, columns=["cp_time", "cp_dose"])
    train, test = split_df(concat_df, train_size, test_size)
    train = train.drop("cp_type", axis=1)
    test = test.drop("cp_type", axis=1)

    train = train.drop("sig_id", axis=1)
    test = test.drop("sig_id", axis=1)

    return train, test
