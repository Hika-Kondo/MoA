import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import QuantileTransformer

from copy import deepcopy


def split_df(data, train_size, test_size):
    train = data[:train_size]
    test = data[-test_size:]
    return train, test


def pca(df, n_comp, columns_name):
    '''
    pca
    args:
        df: pandas dataframe you want to pca
        n_comp: number of dimentions to pca
    '''
    data = (PCA(n_components=n_comp)).fit_transform(df)
    data = pd.DataFrame(data, columns=[columns_name.format(i) for i in range(n_comp)])
    return data


def ica(df, n_comp, columns_name):
    '''
    ica
    args:
        df: pandas dataframe
        n_comp: number of dimention to lca
        columns_name: res features name
    '''
    data = (FastICA(n_componets=n_comp)).fit_transform(df)
    return pd.DataFrame(data, columns=[columns_name.format(i) for i in raneg(n_comp)])


def rankgauss(df):
    '''
    Rank Gauss func
    args:
        df: train feature pandas dataframe
    '''
    df_cpy = deepcopy(df)

    for col in df.columns:
        transformer = QuantileTransformer(n_quantiles=100, random_state=1234, output_distribution="normal")

        vec_len = len(df[col].values)
        raw_vec = df[col].values.reshape(vec_len, 1)
        transformer.fit(raw_vec)

        df_cpy[col] = transformer.transform(raw_vec).reshape(1, vec_len)[0]

    return df_cpy


def drop_low_variace(df, threshold):
    var_thresh = VarianceThreshold(threshold=threshold)
    data_transformed = var_thresh.fit_transform(df.iloc[:, 4:])
    data_transformed = pd.DataFrame(data_transformed)
    train_features = pd.DataFrame(
        df[["sig_id","cp_type","cp_time","cp_dose"]].values.reshape(-1, 4),
        columns=["sig_id","cp_type","cp_time","cp_dose"])

    return pd.concat([train_features, data_transformed], axis=1)
