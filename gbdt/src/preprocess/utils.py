import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold


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


def drop_low_variace(df, threshold):
    var_thresh = VarianceThreshold(threshold=threshold)
    data_transformed = var_thresh.fit_transform(df.iloc[:, 4:])
    data_transformed = pd.DataFrame(data_transformed)
    train_features = pd.DataFrame(
        df[["sig_id","cp_type","cp_time","cp_dose"]].values.reshape(-1, 4),
        columns=["sig_id","cp_type","cp_time","cp_dose"])

    return pd.concat([train_features, data_transformed], axis=1)
