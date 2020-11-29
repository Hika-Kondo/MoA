import torch
import pandas as pd


def label_encoding(data: pd.DataFrame,
                   encode_cols: list):
    for col in encode_cols:
        lbl = preprocessing.LabelEncoder()
        data[col] = lbl.fit_transform(data[col].values)
    return data


class TrainDataset:
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            # 'x' : torch.tensor(self.features[idx, :], dtype=torch.float),
            # 'y' : torch.tensor(self.targets[idx, :], dtype=torch.float)
            'x' : torch.tensor(self.features.iloc[idx], dtype=torch.float),
            'y' : torch.tensor([self.targets.iloc[idx]], dtype=torch.float)
        }
        return dct
