import torch.nn as nn


class SimpleNet(nn.Module):

    def __init__(self, num_hidden_layers, dropout_rate, input_size, hidden_size, output_size):
        super(SimpleNet, self).__init__()
        layer = []
        layer.append(nn.Linear(input_size, hidden_size))
        for i in range(num_hidden_layers-1):
            layer.append(nn.BatchNorm1d(hidden_size))
            layer.append(nn.Dropout(dropout_rate))
            layer.append(nn.Linear(hidden_size, hidden_size))
            layer.append(nn.ReLU())
        self.layer = nn.ModuleList(layer)
        self.ft_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        for f in self.layer:
            x = f(x)
        x = self.ft_layer(x)
        return x
