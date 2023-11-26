import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout_rate=0.3):
        super(Net, self).__init__()
        
        self.hidden1 = nn.Linear(n_feature, n_hidden)
        # 第一个 Dropout 层
        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.hidden2 = nn.Linear(n_hidden, n_hidden)
        
        # 第二个 Dropout 层
        self.dropout2 = nn.Dropout(p=dropout_rate)
        self.hidden3 = nn.Linear(n_hidden, n_hidden)
        
        # 第二个 Dropout 层
        self.dropout3 = nn.Dropout(p=dropout_rate)
        self.predict = nn.Linear(n_hidden, n_output)
        self.relu = nn.ReLU()

    def forward(self, x_train_normalized):
        x_train_normalized = self.relu(self.hidden1(x_train_normalized))
        # 应用第一个 Dropout
        x_train_normalized = self.dropout1(x_train_normalized)
        x_train_normalized = self.relu(self.hidden2(x_train_normalized))
        # 应用第二个 Dropout
        x_train_normalized = self.dropout2(x_train_normalized)
        x_train_normalized = self.relu(self.hidden3(x_train_normalized))
        # 应用第二个 Dropout
        x_train_normalized = self.dropout3(x_train_normalized)
        x_train_normalized = self.predict(x_train_normalized)
        
        return x_train_normalized