import torch


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, dropout_rate=0.3):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)  # 第一个 Dropout 层
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)  # 第二个 Dropout 层
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        self.dropout3 = torch.nn.Dropout(p=dropout_rate)  # 第二个 Dropout 层
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x_train_normalized):
        x_train_normalized = torch.relu(self.hidden1(x_train_normalized))
        x_train_normalized = self.dropout1(x_train_normalized)  # 应用第一个 Dropout
        x_train_normalized = torch.relu(self.hidden2(x_train_normalized))
        x_train_normalized = self.dropout2(x_train_normalized)  # 应用第二个 Dropout
        x_train_normalized = torch.relu(self.hidden3(x_train_normalized))
        x_train_normalized = self.dropout3(x_train_normalized)  # 应用第二个 Dropout
        x_train_normalized = self.predict(x_train_normalized)
        return x_train_normalized