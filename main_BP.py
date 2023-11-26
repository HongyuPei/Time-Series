from data_loader import dataloading,normalize,dataset
import torch
from torch.utils.data import DataLoader, random_split
from model import Net
from train_test import train,test



tensor_data = torch.from_numpy(dataloading())
tensor_data = normalize(tensor_data)



total_samples = len(tensor_data)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size
# 使用 random_split 函数划分数据集
train_dataset, test_dataset = random_split(tensor_data, [train_size, test_size])

# 创建 DataLoader 用于加载数据
train_loader = DataLoader(dataset(train_dataset.dataset), batch_size=64, shuffle=True)
test_loader = DataLoader(dataset(test_dataset.dataset), batch_size=64, shuffle=True)

net = Net(n_feature = 8, n_hidden = 64, n_output = 1, dropout_rate=0.3)
# 定义路径
train(model = net, data = train_loader, epochs = 200, learning_rate = 0.0001, betas=(0.9, 0.999))

test(model = net, data = train_loader, learning_rate = 0.001, betas=(0.9, 0.999))