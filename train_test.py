import torch
import matplotlib.pyplot as plt
import numpy as np
import os

# 训练
def train(model, data, epochs = 50,  learning_rate = 0.0001, betas=(0.9, 0.999)):
    # optimizer 是训练的工具
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=betas)  # 传入 net 的所有参数, 学习率
    loss_func = torch.nn.MSELoss()  # 预测值和真实值的误差计算公式 (均方差)
    # 训练与绘制训练图像
    plt.ion()  # 画图
    plt.show()
    for epoch in range(epochs):
        for i, data_i in enumerate(data, 0):
            x_train,y_train = data_i
            y_train = y_train.unsqueeze(-1)
            model.train()
            # net训练数据x, 输出预测值
            prediction = model(x_train)
            # 计算两者的均方误差
            loss = loss_func(prediction, y_train)
            # 上一步的更新梯度留在net.parameters()中，清空上一步的残余更新参数值
            optimizer.zero_grad()
            # 误差反向传播, 计算参数更新值
            loss.backward()
            # 更新参数
            optimizer.step()

            # 每五步绘制一次
            if i % 10 == 0:
                # plot and show learning process
                plt.cla()
                X = np.linspace(0, len(np.array(y_train)), len(np.array(y_train)))
                plt.plot(X, y_train.detach().numpy(), marker='.', label="origin data")
                plt.xticks([])
                plt.plot(X, prediction.detach().numpy(), 'r-', marker='.', label="predict", lw=1)
                plt.xticks([])
                plt.text(2, 1, 'Loss=%.4f' % loss, fontdict={'size': 20, 'color': 'red'})
                plt.text(2, 2, 'times=%d' % (i + 5), fontdict={'size': 15, 'color': 'red'})
                plt.legend(loc="upper right")
                plt.pause(0.1)

        print([epoch,loss])
    torch.save(model.state_dict(), "BPNN.pt")
    # plt.pause(0)


# 测试
def test(model, data, learning_rate, betas):
    model.load_state_dict(torch.load('BPNN.pt'))
    with torch.no_grad():
        for i, data_i in enumerate(data, 0):
            x_test,y_test = data_i   
            y_test = y_test.unsqueeze(-1)
            output = model(x_test)
            loss = torch.nn.MSELoss()
            error = loss(output, y_test)
            deviation_mean = torch.mean(torch.abs(torch.sub(y_test, output)))
    plt.title("test")
    x = np.linspace(0, len(y_test), len(y_test))
    plt.plot(x, y_test, color='blue', marker='.', label="test data")
    plt.plot(x, output, color='yellow', marker='.', label="predict data")
    plt.xticks([])
    plt.text(0, 0.97, 'error=%.4f' % error, fontdict={'size': 15, 'color': 'red'})
    plt.text(0, 0.92, 'mean deviation=%.4f' % deviation_mean, fontdict={'size': 15, 'color': 'red'})
    plt.legend(loc="upper right")
    plt.savefig('test.png')
    plt.show()