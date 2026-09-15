from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch

from MNIST_Net import MNIST_Net
import torch.nn as nn
import matplotlib.pyplot as plt
from my_dataset import MNIST_Data



if __name__ == '__main__':

    epoch = 10
    learning_rate = 0.01

    # 从MNIST_Net类中实例化一个网络, 命名为 net
    net = MNIST_Net()
    # 载入已经训练好的网络参数值
    net.load_state_dict(torch.load('fcn_mnist_4layers_ckpt.pt', map_location='cpu'))
    # 测试集目录位置
    test_data_root = r'D:\study\深度学习\理论课资料\W\第二节\MNIST_IMG\TEST'
    # 训练集目录位置
    train_data_root = r'D:\study\深度学习\理论课资料\W\第二节\MNIST_IMG\TRAIN'
    # 构建训练集dataloader
    train_data = MNIST_Data(train_data_root)
    train_dataloader = DataLoader(dataset=train_data, batch_size=10, shuffle=True)
    # 构建测试集dataloader
    test_data = MNIST_Data(test_data_root)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, shuffle=False)



    net = MNIST_Net()
    # 定义损失函数
    loss_ce = nn.CrossEntropyLoss()
    # 定义优化器
    opt = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)  # learning rate  学习率

    loss = nn.CrossEntropyLoss()



    loss_epoch = []
    score_epoch = []

    for e in range(epoch):
        loss_all = 0
        for img, label in train_dataloader:
                score = net(img)
                loss = loss_ce(score, label)
                opt.zero_grad()
                loss.backward()
                opt.step()

                loss_all += loss
        print("Epoch %d, loss: %.2f" %(e, loss_all/len(train_dataloader)))
        loss_epoch.append(loss_all.detach().numpy() / len(train_dataloader))

        if not os.path.exists('train_checkpoint'):
            os.mkdir("train_checkpoint")
        torch.save(net.state_dict(), "train_checkpoint/mnist_ckpt_%d.pth" % e)

    torch.save(net.state_dict(), "mnist_ckpt.pt") # checkpoint
    score_all = 0

    for img, label in test_dataloader:
        score = net(img)  # 输入测试图像，此时net会自动调用执行forward函数的过程，返回网络的输出结果
        softmax = nn.Softmax(dim=1)
        score = softmax(score)  # 利用 softmax 将 网络输出结果转化为是分类的概率向量
        # print("预测概率：", score)
        answer = torch.argmax(score, dim=1)  # 利用 argmax，找到score中最大值的位置
        # print("输入图像的类别序号为：", answer)

        score_batch = torch.eq(answer, label).float().sum()  # 计算结果是否和类别标签一致
        score_all = score_all + score_batch

    score_all = score_all / len(test_data)  # 计算平均值
    print('Test score is: %.2f' % score_all)


    x = range(epoch)
    plt.plot(x, loss_epoch)
    plt.title("loss")
    plt.show()
    plt.plot(x, score_epoch)
    plt.title("score")
    plt.show()