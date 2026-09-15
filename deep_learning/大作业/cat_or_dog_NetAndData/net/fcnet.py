import torch
import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):  # 全连接神经网络
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(784, 500)  # 第一层全连接网络输入784条通道，输出500条通道
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(500, 10)  # 第二层全连接网络输入500条通道，输出10条通道

    def forward(self, x):  # 进行全连接神经网络
        out = self.fc1(x)  # 第一层全连接网络
        out = self.relu(out)  # 激活第一层全连接网络
        out = self.fc2(out)  # 第二层全连接网络
        return out


if __name__ == "__main__":
    net = FCNet()
    print(net)