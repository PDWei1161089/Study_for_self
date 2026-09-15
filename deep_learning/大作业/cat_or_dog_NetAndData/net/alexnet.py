import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, 11, 4, 0)  # 定义conv1函数的是图像卷积函数：输入为图像（3个通道RGB，即彩色图）,卷积产生的通道数为96，输出为96张特征图,
        # 卷积核为11*11的正方形，卷积步长为4，不填充
        self.pool = nn.MaxPool2d(3, 2)  # 池化窗口大小3*3， 移动步长为2
        self.conv2 = nn.Conv2d(96, 256, 5, 1, 2)  # 定义conv2函数的是图像卷积函数，输入为96条通道的图像，卷积输出为256条通道，卷积核大小为5*5，卷积步长为1，填充零值像素为2
        self.conv3 = nn.Conv2d(256, 384, 3, 1, 1)  # 定义conv3函数的是图像卷积函数，输入为图像256条通道，卷积输出为384条通道，卷积核大小为3*3，卷积步长为1，填充零值像素为1
        self.conv4 = nn.Conv2d(384, 384, 3, 1, 1)  # 定义conv4函数的是图像卷积函数，输入为图像384条通道，卷积输出为384条通道，卷积核大小为3*3，卷积步长为1，填充零值像素为1
        self.conv5 = nn.Conv2d(384, 256, 3, 1, 1)  # 定义conv4函数的是图像卷积函数，输入为图像384条通道，卷积输出为256条通道，卷积核大小为3*3，卷积步长为1，填充零值像素为1
        self.drop = nn.Dropout(0.5)  # 有50%的概率随机丢弃神经元的输出
        self.fc1 = nn.Linear(9216, 4096)  # 全连接网络，输入神经元为96*96=9216， 输出神经元为4096
        self.fc2 = nn.Linear(4096, 4096)  # 全连接网络，输入神经元为4096， 输出神经元为4096
        self.fc3 = nn.Linear(4096, 100)   # 全连接网络，输入神经元为4096， 输出神经元为100

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 池化第一个卷积层
        x = self.pool(F.relu(self.conv2(x)))  # 池化第二个卷积层
        x = F.relu(self.conv3(x))  # 激活第三层卷积层
        x = F.relu(self.conv4(x))  # 激活第四层卷积层
        x = self.pool(F.relu(self.conv5(x)))  # 激活并池化第五层卷积层
        x = x.view(-1, self.num_flat_features(x))  # 将x重新塑型为二维张量
        x = F.relu(self.fc1(x)) # 激活第一层全连接网络
        x = self.drop(F.relu(self.fc1(x)))  # 将激活的第一层全连接网络中的单元随机丢弃，防止数据过拟合
        x = self.drop(F.relu(self.fc2(x)))  # 将第二层全连接网络激活并随机丢弃已激活的单元，防止数据过拟合
        x = self.fc3(x)  # 实现第三层全连接网络
        return x

    def num_flat_features(self, x):  # self参数可能不被需要
        size = x.size()[1:]  # all dimensions except the batch dimension 取出除样本数量外的其他维度参数
        num_features = 1  # 初始化参数卷积核的数量
        for s in size:
            num_features *= s  # 计算总的特征数量
        return num_features


if __name__ == "__main__":
    net = AlexNet()  # 实现Alexnet
    print(net)