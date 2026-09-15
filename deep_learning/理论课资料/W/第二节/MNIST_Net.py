import cv2
import torch
import numpy as np
from torch import nn

import os

"""
创建自定义的数据集需要继承torch下的nn.Module类
实现两个方法：
__init__：网络的初始化函数, 在该函数中 定义所需要的神经网络层
forward: 前向传播函数
"""


class MNIST_Net(nn.Module):
    def __init__(self):
        super().__init__()  # 需要实现父类的__init__方法，不然会报错
        # 整个全连接神经网络放在Sequential这个容器之中
        # bias:设置是否存在偏移量，默认为有
        # 使用Linear实现全连接神经网络的一层
        # 图片的原尺寸为28*28，转化为784，输入层为784，输出层为512
        # 使用ReLU激活函数进行激活
        self.layer1 = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.layer4 = nn.Linear(64, 10)

    # 进行前向传播返回结果
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


if __name__ == '__main__':

    # 从MNIST_Net类中实例化一个网络, 命名为 net
    net = MNIST_Net()
    # 载入已经训练好的网络参数值
    net.load_state_dict(torch.load('fcn_mnist_4layers_ckpt.pt', map_location='cpu'))

    # 从 指定路径 path 加载测试图像

    data_root = "MNIST_IMG/TEST"
    class_path = os.listdir(data_root)
    for c in class_path:
        img_names = os.listdir(os.path.join(data_root, c))
        for img_name in img_names:
            path = os.path.join(data_root, c, img_name)
            test_img = cv2.imread(path, 0) # 用cv2读入灰度图片

            test_img = test_img.raad
    path = 'MNIST_IMG/TEST/0/0.jpg'
    test_img = cv2.imread(path, 0)  # 用 cv2 读入灰度图片
    # print(test_img)
    # print(test_img.shape)
    test_img = test_img.reshape(1, 784)  # 把图像从 二维矩阵，变形为一个向量
    # print(test_img)
    # print(test_img.shape)
    test_img = np.float32(test_img / 255)  # 归一化，把图像中的每一个像素值都除以255，使图像像素值范围从原本的 [0,255] 区间 变为 [0, 1]
    test_img = torch.from_numpy(test_img)  # 把test_img 从 numpy.ndarray格式 转化成 torch.Tensor 格式
    # print(test_img)

    # 从MNIST_Net类中实例化一个网络, 命名为 net
    net = MNIST_Net()
    # 载入已经训练好的网络参数值
    net.load_state_dict(torch.load('fcn_mnist_4layers_ckpt.pt', map_location='cpu'))

    score = net(test_img)  # 输入测试图像，此时net会自动调用执行forward函数的过程，返回网络的输出结果
    print("网络输出：", score)

    softmax = nn.Softmax(dim=1)
    score = softmax(score)  # 利用 softmax 将 网络输出结果转化为是分类的概率向量
    print("预测概率：", score)

    answer = torch.argmax(score, dim=1)  # 利用 argmax，找到score中最大值的位置
    print("输入图像的类别序号为：", answer)
