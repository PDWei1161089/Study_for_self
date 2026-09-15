import torch
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import datasets, models, transforms
import os
import torch.nn as nn
import tensorflow as tf
import tensorflow_datasets as tfds
import cv2

# 模型
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


path = "data"

# 数据加载，模型训练等代码需要自己补全

model = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.DEFAULT)  # 使用VGG16模型

model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 4096),
                                       torch.nn.ReLU(),
                                       torch.nn.Dropout(p=0.5),
                                       torch.nn.Linear(4096, 2))

# 初始化模型、损失函数和优化器
model = FCNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 加载数据
transform = transforms.Compose([transforms.CenterCrop(784 * 500),
                                transforms.ToTensor(),
                                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
# CenterCrop中心裁剪224*224大小的图片，ToTensor转为张量，Normalize标准化操作，均值为[0.5, 0.5, 0.5], 标准差为[0.5, 0.5, 0.5]


# train_dataset = ImageFolder(r"D:\study\深度学习\大作业\cat_or_dog_NetAndData\net\data\train", transform=transform)
# train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)
dataset_train = tfds.load(r"D:\study\深度学习\大作业\cat_or_dog_NetAndData\net\data\train", shuffle=True, batch=100)

for batch in dataset_train.take(4500):
    # 训练模型
    epochs = 10
    for epoch in range(epochs):
        train_loader = epoch
        for i, (images, labels) in enumerate(train_loader):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backword()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(
                    f"训练次数：[{epoch + 1}/{epochs}], 已完成比例 [{i + 1}/{len(train_loader)}, 损失率：{loss.item():.4f}]")
print("模型已训练完毕")

dataset_test = tfds.load(r"D:\study\深度学习\大作业\cat_or_dog_NetAndData\net\data\test", shuffle=True, batch=100)
for batch in dataset_test.take(4500):
    # 测试模型
    test_dataset = ImageFolder(r"net\data\test", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=True)

    # 在测试集上的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"在测试集上的准确率为：{100 * correct / total}%")

'''for parma in model.parameters():
    parma.requires_grad = False

for index, parma in enumerate(model.classifier.parameters()):
    if index == 6:
        parma.requires_grad = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_gpu = False
if use_gpu:
    model = model.cuda()

#cost = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.classifier.parameters())
'''
