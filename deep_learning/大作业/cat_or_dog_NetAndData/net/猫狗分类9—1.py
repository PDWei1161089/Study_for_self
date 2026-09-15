import os  # 文件操作模块
import torch  # 搭建pytorch框架
import torch.utils.data as data  # 用于继承一个父类（data.Dataset）里的函数
from PIL import Image  # 图像处理库
import torchvision.transforms as Trans  # 在定义图片转换的格式时，会用到相关的函数
from torch.utils.data import DataLoader as DataLoader  # 数据加载器
from torch.autograd import Variable  # 构建计算图并进行梯度自动求导
import torch.nn.functional as F  # 可以实现神经网络函数式接口
import torch.nn as nn  # 调用及构建神经网络

img_size = 200  # 设置图片尺寸

tran = Trans.Compose([Trans.Resize(img_size),
                      Trans.CenterCrop([img_size, img_size]),
                      Trans.ToTensor()])


# 封装，对后面读取的图片的格式转换，改变图片尺寸（中心裁剪）为200*200，并转换为张量

class DVCD(data.Dataset):  # 数据集的导入
    def __init__(self, mode, dir):
        self.data_size = 0  # 数据集的大小
        self.img_list = []  # 用于存图
        self.img_label = []  # 标签
        self.trans = tran  # 转换的属性设置
        self.mode = mode  # 下面打开集的模式

        if self.mode == 'train':  # 对于训练集的操作
            dir += '/train/'  # 更新地址
            for file in os.listdir(dir):  # 遍历
                self.img_list.append(dir + file)  # 存图
                self.data_size += 1  # 数据集大小更新
                name = file.split(sep='.')  # 将 该文件名拆分，便于判断是cat还是dog
                label_x = 0  # 判断是猫还是狗
                if name[0] == 'cat':  # 对于猫的情况
                    label_x = 1  # 规定标签为1的是猫
                self.img_label.append(label_x)  # 设置入相对于的标签；cat:1； dog:0

        elif self.mode == 'test':  # 对于测试集的操作
            dir += '/test/'  # 更新地址
            for file in os.listdir(dir):  # 遍历
                self.img_list.append(dir + file)  # 存图
                self.data_size += 1  # 数据集大小更新
                self.img_label.append(2)  # 无意义的标签
        else:  # 其他非正常情况
            print("没有这个mode")

    def __getitem__(self, item):  # 获取数据
        if self.mode == 'train':  # 对于训练集的操作
            img = Image.open(self.img_list[item])  # 打开并加载图片文件
            label_y = self.img_label[item]  # 记忆图片的标签
            return self.trans(img), torch.LongTensor([label_y])  # 返回该图片的地址和标签
        elif self.mode == 'test':  # 对于测试集的操作
            img = Image.open(self.img_list[item])  # 打开并加载图片
            return self.trans(img)  # 转换属性
        else:  # 其他非正常情况
            print("None")

    def __len__(self):  # 数据集的大小
        return self.data_size


class Net(nn.Module):  # 继承nn.Module 里的东西
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, padding=1)
        # 第一个卷积层，Conv2d用于图片的卷积，输入神经元数量为3，输出神经元数量为16，卷积核为3*3，移动步长为1。
        self.conv2 = torch.nn.Conv2d(16, 16, 3, padding=1)
        # 第二个卷积层，输入神经元数量为16，输出神经元数量为16，卷积核大小为3*3，移动步长为1

        self.fc1 = torch.nn.Linear(50 * 50 * 16, 128)
        # 第一个线性层，输入特征数量为50*50*16，输出特征量为128
        self.fc2 = torch.nn.Linear(128, 64)
        # 第二个线性层，输入特征量为128，输出特征量为64
        self.fc3 = torch.nn.Linear(64, 2)
        # 第三个线性层，输入特征量为64，输出特征量为2（猫或者狗

    def forward(self, x):  # 定义前向传播函数
        x = self.conv1(x)  # 卷积
        x = F.relu(x)  # 激活
        x = F.max_pool2d(x, 2)  # 池化，池化核为2*2

        x = self.conv2(x)  # 卷积
        x = F.relu(x)  # 卷积
        x = F.max_pool2d(x, 2)  # 池化，池化核为2*2

        x = x.view(x.size()[0], -1)  # 改变张量维度为二维
        x = F.relu(self.fc1(x))  # 全连接
        x = F.relu(self.fc2(x))  # 全连接
        x = self.fc3(x)  # 线性变换

        return F.softmax(x, dim=1)  # 将输出转换为概率分布


dataset_dir = r'D:\study\深度学习\大作业\cat_or_dog_NetAndData\net\data101\train'  # 训练集路径

model_dir = r'D:\study\深度学习\大作业\cat_or_dog_NetAndData\net\data101\model'
# 网络参数保存位置
workers = 10  # 线程数量
batch_size = 16  # 一次训练所选取的样本数
lr = 0.001  # 学习率
nepoch = 10  # 训练的次数


def train():  # 定义训练函数
    datafile = DVCD('train', dataset_dir)  # 实例化
    dataloader = DataLoader(datafile,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=workers,
                            drop_last=True)
    # 封装，数据打乱，剩下的样本数据被丢弃

    print('数据加载完成! 训练数据总数为：{0}'.format(len(datafile)))

    model = Net()  # 实例化一个网络

    model = nn.DataParallel(model)  # 传入
    model.train()  # 训练模式

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Adam优化器

    Lossfuc = torch.nn.CrossEntropyLoss()  # loss计算方法，cross entropy，交叉熵

    for epoch in range(nepoch):  # 进行nepoch次训练
        cnt = 0  # 重置训练图片数量
        for img, label in dataloader:  # 遍历
            img, label = Variable(img), Variable(label)  # 将数据放置在PyTorch的Variable节点中
            out = model(img)  # 设置输出值
            loss = Lossfuc(out, label.squeeze())  # 计算损失
            loss.backward()  # 启动反向传播算法，计算损失函数对网络参数的梯度
            optimizer.step()  # 优化器更新模型权重
            optimizer.zero_grad()  # 清楚模型参数梯度
            cnt += 1  # 更新已训练图片的数量
            acc = (1 - loss / batch_size) * 100  # 准确率
            cnt_epoch = epoch + 1  # 轮次
            cnt_cnt = cnt / 1562 * 100  # 计算已完成程度
            if (cnt) % (260) == 2:  # 便于查看是否再该轮次下全部训练完毕
                print(f'训练轮次:{cnt_epoch} / {nepoch}, 第{cnt_epoch}轮训练完成度:{cnt_cnt:.2f}%, 准确率 {acc:.5f}%')
    torch.save(model.state_dict(), '{0}/model.pth'.format(model_dir))  # 训练所有数据后，保存网络的参数


if __name__ == '__main__':  # 主函数
    train()  # 开始训练

# #from getdata import DogsVSCatsDataset as DVCD #test里的基本框架和train里的差不多，这里不细讲
# #from network import Net
# from torch.autograd import Variable
# import numpy as np
# import torch.nn.functional as F
# import torch
# import torch.nn as nn
# import matplotlib.pyplot as plt
# from PIL import Image
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #防止打印图片出错
#
# data_dir = r'D:\study\深度学习\大作业\cat_or_dog_NetAndData\net\data101\test'
# model_file = './model/model.pth'
# N = 10
#
# model =Net()
# model = nn.DataParallel(model)
# model.load_state_dict(torch.load(model_file))
# model.eval()
#
# datafile = DVCD('test', data_dir)
#
# index = np.random.randint(0, datafile.data_size, 1)[0]
# img = datafile.__getitem__(index)
#
# img = img.unsqueeze(0)
# img = Variable(img)
#
# out =model(img)
# out = F.softmax(out, dim=1)
# print(out.data)
# if out[0, 0]>out[0, 1]:
#     print("the picture is a cat")
# else:
#     print("the picture is a dog")
# img = Image.open(datafile.img_list[index])
# plt.figure('image')
# plt.imshow(img)
# plt.show()
