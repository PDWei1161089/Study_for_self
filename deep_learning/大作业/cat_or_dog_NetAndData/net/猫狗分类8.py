import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import transforms
from torchvision import models
from torchvision.io import read_image
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import os
os.environ['TORCH_HOME']='./download_model'  # 修改Pytorch模型默认下载位置

# 获取图片路径
train_path = r'D:\study\深度学习\大作业\cat_or_dog_NetAndData\net\data101\train'

image_file_path = np.array([train_path + i for i in os.listdir(train_path)])

# 根据文件名获取图片对应的标签
labels = np.array([
    0 if name.split('/')[-1].startswith('cat') else 1
    for name in image_file_path
])

print(len(image_file_path),len(labels))

# 分层抽样

# 将数据集按照 8:2 的比例分为训练集和其他数据集
train_val_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=2021)

for train_index, val_index in train_val_sss.split(image_file_path, labels):
    train_path, val_path = image_file_path[train_index], image_file_path[val_index]
    train_labels, val_labels = labels[train_index], labels[val_index]

# 将其他数据集按照 1:1 的比例分为验证集和测试集
val_test_sss = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=2021)

for val_index, test_index in val_test_sss.split(val_path, val_labels):
    val_path, test_path = val_path[val_index], val_path[test_index]
    val_labels, test_labels = val_labels[val_index], val_labels[test_index]

# 最终我们得到 8:1:1 的训练集、验证集和测试集
# 每个数据集中猫和狗所占比例相同

# 由于猫的标签是0，狗的标签是1，因此我们可以用.sum()函数获取到狗的个数
print(len(train_path),len(train_labels),train_labels.sum())
print(len(val_path),len(val_labels),val_labels.sum())
print(len(test_path),len(test_labels),test_labels.sum())

# 定义Dataset加载方式
class MyData(Dataset):
    def __init__(self, filepath, labels=None, transform=None):
        self.filepath = filepath
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image = read_image(self.filepath[index]) # 读取后的图片维度是[channl, height, width]
        image = image.to(torch.float32) / 255. # 转换为float32类型，除以255进行归一化
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is not None:
            return image, self.labels[index]
        return image

    def __len__(self):
        return self.filepath.shape[0]

image_size = [224, 224]  # 图片大小
batch_size = 64 # 批大小

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),	# 随机水平翻转
    transforms.RandomRotation(30),	# 随机旋转
    transforms.Resize([256, 256]),	# 设置图片大小
    transforms.RandomCrop(image_size),	# 将图片随机裁剪为所需图片大小
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 使用ImageNet的标准化参数
])

ds1 = MyData(train_path, train_labels, transform)
ds2 = MyData(val_path, val_labels, transform)
ds3 = MyData(test_path, test_labels, transform)

train_ds = DataLoader(ds1, batch_size=batch_size, shuffle=True)
val_ds = DataLoader(ds2, batch_size=batch_size, shuffle=True)
test_ds = DataLoader(ds3, batch_size=batch_size, shuffle=True)

# 对一张图片查看增强效果

image = read_image(train_path[0])
image = image.to(torch.float32) / 255.	# 读取单张图片，并进行归一化
image = image.unsqueeze(0)	# 为单张图片加入batch维度

plt.figure(figsize=(12,12))
mean = np.array([0.485, 0.456, 0.406]) # 图像增强所用的标准化参数
std = np.array([0.229, 0.224, 0.225])
for i in range(25):
    plt.subplot(5,5,i+1)
    img = transform(image)
    img = img[0,:,:,:].data.numpy().transpose([1,2,0]) # 将图片从[channel,height,width]变为[height,width,channel]
    img = std * img + mean # 反标准化
    img = np.clip(img, 0, 1) # 将像素值限制在0~1之间
    plt.imshow(img)
    plt.axis('off')
plt.show()

# 查看训练集一个batch的图片
for step, (bx, by) in enumerate(train_ds):
    if step > 0:
        break
    plt.figure(figsize=(16,16))
    for i in range(len(by)):
        plt.subplot(8,8,i+1)
        image = bx[i,:,:,:].data.numpy().transpose([1,2,0])
        image = std * image + mean # 反标准化
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('cat' if by[i].data.numpy()==0 else 'dog')
    plt.show()

model = models.resnet50(pretrained=True)  # 获取预训练的ResNet50架构
in_features = model.fc.in_features  # 获取顶层输出层的输入维度
model.fc = nn.Linear(in_features, 2)  # 将输出层替换为自己需要的输出层，输入维度为2
model.add_module('softmax', nn.Softmax(dim=-1)) # 添加softmax层

# 将除了顶层以为，其他层的参数冻结
for name, m in model.named_parameters():
    if name.split('.')[0] != 'fc':
        m.requires_grad_(False)

# 如果有GPU则使用GPU，否则使用CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = Adam(model.parameters(), lr=0.0001)
loss_fun = F.cross_entropy

def train(model, epoch, train_ds):
    model.train() # 开启训练模式
    total_num = len(train_ds.dataset) # 训练集总个数
    train_loss = 0	# 训练损失
    correct_num = 0 # 分类正确的个数

    for image, label in train_ds:
        image = image.to(device)	# 将图片和标签都迁移到GPU上进行加速
        label = label.to(device)
        label = label.to(torch.long) # 将标签从int32类型转化为long类型，否则计算损失会报错

        output = model(image) # 获取模型输出
        loss = loss_fun(output, label) # 计算交叉熵损失
        train_loss += loss.item() * label.size(0) # 累计总训练损失
        optimizer.zero_grad()	# 清除优化器上一次的梯度
        loss.backward()	# 进行反向传播
        optimizer.step()	# 更新优化器

        predict = torch.argmax(output, dim=-1)	# 获取预测类别
        correct_num += label.eq(predict).sum() # 统计预测正确的个数

    train_loss = train_loss / total_num	# 计算一个轮次的训练损失
    train_acc = correct_num / total_num	# 计算一个轮次的训练准确率
    # 打印相关信息
    print('epoch: {} --> train_loss: {:.6f} - train_acc: {:.6f} - '.format(
        epoch, train_loss, train_acc), end='')


def evaluate(model, eval_ds, mode='val'):
    model.eval()  # 开启测试模式

    total_num = len(eval_ds.dataset)
    eval_loss = 0
    correct_num = 0

    for image, label in eval_ds:
        image = image.to(device)
        label = label.to(device)
        label = label.to(torch.long)

        output = model(image)
        loss = loss_fun(output, label)
        eval_loss += loss.item() * label.size(0)

        predict = torch.argmax(output, dim=-1)
        correct_num += label.eq(predict).sum()

    eval_loss = eval_loss / total_num
    eval_acc = correct_num / total_num

    print('{}_loss: {:.6f} - {}_acc: {:.6f}'.format(
        mode, eval_loss, mode, eval_acc))

for epoch in range(20):
    train(model, epoch, train_ds)
    evaluate(model, val_ds)

torch.save(model,'model4.pkl')
