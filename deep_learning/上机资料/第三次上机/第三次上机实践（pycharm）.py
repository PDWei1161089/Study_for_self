import torch
from torchvision import datasets, transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # 图片格式是灰度图只有一个channel,将其变成RGB的3个channel
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
     ]
)

data_train = datasets.MNIST(root="data/", transform=transform, train=True, download=True)

data_test = datasets.MNIST(root="data/", transform=transform, train=False)

import torch
from torchvision import datasets, transforms

data_train = datasets.MNIST(root="data/", transform=transform, train=True, download=True)

data_test = datasets.MNIST(root="data/", transform=transform, train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                shuffle=True,
                                                batch_size=64)
data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               shuffle=True,
                                               batch_size=64)

import torchvision
import matplotlib.pyplot as plt

# %matplotlib inline

images, labels = next(iter(data_loader_train))
img = torchvision.utils.make_grid(images)

img = img.numpy().transpose(1, 2, 0)
std = [0.5, 0.5, 0.5]
mean = [0.5, 0.5, 0.5]
img = img * std + mean
print([labels[i] for i in range(64)])
plt.imshow(img)

# 1 加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# 2 定义超参数
BATCH_SIZE = 16  # 每批处理的数据个数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是使用GPU还是使用CPU进行训练
EPOCHS = 10  # 训练数据集的轮数

# 3 构建pipeline, 对图像进行处理
pipeline = transforms.Compose([
    transforms.ToTensor(),  # 将图片转换成tensor
    transforms.Normalize((0.1307,), (0.3081,))  # 正则化，用来降低模型复杂度
])

# 4 下载、加载数据
from torch.utils.data import DataLoader

# 下载数据集
data_train = datasets.MNIST(root="data/", transform=pipeline, train=True, download=True)

data_test = datasets.MNIST(root="data/", transform=pipeline, train=False, download=True)

# 加载数据
train_loader = DataLoader(data_train, batch_size=BATCH_SIZE, shuffle=True)

test_loader = DataLoader(data_test, batch_size=BATCH_SIZE, shuffle=True)


# 5 构建网络模型
class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 6 定义优化器

# 补充定义超参数
INPUT_SIZE = 784
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10
LEARNING_RATE = 0.01

model = FCNet(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 7 定义训练方法
total_step = len(train_loader)


def train_model(model, device, train_loader, optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, label) in enumerate(train_loader):
        # 部署到DEVICE上
        data, label = data.reshape(-1, 28 * 28).to(DEVICE), label.to(DEVICE)
        # 将模型参数的梯度初始化为0
        optimizer.zero_grad()
        # 前向传播计算预测值
        output = model(data)
        # 计算当前损失
        loss = F.cross_entropy(output, label)
        # 反向传播计算预测值
        loss.backward()
        # 更新所有参数
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Epoch[{}/{}], Step[{}/{}], Loss: {:.4f}".format(epoch, EPOCHS, batch_index + 1, total_step, loss))


# 8 定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():  # 在测试阶段，不用计算梯度，也不用进行反向传播
        for data, label in test_loader:
            # 部署到DEVICE上
            data, label = data.reshape(-1, 28 * 28).to(device), label.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, label).item()
            # 找到概率最大值的下标
            _, predict = torch.max(output, dim=1)  # 这里返回最大值和最大值索引
            # 累计正确的值
            correct += predict.eq(label.view_as(predict)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test -- Average loss : {:.4f}, Accuracy : {:.3f}\n".format(test_loss,
                                                                          100.0 * correct / len(test_loader.dataset)))


# 9 调用方法 7 / 8
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)

sample_test = torch.utils.data.DataLoader(dataset=data_test,
                                          shuffle=True,
                                          batch_size=4)
X_test, Y_test = next(iter(sample_test))
inputs = X_test.reshape(-1, 28 * 28).to(DEVICE)
pred = model(inputs)
_, pred = torch.max(pred, 1)

print("Predict lable is :", [i for i in pred.data])
print("Real label is :", [i for i in Y_test])

img = torchvision.utils.make_grid(X_test)
img = img.numpy().transpose(1, 2, 0)

# std = [0.1307, 0.1307, 0.1307]
# mean = [0.3081, 0.3081, 0.3081]
std = (0.1307,)
mean = (0.3081,)
img = img * std + mean
plt.imshow(img)
