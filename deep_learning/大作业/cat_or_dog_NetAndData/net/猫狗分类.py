# 1 加载必要的库
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import torch.utils.data
import matplotlib.pyplot as plt
from PIL import Image

# 2 定义超参数
BATCH_SIZE = 16  # 每批处理的数据个数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 是使用GPU还是使用CPU进行训练
EPOCHS = 10  # 训练数据集的轮数

# 3 构建pipeline, 对图像进行处理
# pipeline = transforms.Compose([
#     transforms.ToTensor(),  # 将图片转换成tensor
#     transforms.Normalize((0.1307,), (0.3081,))  # 正则化，用来降低模型复杂度
# ])

# 4 下载、加载数据
# 加载猫狗数据集
# 定义数据预处理操作
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 加载训练数据集
train_data = datasets.DatasetFolder(root=r'\data\train', transform=transform, loader=Image.open)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
# 加载测试数据集
test_data = datasets.DatasetFolder(root=r'\data\test', transform=transform, loader=Image.open)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=False)

# 5 训练模型
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


net = FCNet()

# 6 定义优化器
# 补充定义超参数
INPUT_SIZE = 784
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10
LEARNING_RATE = 0.01

model = net(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE).to(DEVICE)

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
            print("Epoch[{}/{}], Step[{}/{}], Loss: {:.4f}"
                  .format(epoch, EPOCHS, batch_index + 1, total_step, loss))


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
        print("Test -- Average loss : {:.4f}, Accuracy : {:.3f}\n".
              format(test_loss, 100.0 * correct / len(test_loader.dataset)))


# 9 调用方法 7 / 8
for epoch in range(1, EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)

sample_test = torch.utils.data.DataLoader(dataset=test_data,
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
