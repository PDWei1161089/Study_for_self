import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 加载MNIST数据集
transform = transforms.Compose([transforms.ToTensor()])
mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 创建一个新的数据集，将MNIST的标签更改为猫狗类别
new_data = []
new_labels = []
for i, (image, label) in enumerate(mnist_data):
    if label == 0:
        new_labels.append(0)  # 将数字0的标签设置为猫
    elif label == 1:
        new_labels.append(1)  # 将数字1的标签设置为狗
    else:
        new_data.append(image)
        new_labels.append(label)

# 将新的数据和标签转换为PyTorch张量
new_data = torch.stack(new_data)
new_labels = torch.tensor(new_labels)

# 创建一个新的数据加载器
new_data_loader = DataLoader(dataset=new_data, batch_size=64, shuffle=True)

# 定义模型
class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化模型
model = FCNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10):  # 遍历数据集多次
    for i, (images, labels) in enumerate(new_data_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/10], Step [{i + 1}/len(new_data_loader)], Loss: {loss.item():.4f}')

# 保存模型
torch.save(model.state_dict(), 'model.pth')

# 加载模型
model = FCNet()
model.load_state_dict(torch.load('model.pth'))

# 使用模型进行预测
images = new_data[:10]
labels = new_labels[:10]
outputs = model(images)
_, predicted = torch.max(outputs, 1)
print(predicted)
