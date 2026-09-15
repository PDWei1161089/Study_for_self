import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# 加载自己的数据集
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root, train, download):
        self.data = ...  # 你的数据集，应该是一个列表或张量，包含数据和标签
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.data[index]
        if self.train:
            transform = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip()])
        else:
            transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        return image, label

train_dataset = MyDataset(root='./data', train=True, download=True)
test_dataset = MyDataset(root='./data', train=False, download=True)

# 创建数据加载器
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

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
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/10], Step [{i + 1}/len(train_loader)], Loss: {loss.item():.4f}')

# 在测试集上评估模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the model on the test set: {100 * correct / total}%')
