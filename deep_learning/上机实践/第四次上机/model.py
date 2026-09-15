{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3e3ec746-50fe-4267-9e8e-e60cc98eb803",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 该代码定义了LeNet模型\n",
    "\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "class LeNet(nn.Module):\n",
    "\n",
    "    def __init__(self):\n",
    "        super(LeNet, self).__init__()\n",
    "        self.conv1 = nn.Conv2d(1, 6, 3)\n",
    "        self.pool1 = nn.MaxPool2d(2, 2)\n",
    "        self.conv2 = nn.Conv2d(6, 16, 3)\n",
    "        self.pool2 = nn.MaxPool2d(2, 2)\n",
    "        self.fc3 = nn.Linear(16*6*6, 120)\n",
    "        self.fc4 = nn.Linear(120, 84)\n",
    "        self.fc5 = nn.Linear(84, 10)\n",
    "\n",
    "    def forward(self, x):\n",
    "        x = self.pool1(torch.relu(self.conv1(x)))\n",
    "        x = self.pool2(torch.relu(self.conv2(x)))\n",
    "        x = x.view(x.size(0), -1)\n",
    "        x = torch.relu(self.fc3(x))\n",
    "        x = torch.relu(self.fc4(x))\n",
    "        x = self.fc5(x)\n",
    "        return x"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "c7f6484e-2a45-4892-adaa-43d98c3316e7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "torch.Size([1, 10])"
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = LeNet() # 初始化实例\n",
    "ret = model(torch.randn(1, 1, 32, 32)) #输入一张图片， 测试输出结果\n",
    "ret.shape # torch.Size([1, 10]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5fb592ee-577a-49e8-91bb-f0db08873caa",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.13"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
