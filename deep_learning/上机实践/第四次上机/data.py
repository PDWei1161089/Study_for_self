{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c1c86f31-4523-4624-90d4-b66cbc8c62c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 该段代码用于leNet的训练数据集MNIST的载入\n",
    "\n",
    "from torchvision.datasets import MNIST\n",
    "import torchvision.transforms as transforms\n",
    "from torch.utils.data import DataLoader\n",
    "\n",
    "data_train = MNIST(\"./data\",\n",
    "                    download=True,\n",
    "                    transform=transforms.Compose([\n",
    "                        transforms.Resize((32, 32)),\n",
    "                        transforms.ToTensor()\n",
    "                    ]))\n",
    "data_test = MNIST(\"./data\",\n",
    "                    train=False,\n",
    "                    download=True,\n",
    "                    transform=transforms.Compose([\n",
    "                        transforms.Resize((32, 32)),\n",
    "                        transforms.ToTensor()\n",
    "                    ]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "e5419d60-8d85-43b6-b5bc-9fb10e1a3c77",
   "metadata": {},
   "outputs": [],
   "source": [
    "data_train_loader = DataLoader(data_train, batch_size=256,\n",
    "    shuffle=True, num_workers=2)\n",
    "data_test_loader = DataLoader(data_test, batch_size=1024,\n",
    "    num_workers=2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b005a9be-0a3e-4c6a-9876-33d451eaccd1",
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
