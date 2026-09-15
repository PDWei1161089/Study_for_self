from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import torch

from MNIST_Net import MNIST_Net
import torch.nn as nn


class MNIST_Data(Dataset):
    def __init__(self, data_root):
        self.data_root = data_root

        self.dataset = []

        class_path = os.listdir(self.data_root)
        for c in class_path:
            img_names = os.listdir(os.path.join(data_root, c))
            for img_name in img_names:
                path = os.path.join(data_root, c, img_name)
                label = int(c)
                self.dataset.append((path, label))

    def __getitem__(self, idx):
        path, label = self.dataset[idx]
        img = cv2.imread(path, 0)
        img = img.reshape(-1)
        img = np.float32(img / 255)
        img = torch.from_numpy(img)
        return img, label

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    test_data_root = "MNIST_IMG/TEST"
    my_data = MNIST_Data(test_data_root)

    dataloader = DataLoader(dataset=my_data, batch_size=100, shuffle=False)

    score_sum = 0

    for img, label in dataloader:
        softmax = nn.Softmax(dim=1)
        score = softmax(score)
        # print("预测概率：",score)
        answer = torch.argmax(score, dim=1)
        # print("输入图像的类别序号为：", answer)
        # print(img.shape, label)
        # print("hello")

        score_batch = torch.eq(answer, label).float().sum()
        score_all = score_all + score_batch

    score_all = score_all / len(my_data) * 100
    print(f"Test score is:{score_all:2f}%")
