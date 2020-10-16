import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

"""
하이퍼 파라미터 지정
"""
BATCH_SIZE = 256
LEARNING_RATE = 0.0002
NUM_EPOCH = 10

"""
데이터 다운로드, 체크, DataLoader 설정
"""

mnist_train = dset.MNIST(root="./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST(root="./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
# transforms.ToTensor을 경우 PIL 이미지나 Numpy 배열을 토치 텐서로 바꿔줄지의 여부

print(mnist_train.__getitem__(0)[0].size(), mnist_train.__len__())
print(mnist_test.__getitem__(0)[0].size(), mnist_test.__len__())

train_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, drop_last=True)
test_loader = DataLoader(mnist_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=True)

"""
CNN 모델, 손실함수, 최적화 함
"""

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*3*3, 100),
            nn.ReLU(),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        out = self.layer(x)
        out = out.view(BATCH_SIZE, -1)
        out = self.fc_layer(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

"""
학습
"""
loss_arr = []
for i in range(NUM_EPOCH):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()
        if j % 1000 == 0:
            print(loss)
            loss_arr.append(loss.cpu().detach().numpy())

plt.plot(loss_arr)
plt.show()

"""
테스트 데이터 정확도 측
"""

correct = 0
total = 0

with torch.no_grad():
    for image, label in test_loader:
        x = image.to(device)
        y_ = label.to(device)
        output = model.forward(x)

        _, output_index = torch.max(output, 1)
        total += label.size(0)
        correct += (output_index == y_).sum().float()

    print("Accuracy of Test Data : {}%".foramt(100*correct/total))