import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

batch_size = 256
learning_rate = 0.0002
num_epoch = 10

mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
print(mnist_train.__getitem__(0)[0].size(), mnist_train.__len__())
print(mnist_test.__getitem__(0)[0].size(), mnist_test.__len__())

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=2,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=2,drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(1,16,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(64*7*7,100),
            nn.ReLU(),
            nn.Linear(100,10)
        )

        # weight 초기화
        # 모델의 모듈을 차례대로 불러옴
        for m in self.modules():
            # 모듈이 nn.Conv2d인 경우
            if isinstance(m, nn.Conv2d):
                """
                # 작은 숫자 초기화
                # 가중치를 평균 0, 편차 0.02로 초기화
                # 편차를 0으로 초기화
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

                # Xavier Initialization
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
                """

                # Kaiming Initialization
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

            # 모듈이 nn.Linear인 경우
            elif isinstance(m, nn.Linear):
                """
                                # 작은 숫자 초기화
                # 가중치를 평균 0, 편차 0.02로 초기화
                # 편차를 0으로 초기화
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)

                # Xavier Initialization
                init.xavier_normal(m.weight.data)
                m.bias.data.fill_(0)
                """

                # Kaiming Initialization
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)


    def forward(self, x):
        out = self.layer(x)
        out = out.view(batch_size, -1)
        out = self.fc_layer(out)
        return out

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = CNN().to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for i in range(num_epoch):
    for j, [image, label] in enumerate(train_loader):
        x = image.to(device)
        y_ = label.to(device)

        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output, y_)
        loss.backward()
        optimizer.step()

    if i % 10 == 0:
        print(loss)

param_list = list(model.parameters())
print(param_list)

correct = 0
total = 0

with torch.no_grad():
  for image,label in test_loader:
      x = image.to(device)
      y_= label.to(device)

      output = model.forward(x)
      _,output_index = torch.max(output,1)

      total += label.size(0)
      correct += (output_index == y_).sum().float()

  print("Accuracy of Test Data: {}".format(100*correct/total))