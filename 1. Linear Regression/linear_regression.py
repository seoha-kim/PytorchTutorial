import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt

num_data = 1000
num_epoch = 500

x = init.uniform_(torch.Tensor(num_data,1),-10,10) # -10부터 10까지 균등하게 초기화
noise = init.normal_(torch.FloatTensor(num_data, 1), std=1)  # 가우시안 노이즈 추가
y = 2*x + 3
y_noise = y + noise

plt.figure(figsize=(10, 10))
plt.scatter(x.numpy(), y_noise.numpy(), s=7, c='gray')
plt.axis([-12, 12, -25, 25])
plt.show(block=False)
plt.pause(0.05)
plt.close()

model = nn.Linear(1, 1)
loss_func = nn.L1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_arr = []
label = y_noise
for i in range(num_epoch):
    optimizer.zero_grad() # 기울기 0으로 초기
    output = model(x)

    loss = loss_func(output, label)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        plt.scatter(x.detach().numpy(), output.detach().numpy())
        plt.axis([-10, 10, -30, 30])
        plt.show(block=False)
        plt.pause(0.05)
        plt.close()
        print(loss.data)

    loss_arr.append(loss.detach().numpy())

plt.plot(loss_arr)
plt.show()

plt.figure(figsize=(10,10))
plt.scatter(x.numpy(),y_noise.numpy(),s=5,c="gray")
plt.scatter(x.detach().numpy(),output.detach().numpy(),s=5,c="red")
plt.axis([-10, 10, -30, 30])
plt.show()

param_list = list(model.parameters())
print('Weight:', param_list[0].item(),'\nBias' ,param_list[1].item())
