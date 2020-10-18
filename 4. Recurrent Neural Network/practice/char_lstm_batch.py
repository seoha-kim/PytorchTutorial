import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np



# preprocessing string data
# alphabet(0-25), space(26), start(27), end(28) -> 29 chars (0-28)
string = "hello pytorch. how long can a rnn cell remember? show me your limit!"
chars = "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
char_list = [i for i in chars]
char_len = len(char_list)
print(char_len)

def string_to_onehot(string):
    start = np.zeros(shape=char_len ,dtype=int)
    end = np.zeros(shape=char_len ,dtype=int)
    start[-2] = 1
    end[-1] = 1
    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=char_len ,dtype=int)
        zero[idx]=1
        start = np.vstack([start,zero])
    output = np.vstack([start,end])
    return output

def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]

batch_size = 5
seq_len = 1
num_layers = 3
input_size = char_len
hidden_size = 35
lr = 0.01
num_epochs = 1000

one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor())
print(one_hot.size())

# RNN with 1 hidden layer
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell

    def init_hidden_cell(self):
        hidden = torch.zeros(num_layers, batch_size, hidden_size)
        cell = torch.zeros(num_layers, batch_size, hidden_size)
        return hidden, cell


rnn = RNN(input_size, hidden_size, num_layers)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)

j=0
input_data = one_hot[j:j+batch_size].view(batch_size,seq_len,input_size)
print(input_data.size())

hidden,cell = rnn.init_hidden_cell()
print(hidden.size(),cell.size())

output,hidden,cell = rnn(input_data,hidden,cell)
print(output.size(),hidden.size(),cell.size())

unroll_len = one_hot.size()[0] // seq_len - 1
for i in range(num_epochs):
    optimizer.zero_grad()
    hidden, cell = rnn.init_hidden_cell()

    loss = 0
    for j in range(unroll_len - batch_size + 1):
        # batch size에 맞게 one-hot 벡터를 스택 합니다.
        # 예를 들어 batch size가 3이면 pytorch에서 pyt를 one-hot 벡터로 바꿔서 쌓고
        # 목표값으로 yto를 one-hot 벡터로 바꿔서 쌓는 과정입니다.
        input_data = torch.stack([one_hot[j + k:j + k + seq_len] for k in range(batch_size)], dim=0)
        label = torch.stack([one_hot[j + k + 1:j + k + seq_len + 1] for k in range(batch_size)], dim=0)

        input_data = input_data
        label = label

        output, hidden, cell = rnn(input_data, hidden, cell)
        loss += loss_func(output.view(1, -1), label.view(1, -1))

    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(loss)

hidden, cell = rnn.init_hidden_cell()
for j in range(unroll_len-batch_size+1):
    input_data = torch.stack([one_hot[j+k:j+k+seq_len] for k in range(batch_size)], dim=0)
    label = torch.stack([one_hot[j+k+1:j+k+seq_len+1] for k in range(batch_size)], dim=0)
    input_data = input_data
    label = label
    output, hidden, cell = rnn(input_data, hidden, cell)
    for k in range(batch_size):
        print(onehot_to_word(output[k].data), end="")
        if j < unroll_len-batch_size:
            break