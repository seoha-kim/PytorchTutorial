import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

N_HIDDEN = 35
LR = 0.01
EPOCHS = 1000

string = "hello pytorch. how long can a rnn cell remember? show me your limit!"
chars =  "abcdefghijklmnopqrstuvwxyz ?!.,:;01"
char_list = [i for i in chars]
n_letters = len(char_list)

def string_to_onehot(string):
    start = np.zeros(shape=n_letters, dtype=int)
    end = np.zeros(shape=n_letters, dtype=int)
    start[-2] = 1
    end[-1] = 1

    for i in string:
        idx = char_list.index(i)
        zero = np.zeros(shape=n_letters, dtype=int)
        zero[idx] = 1
        start = np.vstack([start, zero])
    output = np.vstack([start, end])
    return output

def onehot_to_word(onehot_1):
    onehot = torch.Tensor.numpy(onehot_1)
    return char_list[onehot.argmax()]

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size+hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size+hidden_size, output_size)
        self.act_fn = nn.Tanh()

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.act_fn(self.i2h(combined))
        output = self.i2o(combined)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.hidden_size)

rnn = RNN(n_letters, N_HIDDEN, n_letters)

loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
one_hot = torch.from_numpy(string_to_onehot(string)).type_as(torch.FloatTensor())

"""
학습
"""

for i in range(EPOCHS):
    optimizer.zero_grad()
    hidden = rnn.init_hidden()
    total_loss = 0
    for j in range(one_hot.size()[0]-1):
        # 입력값은 앞의 글자
        # pytorch as pytorc
        input_ = one_hot[j:j+1, :]
        # 목표값은 뒤에 글자
        # pytorch as ytorch
        target = one_hot[j+1]
        output, hidden = rnn.forward(input_, hidden)
        loss = loss_func(output.view(-1), target.view(-1))
        total_loss += loss

    total_loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(total_loss)

"""
테스트
"""

start = torch.zeros(1,n_letters)
start[:,-2] = 1

with torch.no_grad():
    hidden = rnn.init_hidden()
    input_ = start
    output_string = ""

    for i in range(len(string)):
        output, hidden = rnn.forward(input_, hidden)
        output_string += onehot_to_word(output.data)
        input_ = output

print(output_string)