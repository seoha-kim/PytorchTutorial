import torch
import torch.nn as nn

# input_size : 입력의 특성 개수
# hidden_size : hidden state의 특성 개수
# num_layers : LSTM을 몇 층으로 쌓을 것인가 여부
# bias : 편차의 사용 여부
# batch_first : 사용하면 입력과 출력의 형태가 [batch, seq, feature]
# dropout : 드롭아웃 사용 여부
# bidirectional : 참고 http://solarisailab.com/archives/1515

rnn = nn.LSTM(input_size=3, hidden_size=5, num_layers=2)

# 입력의 형태 (seq_len, batch, input_size)
input_ = torch.randn(5, 3, 3)

# hidden state, cell state의 형태는 (num_layers*num_directions, batch, hidden_size)를 따릅니다.
h0 = torch.randn(2, 3, 5)
c0 = torch.randn(2, 3, 5)

# LSTM에 입력을 전달할 때는 input, (h_0, c_0)처럼 상태를 튜플로 묶어서 전달함
output, (hidden_state, cell_state) = rnn(input_, (h0, c0))
print(output.size(), hidden_state.size(), cell_state.size())

"""
하드코딩
"""

# batch_first를 한번 사용함
rnn = nn.LSTM(3, 5, 2, batch_first=True)

# batch_first를 사용하면 입력의 형태는 (batch, seq, feature)로 받게 됨
input_ = torch.randn(3, 5, 3)

# hidden stat,e cell state의 형태는 동일하게 (num_layers*num_directions, batch, hidden_size)를 따름
h0 = torch.randn(2, 3, 5)
c0 = torch.randn(2, 3, 5)

# LSTM에 입력을 전달할 때는 동일하게 input, (h_0, c_0)처럼 상태를 튜플로 묶어서 전달함
output, (hidden_state, cell_state) = rnn(input_, (h0, c0))
print(input_.size(), h0.size(), c0.size())
print(output.size(), hidden_state.size(), cell_state.size())

"""
with parameters
"""

input_size = 7
batch_size = 4
hidden_size = 3
seq_len = 2

rnn = nn.LSTM(input_size, hidden_size, seq_len, batch_first=True)

input_ = torch.randn(batch_size, hidden_size, input_size)
h0 = torch.randn(seq_len, batch_size, hidden_size)
c0 = torch.randn(seq_len, batch_size, hidden_size)

output, (hidden_state, cell_state) = rnn(input_, (h0, c0))

print(input_.size(), h0.size(), c0.size())
print(output.size(), hidden_state.size(), cell_state.size())