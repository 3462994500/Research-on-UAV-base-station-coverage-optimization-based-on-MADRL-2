import torch
import torch.nn as nn
import torch.nn.functional as F

class DRQNNetwork(nn.Module):
    def __init__(self, in_dim, hidden_dim, action_dim):
        super(DRQNNetwork, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim

        # Encoder layer to process the input observation
        self.encoder = nn.Linear(in_dim, hidden_dim)

        # Recurrent layer (GRU) to handle temporal dependencies
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)

        # Two hidden layers after the GRU
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, action_dim)


    def forward(self, x, hidden_state):
        # x: [ batch_size,n_ant, input_dim]
        # hidden: [1, batch_size, hidden_dim]
        bs, n_agent, _ = x.shape

        # Encode the input observation
        h0 = F.relu(self.encoder(x))

        # Pass through the GRU
        h0 = h0.reshape([bs * n_agent, -1])
        hidden_state = hidden_state.reshape([bs * n_agent, -1])
        rnn_out = self.rnn(h0, hidden_state).reshape([bs, n_agent, -1])
        next_hidden_state = rnn_out

        # Pass through additional linear layers
        h1 = F.relu(self.linear1(rnn_out))
        h2 = F.relu(self.linear2(h1))
        qs = self.linear3(h2)

        # Reshape the output to match the original input shape
        next_hidden_state = next_hidden_state.reshape([bs, n_agent, -1])
        qs = qs.reshape([bs, n_agent, self.action_dim])

        return qs, next_hidden_state
# # 创建 DRQNNetwork 实例
# in_dim = 10
# hidden_dim = 20
# action_dim = 5
# model = DRQNNetwork(in_dim, hidden_dim, action_dim)
#
# # 打印网络结构
# print(model)