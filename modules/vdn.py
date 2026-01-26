import torch
import torch.nn as nn
import torch.nn.functional as F


class VDNAgentNetwork(nn.Module):
    """
    VDN智能体网络：每个智能体独立的DRQN网络
    
    与QMIX使用相同的Agent网络结构
    """
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(VDNAgentNetwork, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 观测编码器
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        
        # GRU循环层
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)
        
        # Q值输出层
        self.q_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, obs, hidden_state):
        """
        Args:
            obs: [batch_size, obs_dim] 当前观测
            hidden_state: [batch_size, hidden_dim] 隐藏状态
        Returns:
            q_values: [batch_size, action_dim] Q值
            next_hidden: [batch_size, hidden_dim] 下一隐藏状态
        """
        # 编码观测
        encoded_obs = F.relu(self.encoder(obs))
        
        # GRU处理
        next_hidden = self.gru(encoded_obs, hidden_state)
        
        # 计算Q值
        q_values = self.q_network(next_hidden)
        
        return q_values, next_hidden
