import torch
import torch.nn as nn
import torch.nn.functional as F

class QMIXAgentNetwork(nn.Module):
    """QMIX智能体网络：每个智能体独立的DRQN网络"""
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(QMIXAgentNetwork, self).__init__()
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

class QMIXMixingNetwork(nn.Module):
    """QMIX混合网络：将个体Q值混合为全局Q值"""
    def __init__(self, n_agents, state_dim, hidden_dim):
        super(QMIXMixingNetwork, self).__init__()
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 超网络生成混合权重
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_agents * hidden_dim)
        )
        
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 超网络生成偏置
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, agent_qs, state):
        """
        Args:
            agent_qs: [batch_size, n_agents] 各个智能体的Q值
            state: [batch_size, state_dim] 全局状态
        Returns:
            total_q: [batch_size, 1] 全局Q值
        """
        batch_size = agent_qs.size(0)
        
        # 第一层混合
        w1 = torch.abs(self.hyper_w1(state))  # [batch_size, n_agents * hidden_dim]
        w1 = w1.view(batch_size, self.n_agents, self.hidden_dim)  # [batch_size, n_agents, hidden_dim]
        b1 = self.hyper_b1(state)  # [batch_size, hidden_dim]
        b1 = b1.unsqueeze(1).expand(-1, self.n_agents, -1)  # [batch_size, n_agents, hidden_dim]
        
        # 智能体Q值扩展维度
        agent_qs = agent_qs.unsqueeze(-1)  # [batch_size, n_agents, 1]
        
        # 第一层混合计算
        hidden = F.elu(torch.bmm(agent_qs.transpose(1, 2), w1).squeeze(1) + b1.mean(dim=1))  # [batch_size, hidden_dim]
        
        # 第二层混合
        w2 = torch.abs(self.hyper_w2(state))  # [batch_size, hidden_dim]
        b2 = self.hyper_b2(state)  # [batch_size, 1]
        
        # 最终Q值计算
        total_q = torch.bmm(hidden.unsqueeze(1), w2.unsqueeze(-1)).squeeze(-1) + b2  # [batch_size, 1]
        
        return total_q