import torch
import torch.nn as nn
import torch.nn.functional as F


class MAPPOActor(nn.Module):
    """MAPPO Actor网络：基于局部观测输出动作概率分布（离散动作空间）"""
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(MAPPOActor, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        
        # 观测编码器
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        
        # 特征提取层
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 动作输出层
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs):
        """
        Args:
            obs: [batch_size, obs_dim] 局部观测
        Returns:
            action_logits: [batch_size, action_dim] 动作logits
            action_probs: [batch_size, action_dim] 动作概率分布
        """
        # 编码观测
        h1 = F.relu(self.encoder(obs))
        h2 = F.relu(self.linear1(h1))
        h3 = F.relu(self.linear2(h2))
        
        # 动作logits
        action_logits = self.action_head(h3)
        
        # 动作概率分布
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, action_probs
    
    def get_action_and_log_prob(self, obs):
        """
        采样动作并计算log概率
        Args:
            obs: [batch_size, obs_dim] 局部观测
        Returns:
            action: [batch_size] 采样的动作
            action_log_prob: [batch_size] 动作的log概率
            action_probs: [batch_size, action_dim] 动作概率分布
        """
        action_logits, action_probs = self.forward(obs)
        
        # 创建分类分布并采样
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, action_probs
    
    def evaluate_actions(self, obs, actions):
        """
        评估给定动作的log概率和熵
        Args:
            obs: [batch_size, obs_dim] 局部观测
            actions: [batch_size] 动作
        Returns:
            action_log_prob: [batch_size] 动作的log概率
            entropy: [batch_size] 动作分布的熵
        """
        action_logits, action_probs = self.forward(obs)
        dist = torch.distributions.Categorical(action_probs)
        action_log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_prob, entropy


class MAPPOCritic(nn.Module):
    """MAPPO Critic网络：基于全局状态评估状态价值"""
    def __init__(self, state_dim, hidden_dim):
        super(MAPPOCritic, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 状态编码器
        self.encoder = nn.Linear(state_dim, hidden_dim)
        
        # 特征提取层
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 价值输出层
        self.value_head = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] 全局状态
        Returns:
            value: [batch_size, 1] 状态价值
        """
        # 编码状态
        h1 = F.relu(self.encoder(state))
        h2 = F.relu(self.linear1(h1))
        h3 = F.relu(self.linear2(h2))
        
        # 价值输出
        value = self.value_head(h3)
        
        return value
