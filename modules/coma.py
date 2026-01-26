import torch
import torch.nn as nn
import torch.nn.functional as F


class COMAActor(nn.Module):
    """COMA Actor网络：基于局部观测输出动作概率分布（离散动作空间）"""
    def __init__(self, obs_dim, hidden_dim, action_dim):
        super(COMAActor, self).__init__()
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


class COMACritic(nn.Module):
    """
    COMA Critic网络：基于全局状态和所有智能体的动作评估Q值
    
    COMA的关键特点：
    1. Critic接收全局状态和所有智能体的动作
    2. 输出每个智能体的Q值 Q(s, a_i, a_{-i})
    3. 用于计算反事实基线
    """
    def __init__(self, state_dim, obs_dim, action_dim, n_agents, hidden_dim):
        super(COMACritic, self).__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        
        # 输入维度：全局状态 + 所有智能体的动作（one-hot编码）
        # state_dim + n_agents * action_dim
        input_dim = state_dim + n_agents * action_dim
        
        # 状态和动作编码器
        self.encoder = nn.Linear(input_dim, hidden_dim)
        
        # 特征提取层
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Q值输出层（为每个智能体输出Q值）
        self.q_head = nn.Linear(hidden_dim, n_agents)
        
    def forward(self, state, actions):
        """
        计算每个智能体的Q值
        
        Args:
            state: [batch_size, state_dim] 全局状态
            actions: [batch_size, n_agents] 所有智能体的动作（整数索引）
        Returns:
            q_values: [batch_size, n_agents] 每个智能体的Q值
        """
        batch_size = state.shape[0]
        
        # 将动作转换为one-hot编码
        # actions: [batch_size, n_agents] -> [batch_size, n_agents, action_dim]
        actions_one_hot = F.one_hot(actions.long(), num_classes=self.action_dim).float()
        
        # 展平动作：[batch_size, n_agents, action_dim] -> [batch_size, n_agents * action_dim]
        actions_flat = actions_one_hot.view(batch_size, -1)
        
        # 拼接状态和动作
        input_features = torch.cat([state, actions_flat], dim=-1)  # [batch_size, state_dim + n_agents * action_dim]
        
        # 编码
        h1 = F.relu(self.encoder(input_features))
        h2 = F.relu(self.linear1(h1))
        h3 = F.relu(self.linear2(h2))
        
        # Q值输出
        q_values = self.q_head(h3)  # [batch_size, n_agents]
        
        return q_values
    
    def get_counterfactual_baseline(self, state, actions, agent_id, action_probs):
        """
        计算反事实基线：Q(s, a_{-i}, a_i^b)
        
        对于智能体i，基线是使用其策略的平均动作（加权平均）
        
        Args:
            state: [batch_size, state_dim] 全局状态
            actions: [batch_size, n_agents] 所有智能体的实际动作
            agent_id: int 要计算基线的智能体ID
            action_probs: [batch_size, action_dim] 智能体i的动作概率分布
        Returns:
            baseline: [batch_size] 反事实基线值
        """
        batch_size = state.shape[0]
        baseline = torch.zeros(batch_size, device=state.device)
        
        # 对每个可能的动作，计算加权Q值
        for action_idx in range(self.action_dim):
            # 创建反事实动作：其他智能体保持原动作，智能体i使用action_idx
            counterfactual_actions = actions.clone()
            counterfactual_actions[:, agent_id] = action_idx
            
            # 计算Q值
            q_values = self.forward(state, counterfactual_actions)  # [batch_size, n_agents]
            q_i = q_values[:, agent_id]  # [batch_size] 智能体i的Q值
            
            # 加权平均（使用动作概率作为权重）
            weight = action_probs[:, action_idx]  # [batch_size]
            baseline += weight * q_i
        
        return baseline
