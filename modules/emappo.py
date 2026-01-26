import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base import MultiHeadAttentionLayer


class EnhancedMAPPOActor(nn.Module):
    """
    Enhanced MAPPO Actor网络：使用注意力机制和残差连接改进的Actor网络
    
    改进点：
    1. 使用自注意力机制处理多智能体信息
    2. 残差连接和层归一化提高训练稳定性
    3. 更深的网络结构增强表达能力
    4. 考虑邻接矩阵的图结构信息
    """
    def __init__(self, obs_dim, hidden_dim, action_dim, num_heads=4, use_attention=True):
        super(EnhancedMAPPOActor, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_heads = num_heads
        self.use_attention = use_attention
        
        # 观测编码器
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        if use_attention:
            # 自注意力层（用于处理多智能体信息）
            self.self_attention = MultiHeadAttentionLayer(
                hidden_dim, hidden_dim // num_heads, hidden_dim, num_heads
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
        
        # 特征提取层（带残差连接）
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim) for _ in range(2)
        ])
        
        # 动作输出层
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def _make_residual_block(self, dim):
        """创建残差块"""
        block = nn.ModuleDict({
            'linear1': nn.Linear(dim, dim),
            'norm1': nn.LayerNorm(dim),
            'linear2': nn.Linear(dim, dim),
            'norm2': nn.LayerNorm(dim)
        })
        return block
    
    def forward(self, obs, adj_mask=None):
        """
        Args:
            obs: [batch_size, obs_dim] 或 [batch_size, n_agents, obs_dim] 或 [n_agents, obs_dim] 局部观测
            adj_mask: [batch_size, n_agents, n_agents] 或 [n_agents, n_agents] 邻接矩阵掩码（可选）
        Returns:
            action_logits: [batch_size, action_dim] 或 [batch_size, n_agents, action_dim] 动作logits
            action_probs: [batch_size, action_dim] 或 [batch_size, n_agents, action_dim] 动作概率分布
        """
        # 处理输入维度
        squeeze_output = False
        if obs.ndim == 2:
            # 可能是 [batch_size, obs_dim] 或 [n_agents, obs_dim]
            # 检查adj_mask的维度来判断
            if adj_mask is not None and adj_mask.ndim == 2:
                # adj_mask是 [n_agents, n_agents]，所以obs是 [n_agents, obs_dim]
                # 转换为 [1, n_agents, obs_dim]
                obs = obs.unsqueeze(0)
                adj_mask = adj_mask.unsqueeze(0)  # [1, n_agents, n_agents]
                squeeze_output = True
            else:
                # [batch_size, obs_dim] -> [batch_size, 1, obs_dim]
                obs = obs.unsqueeze(1)
                squeeze_output = True
        
        batch_size, n_agents, _ = obs.shape
        
        # 编码观测
        h = self.encoder(obs)  # [batch_size, n_agents, hidden_dim]
        
        # 自注意力机制（如果启用）
        if self.use_attention and adj_mask is not None:
            # 确保adj_mask的batch_size与h匹配
            if adj_mask.shape[0] == 1 and batch_size > 1:
                adj_mask = adj_mask.repeat(batch_size, 1, 1)
            elif adj_mask.shape != (batch_size, n_agents, n_agents):
                # 如果维度不匹配，不使用注意力
                pass
            else:
                # 使用邻接矩阵作为注意力掩码
                h_att, _ = self.self_attention(h, adj_mask)
                h = self.attention_norm(h + h_att)  # 残差连接
        
        # 残差块
        for residual_block in self.residual_blocks:
            h_res = residual_block['linear1'](h)
            h_res = residual_block['norm1'](h_res)
            h_res = F.relu(h_res)
            h_res = residual_block['linear2'](h_res)
            h_res = residual_block['norm2'](h_res)
            h = F.relu(h + h_res)  # 残差连接
        
        # 动作输出
        action_logits = self.action_head(h)
        
        # 动作概率分布
        action_probs = F.softmax(action_logits, dim=-1)
        
        if squeeze_output:
            action_logits = action_logits.squeeze(1)
            action_probs = action_probs.squeeze(1)
        
        return action_logits, action_probs
    
    def get_action_and_log_prob(self, obs, adj_mask=None):
        """
        采样动作并计算log概率
        Args:
            obs: [batch_size, obs_dim] 或 [batch_size, n_agents, obs_dim] 局部观测
            adj_mask: [batch_size, n_agents, n_agents] 邻接矩阵掩码（可选）
        Returns:
            action: [batch_size] 或 [batch_size, n_agents] 采样的动作
            action_log_prob: [batch_size] 或 [batch_size, n_agents] 动作的log概率
            action_probs: [batch_size, action_dim] 或 [batch_size, n_agents, action_dim] 动作概率分布
        """
        action_logits, action_probs = self.forward(obs, adj_mask)
        
        # 处理维度
        if action_probs.ndim == 2:
            # [batch_size, action_dim]
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            action_log_prob = dist.log_prob(action)
        else:
            # [batch_size, n_agents, action_dim]
            batch_size, n_agents, action_dim = action_probs.shape
            action_probs_flat = action_probs.view(-1, action_dim)
            dist = torch.distributions.Categorical(action_probs_flat)
            action_flat = dist.sample()
            action_log_prob_flat = dist.log_prob(action_flat)
            action = action_flat.view(batch_size, n_agents)
            action_log_prob = action_log_prob_flat.view(batch_size, n_agents)
        
        return action, action_log_prob, action_probs
    
    def evaluate_actions(self, obs, actions, adj_mask=None):
        """
        评估给定动作的log概率和熵
        Args:
            obs: [batch_size, obs_dim] 或 [batch_size, n_agents, obs_dim] 局部观测
            actions: [batch_size] 或 [batch_size, n_agents] 动作
            adj_mask: [batch_size, n_agents, n_agents] 邻接矩阵掩码（可选）
        Returns:
            action_log_prob: [batch_size] 或 [batch_size, n_agents] 动作的log概率
            entropy: [batch_size] 或 [batch_size, n_agents] 动作分布的熵
        """
        action_logits, action_probs = self.forward(obs, adj_mask)
        
        # 处理维度
        if action_probs.ndim == 2:
            # [batch_size, action_dim]
            dist = torch.distributions.Categorical(action_probs)
            action_log_prob = dist.log_prob(actions)
            entropy = dist.entropy()
        else:
            # [batch_size, n_agents, action_dim]
            batch_size, n_agents, action_dim = action_probs.shape
            action_probs_flat = action_probs.view(-1, action_dim)
            actions_flat = actions.view(-1)
            dist = torch.distributions.Categorical(action_probs_flat)
            action_log_prob_flat = dist.log_prob(actions_flat)
            entropy_flat = dist.entropy()
            action_log_prob = action_log_prob_flat.view(batch_size, n_agents)
            entropy = entropy_flat.view(batch_size, n_agents)
        
        return action_log_prob, entropy


class EnhancedMAPPOCritic(nn.Module):
    """
    Enhanced MAPPO Critic网络：使用改进的全局状态编码
    
    改进点：
    1. 残差连接和层归一化提高训练稳定性
    2. 更深的网络结构增强表达能力
    3. 注意：Critic处理全局状态，不需要注意力机制（注意力只在Actor中使用）
    """
    def __init__(self, state_dim, hidden_dim, num_heads=4, use_attention=False):
        super(EnhancedMAPPOCritic, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.use_attention = use_attention
        
        # 状态编码器
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
        # 注意：Critic网络处理全局状态，不需要注意力机制
        # 注意力机制只在Actor网络中使用（处理多智能体局部观测）
        
        # 特征提取层（带残差连接）
        self.residual_blocks = nn.ModuleList([
            self._make_residual_block(hidden_dim) for _ in range(2)
        ])
        
        # 价值输出层
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def _make_residual_block(self, dim):
        """创建残差块"""
        block = nn.ModuleDict({
            'linear1': nn.Linear(dim, dim),
            'norm1': nn.LayerNorm(dim),
            'linear2': nn.Linear(dim, dim),
            'norm2': nn.LayerNorm(dim)
        })
        return block
    
    def forward(self, state, adj_mask=None):
        """
        Args:
            state: [batch_size, state_dim] 全局状态
            adj_mask: 保留参数以兼容接口，但Critic不使用注意力机制
        Returns:
            value: [batch_size, 1] 状态价值
        """
        # 编码状态
        h = self.encoder(state)  # [batch_size, hidden_dim]
        
        # Critic网络处理全局状态，不需要注意力机制
        # 注意力机制只在Actor网络中使用（处理多智能体局部观测）
        
        # 残差块
        for residual_block in self.residual_blocks:
            h_res = residual_block['linear1'](h)
            h_res = residual_block['norm1'](h_res)
            h_res = F.relu(h_res)
            h_res = residual_block['linear2'](h_res)
            h_res = residual_block['norm2'](h_res)
            h = F.relu(h + h_res)  # 残差连接
        
        # 价值输出
        value = self.value_head(h)
        
        return value
