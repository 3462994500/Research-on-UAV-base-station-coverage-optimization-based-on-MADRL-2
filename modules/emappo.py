import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.base import MultiHeadAttentionLayer


class GATLayer(nn.Module):
    """图注意力层：利用邻接矩阵进行信息聚合（简化版本）"""
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.head_dim = out_dim // num_heads
        
        # 为每个头创建独立的线性变换
        self.W_heads = nn.ModuleList([
            nn.Linear(in_dim, self.head_dim, bias=False) for _ in range(num_heads)
        ])
        
        # 注意力参数（每个头一个）
        self.a_heads = nn.ParameterList([
            nn.Parameter(torch.empty(size=(2 * self.head_dim, 1))) for _ in range(num_heads)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
        # 初始化参数
        for W in self.W_heads:
            nn.init.xavier_uniform_(W.weight)
        for a in self.a_heads:
            nn.init.xavier_uniform_(a)
        
    def forward(self, h, adj):
        """
        Args:
            h: [batch_size, n_agents, in_dim] 节点特征
            adj: [batch_size, n_agents, n_agents] 邻接矩阵
        Returns:
            out: [batch_size, n_agents, out_dim] 输出特征
        """
        batch_size, n_agents, _ = h.shape
        
        # 为每个头计算特征和注意力
        head_outputs = []
        for head in range(self.num_heads):
            # 线性变换
            Wh_head = self.W_heads[head](h)  # [batch_size, n_agents, head_dim]
            
            # 计算注意力分数
            a_input = self._prepare_attentional_mechanism_input(Wh_head)
            e = self.leaky_relu(torch.matmul(a_input, self.a_heads[head]).squeeze(-1))
            # e: [batch_size, n_agents, n_agents]
            
            # 应用邻接矩阵掩码
            attention = torch.where(adj > 0, e, torch.tensor(-9e15, device=e.device, dtype=e.dtype))
            attention = F.softmax(attention, dim=-1)
            attention = self.dropout(attention)
            
            # 聚合特征
            h_prime = torch.bmm(attention, Wh_head)  # [batch_size, n_agents, head_dim]
            head_outputs.append(h_prime)
        
        # 拼接所有头的输出
        out = torch.cat(head_outputs, dim=-1)  # [batch_size, n_agents, out_dim]
        return F.elu(out)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        """
        准备注意力机制输入
        Args:
            Wh: [batch_size, n_agents, head_dim]
        Returns:
            a_input: [batch_size, n_agents, n_agents, 2*head_dim]
        """
        batch_size, n_agents, head_dim = Wh.shape
        
        # 扩展维度进行拼接
        Wh1 = Wh.unsqueeze(2).expand(batch_size, n_agents, n_agents, head_dim)
        Wh2 = Wh.unsqueeze(1).expand(batch_size, n_agents, n_agents, head_dim)
        a_input = torch.cat([Wh1, Wh2], dim=-1)
        
        return a_input


class EnhancedMAPPOActor(nn.Module):
    """增强的MAPPO Actor网络：使用GAT处理邻接矩阵信息"""
    def __init__(self, obs_dim, hidden_dim, action_dim, num_heads=4, use_gat=True):
        super(EnhancedMAPPOActor, self).__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_gat = use_gat
        
        # 观测编码器
        self.encoder = nn.Linear(obs_dim, hidden_dim)
        
        if use_gat:
            # 使用GAT层处理图结构
            self.gat1 = GATLayer(hidden_dim, hidden_dim, num_heads)
            self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads)
            # 残差连接
            self.residual = nn.Linear(hidden_dim, hidden_dim)
        else:
            # 标准MLP
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 动作输出层
        self.action_head = nn.Linear(hidden_dim, action_dim)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, obs, adj=None):
        """
        Args:
            obs: [batch_size, n_agents, obs_dim] 或 [batch_size * n_agents, obs_dim] 局部观测
            adj: [batch_size, n_agents, n_agents] 邻接矩阵（可选）
        Returns:
            action_logits: [batch_size * n_agents, action_dim] 动作logits
            action_probs: [batch_size * n_agents, action_dim] 动作概率分布
        """
        # 处理输入维度
        if obs.ndim == 2:
            # [batch_size * n_agents, obs_dim]
            batch_size = obs.shape[0]
            n_agents = 1
            obs = obs.unsqueeze(1)  # [batch_size * n_agents, 1, obs_dim]
            use_gat = False  # 单个智能体时不需要GAT
        else:
            # [batch_size, n_agents, obs_dim]
            batch_size, n_agents, _ = obs.shape
            use_gat = self.use_gat and (adj is not None)
        
        # 编码观测
        h = F.relu(self.encoder(obs))  # [batch_size, n_agents, hidden_dim]
        
        if use_gat:
            # GAT处理
            h1 = self.gat1(h, adj)
            h1 = self.ln1(h1 + self.residual(h))  # 残差连接
            h2 = self.gat2(h1, adj)
            h2 = self.ln2(h2 + h1)  # 残差连接
            h = h2
        else:
            # 标准MLP
            h = F.relu(self.linear1(h))
            h = F.relu(self.linear2(h))
        
        # 重塑为 [batch_size * n_agents, hidden_dim]
        h = h.view(-1, self.hidden_dim)
        
        # 动作logits
        action_logits = self.action_head(h)
        
        # 动作概率分布
        action_probs = F.softmax(action_logits, dim=-1)
        
        return action_logits, action_probs
    
    def get_action_and_log_prob(self, obs, adj=None):
        """
        采样动作并计算log概率
        Args:
            obs: [batch_size * n_agents, obs_dim] 或 [batch_size, n_agents, obs_dim] 局部观测
            adj: [batch_size, n_agents, n_agents] 邻接矩阵（可选）
        Returns:
            action: [batch_size * n_agents] 采样的动作
            action_log_prob: [batch_size * n_agents] 动作的log概率
            action_probs: [batch_size * n_agents, action_dim] 动作概率分布
        """
        action_logits, action_probs = self.forward(obs, adj)
        
        # 创建分类分布并采样
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        
        return action, action_log_prob, action_probs
    
    def evaluate_actions(self, obs, actions, adj=None):
        """
        评估给定动作的log概率和熵
        Args:
            obs: [batch_size * n_agents, obs_dim] 或 [batch_size, n_agents, obs_dim] 局部观测
            actions: [batch_size * n_agents] 或 [batch_size, n_agents] 动作
            adj: [batch_size, n_agents, n_agents] 邻接矩阵（可选）
        Returns:
            action_log_prob: [batch_size * n_agents] 动作的log概率
            entropy: [batch_size * n_agents] 动作分布的熵
        """
        action_logits, action_probs = self.forward(obs, adj)
        dist = torch.distributions.Categorical(action_probs)

        # `Categorical.log_prob` expects `actions` to match batch_shape, which is
        # [batch_size * n_agents] here. Accept both [B, N] and [B*N].
        if actions.ndim == 2:
            actions = actions.reshape(-1)
        elif actions.ndim != 1:
            raise ValueError(f"Unexpected actions shape: {actions.shape}")

        action_log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return action_log_prob, entropy


class EnhancedMAPPOCritic(nn.Module):
    """增强的MAPPO Critic网络：使用GAT处理全局状态和邻接矩阵"""
    def __init__(self, state_dim, hidden_dim, num_heads=4, use_gat=True):
        super(EnhancedMAPPOCritic, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.use_gat = use_gat
        
        # 状态编码器
        self.encoder = nn.Linear(state_dim, hidden_dim)
        
        if use_gat:
            # 使用GAT层处理图结构（如果状态包含图信息）
            self.gat1 = GATLayer(hidden_dim, hidden_dim, num_heads)
            self.gat2 = GATLayer(hidden_dim, hidden_dim, num_heads)
            self.residual = nn.Linear(hidden_dim, hidden_dim)
        else:
            # 标准MLP
            self.linear1 = nn.Linear(hidden_dim, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 价值输出层
        self.value_head = nn.Linear(hidden_dim, 1)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
    def forward(self, state, adj=None):
        """
        Args:
            state: [batch_size, state_dim] 全局状态
            adj: [batch_size, n_agents, n_agents] 邻接矩阵（可选）
        Returns:
            value: [batch_size, 1] 状态价值
        """
        batch_size = state.shape[0]
        
        # 编码状态
        h = F.relu(self.encoder(state))  # [batch_size, hidden_dim]
        
        # 如果使用GAT且提供了邻接矩阵，需要将状态重塑为图结构
        if self.use_gat and adj is not None:
            # 假设状态可以重塑为 [batch_size, n_agents, hidden_dim]
            # 这里简化处理：将状态重复n_agents次
            n_agents = adj.shape[1]
            h = h.unsqueeze(1).expand(batch_size, n_agents, self.hidden_dim)
            
            h1 = self.gat1(h, adj)
            h1 = self.ln1(h1 + self.residual(h))
            h2 = self.gat2(h1, adj)
            h2 = self.ln2(h2 + h1)
            
            # 聚合所有智能体的特征（平均池化）
            h = h2.mean(dim=1)  # [batch_size, hidden_dim]
        else:
            # 标准MLP
            h = F.relu(self.linear1(h))
            h = F.relu(self.linear2(h))
        
        # 价值输出
        value = self.value_head(h)
        
        return value
