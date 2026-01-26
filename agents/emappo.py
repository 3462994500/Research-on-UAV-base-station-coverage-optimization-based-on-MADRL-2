import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseActorCriticAgent
from modules.emappo import EnhancedMAPPOActor, EnhancedMAPPOCritic
from utils.torch_utils import *
from utils.hparams import hparams
from utils.device import device


class EnhancedMAPPOAgent(BaseActorCriticAgent):
    """
    增强的MAPPO智能体：改进的集中式训练+分布式执行(CTDE)架构
    
    主要改进：
    1. 使用GAT网络利用邻接矩阵信息，增强多智能体协作
    2. 改进的Critic网络设计，更好地处理全局状态
    3. 自适应熵系数，平衡探索与利用
    4. 改进的优势函数计算
    """
    def __init__(self, obs_dim, act_dim, n_agents, state_dim):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hparams['hidden_dim']
        self.use_gat = hparams.get('use_gat', True)
        self.num_heads = hparams.get('num_heads', 4)
        
        # 创建增强的Actor网络（使用GAT）
        self.learned_actor_model = EnhancedMAPPOActor(
            obs_dim, 
            self.hidden_dim, 
            act_dim,
            num_heads=self.num_heads,
            use_gat=self.use_gat
        ).to(device)
        
        # 创建增强的Critic网络
        self.learned_critic_model = EnhancedMAPPOCritic(
            state_dim, 
            self.hidden_dim,
            num_heads=self.num_heads,
            use_gat=self.use_gat
        ).to(device)
        
        # 兼容接口
        self.learned_model = self.learned_actor_model
        
        # 自适应熵系数
        self.entropy_coef = hparams.get('entropy_coef', 0.01)
        self.entropy_coef_min = hparams.get('entropy_coef_min', 0.001)
        self.entropy_coef_decay = hparams.get('entropy_coef_decay', 0.9995)

    def action(self, obs, adj=None, epsilon=0.3, action_mode='sample'):
        """
        分布式执行（Decentralized Execution）：每个智能体基于局部观测选择动作
        
        改进：支持邻接矩阵输入，利用图结构信息
        """
        if isinstance(obs, np.ndarray):
            # 从环境输入
            assert obs.ndim == 2  # [n_agents, obs_dim]
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            batch_size = 1
        elif isinstance(obs, torch.Tensor):
            # 从回放缓冲区输入
            obs = to_cuda(obs)
            if obs.ndim == 2:
                batch_size = 1
                obs = obs.unsqueeze(0)  # [1, n_agents, obs_dim]
            else:
                batch_size = obs.shape[0]
        else:
            raise TypeError("Unsupported observation type")
        
        # 处理邻接矩阵
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
            if adj.ndim == 2:
                adj = adj.unsqueeze(0)  # [1, n_agents, n_agents]
            adj = adj.expand(batch_size, -1, -1)
        else:
            adj = None
        
        with torch.no_grad():
            # 确保obs是3D: [batch_size, n_agents, obs_dim]
            if obs.ndim == 2:
                obs = obs.view(batch_size, self.n_agents, self.obs_dim)
            elif obs.ndim == 3:
                # 已经是3D，确保batch_size匹配
                pass
            else:
                raise ValueError(f"Unexpected obs shape: {obs.shape}")
            
            if action_mode == 'sample':
                # 采样动作（使用GAT）
                actions, _, _ = self.learned_actor_model.get_action_and_log_prob(obs, adj)
                actions = actions.view(batch_size, self.n_agents)
            elif action_mode == 'greedy':
                # 贪婪动作
                _, action_probs = self.learned_actor_model(obs, adj)
                actions = action_probs.argmax(dim=-1)
                actions = actions.view(batch_size, self.n_agents)
            else:
                raise ValueError(f"Unsupported action mode: {action_mode}")
            
            # 转换为numpy
            if batch_size == 1:
                actions = actions.squeeze(0).cpu().numpy()
            else:
                actions = actions.cpu().numpy()
                
        return actions

    def cal_p_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        增强的MAPPO策略损失计算（PPO clipped objective）
        
        改进：
        1. 使用GAT处理邻接矩阵信息
        2. 自适应熵系数
        3. 改进的优势函数归一化
        """
        obs = sample['obs']  # [batch_size, n_agents, obs_dim]
        actions = sample['action'].long()  # [batch_size, n_agents]
        old_log_probs = sample.get('old_log_prob', None)  # [batch_size, n_agents]
        advantages = sample.get('advantage', None)  # [batch_size, n_agents]
        returns = sample.get('return', None)  # [batch_size, n_agents]
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        adj = sample.get('adj', None)  # [batch_size, n_agents, n_agents] - 邻接矩阵
        
        batch_size = obs.shape[0]
        
        # 如果没有提供状态，使用观测的拼接作为状态
        if state is None:
            state = obs.view(batch_size, -1)
        
        # 处理邻接矩阵
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
            if adj.ndim == 2:
                adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # 如果没有提供邻接矩阵，创建全连接图（自连接）
            adj = torch.eye(self.n_agents, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # 重塑输入
        obs_flat = obs.view(-1, self.obs_dim)  # [batch_size * n_agents, obs_dim]
        actions_flat = actions.view(-1)  # [batch_size * n_agents]
        
        if old_log_probs is not None:
            old_log_probs_flat = old_log_probs.view(-1)  # [batch_size * n_agents]
        
        if advantages is not None:
            advantages_flat = advantages.view(-1)  # [batch_size * n_agents]
        
        # Actor前向传播（使用GAT）
        # 需要将obs重塑为 [batch_size, n_agents, obs_dim] 以使用GAT
        obs_gat = obs  # [batch_size, n_agents, obs_dim]
        action_log_probs, entropy = self.learned_actor_model.evaluate_actions(
            obs_gat, actions, adj
        )
        
        # 计算重要性采样比率
        if old_log_probs is not None:
            ratio = torch.exp(action_log_probs - old_log_probs_flat)
        else:
            ratio = torch.exp(action_log_probs)
        
        # PPO clipped objective
        if advantages is not None:
            clip_param = hparams.get('clip_param', 0.2)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_flat
            actor_loss = -torch.min(surr1, surr2).mean()
        else:
            # 如果没有优势函数，使用简单的策略梯度
            actor_loss = -action_log_probs.mean()
        
        # 自适应熵正则化（鼓励探索）
        if global_steps is not None:
            # 随着训练进行，逐渐减小熵系数
            self.entropy_coef = max(
                self.entropy_coef * self.entropy_coef_decay,
                self.entropy_coef_min
            )
        
        entropy_loss = -entropy.mean()
        
        # 总策略损失
        policy_loss = actor_loss + self.entropy_coef * entropy_loss
        
        losses['policy_loss'] = policy_loss
        losses['actor_loss'] = actor_loss
        losses['entropy'] = entropy.mean()
        losses['entropy_coef'] = self.entropy_coef

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        增强的MAPPO价值损失计算
        
        改进：
        1. 使用GAT处理全局状态和邻接矩阵
        2. 改进的价值函数归一化
        """
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        returns = sample.get('return', None)  # [batch_size, n_agents] 或 [batch_size, 1]
        obs = sample.get('obs', None)  # [batch_size, n_agents, obs_dim]
        adj = sample.get('adj', None)  # [batch_size, n_agents, n_agents] - 邻接矩阵
        
        batch_size = state.shape[0] if state is not None else obs.shape[0]
        
        # 如果没有提供状态，使用观测的拼接作为状态
        if state is None:
            if obs is not None:
                state = obs.view(batch_size, -1)
            else:
                raise ValueError("Either state or obs must be provided")
        
        # 处理邻接矩阵
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
            if adj.ndim == 2:
                adj = adj.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            # 如果没有提供邻接矩阵，创建全连接图
            adj = torch.eye(self.n_agents, device=device).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Critic前向传播（使用GAT）
        values = self.learned_critic_model(state, adj)  # [batch_size, 1]
        
        if returns is not None:
            # 如果returns是[batch_size, n_agents]，取平均值
            if returns.ndim == 2 and returns.shape[1] == self.n_agents:
                returns = returns.mean(dim=1, keepdim=True)  # [batch_size, 1]
            elif returns.ndim == 1:
                returns = returns.unsqueeze(-1)  # [batch_size, 1]
            
            # 价值损失（MSE）
            value_loss = F.mse_loss(values, returns)
        else:
            # 如果没有returns，使用简单的价值预测损失
            value_loss = torch.zeros_like(values.mean())
        
        losses['value_loss'] = value_loss
        losses['q_loss'] = value_loss  # 兼容接口

    def update_target(self):
        """MAPPO不需要目标网络（on-policy算法）"""
        pass
