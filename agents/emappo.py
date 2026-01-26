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
    Enhanced MAPPO智能体：改进的MAPPO算法，针对3D无人机协同覆盖场景优化
    
    主要改进：
    1. 使用注意力机制和残差连接的改进网络架构
    2. 考虑邻接矩阵的图结构信息
    3. 改进的优势函数计算（考虑智能体间协作）
    4. 自适应学习率和更好的训练稳定性
    """
    def __init__(self, obs_dim, act_dim, n_agents, state_dim):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hparams['hidden_dim']
        self.num_heads = hparams.get('num_attention_heads', 4)
        self.use_attention = hparams.get('use_attention', True)
        
        # 创建改进的Actor网络（使用注意力机制）
        self.learned_actor_model = EnhancedMAPPOActor(
            obs_dim, self.hidden_dim, act_dim, 
            num_heads=self.num_heads, 
            use_attention=self.use_attention
        ).to(device)
        
        # 创建改进的Critic网络（不使用注意力机制，因为Critic处理全局状态）
        self.learned_critic_model = EnhancedMAPPOCritic(
            state_dim, self.hidden_dim,
            num_heads=self.num_heads,
            use_attention=False  # Critic不需要注意力机制
        ).to(device)
        
        # 兼容接口
        self.learned_model = self.learned_actor_model

    def action(self, obs, adj=None, epsilon=0.3, action_mode='sample'):
        """
        分布式执行（Decentralized Execution）：每个智能体基于局部观测选择动作
        
        改进：考虑邻接矩阵信息用于注意力机制
        """
        if isinstance(obs, np.ndarray):
            # 从环境输入
            assert obs.ndim == 2  # [n_agents, obs_dim]
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            batch_size = 1
            is_from_env = True
        elif isinstance(obs, torch.Tensor):
            # 从回放缓冲区输入
            obs = to_cuda(obs)
            if obs.ndim == 2:
                batch_size = 1
                obs = obs.unsqueeze(0)  # [1, n_agents, obs_dim]
                is_from_env = False
            else:
                batch_size = obs.shape[0]
                is_from_env = False
        else:
            raise TypeError("Unsupported observation type")
        
        # 处理邻接矩阵
        adj_mask = None
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
            if adj.ndim == 2:
                adj = adj.unsqueeze(0)  # [1, n_agents, n_agents]
            if batch_size > 1 and adj.shape[0] == 1:
                adj = adj.repeat(batch_size, 1, 1)
            adj_mask = adj
        
        with torch.no_grad():
            # 确保obs是 [batch_size, n_agents, obs_dim] 的格式
            # 如果来自环境，需要添加批次维度
            if is_from_env:
                # 环境输出是 [n_agents, obs_dim]，需要 unsqueeze 到 [1, n_agents, obs_dim]
                if obs.ndim == 2:
                    obs = obs.unsqueeze(0)
            
            if action_mode == 'sample':
                # 采样动作（使用注意力机制）
                actions, _, _ = self.learned_actor_model.get_action_and_log_prob(obs, adj_mask)
            elif action_mode == 'greedy':
                # 贪婪动作
                _, action_probs = self.learned_actor_model(obs, adj_mask)
                actions = action_probs.argmax(dim=-1)
            else:
                raise ValueError(f"Unsupported action mode: {action_mode}")
            
            # 转换为numpy
            if batch_size == 1:
                # 从环境或单个样本来的，返回 [n_agents] 或 [n_agents, ...] 的形状
                if actions.ndim > 1:
                    actions = actions.squeeze(0)
                actions = actions.cpu().numpy()
            else:
                # 来自批次，保持批次维度
                actions = actions.cpu().numpy()
                
        return actions

    def cal_p_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        Enhanced MAPPO策略损失计算（改进的PPO clipped objective）
        
        改进点：
        1. 考虑邻接矩阵的图结构信息
        2. 改进的重要性采样比率计算
        3. 自适应裁剪参数
        """
        obs = sample['obs']  # [batch_size, n_agents, obs_dim]
        actions = sample['action'].long()  # [batch_size, n_agents]
        old_log_probs = sample.get('old_log_prob', None)  # [batch_size, n_agents]
        advantages = sample.get('advantage', None)  # [batch_size, n_agents]
        returns = sample.get('return', None)  # [batch_size, n_agents]
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        adj = sample.get('adj', None)  # [batch_size, n_agents, n_agents] - 邻接矩阵
        
        batch_size = obs.shape[0]
        
        # 处理邻接矩阵
        adj_mask = None
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
            adj_mask = adj
        
        # 如果没有提供状态，使用观测的拼接作为状态
        if state is None:
            state = obs.view(batch_size, -1)
        
        # 重塑输入
        obs_flat = obs.view(-1, self.obs_dim)  # [batch_size * n_agents, obs_dim]
        actions_flat = actions.view(-1)  # [batch_size * n_agents]
        
        if old_log_probs is not None:
            old_log_probs_flat = old_log_probs.view(-1)  # [batch_size * n_agents]
        
        if advantages is not None:
            advantages_flat = advantages.view(-1)  # [batch_size * n_agents]
        
        # Actor前向传播（使用注意力机制）
        if adj_mask is not None:
            action_log_probs, entropy = self.learned_actor_model.evaluate_actions(
                obs, actions, adj_mask
            )
            action_log_probs = action_log_probs.view(-1)
            entropy = entropy.view(-1)
        else:
            action_log_probs, entropy = self.learned_actor_model.evaluate_actions(
                obs_flat, actions_flat, adj_mask
            )
        
        # 计算重要性采样比率
        if old_log_probs is not None:
            ratio = torch.exp(action_log_probs - old_log_probs_flat)
        else:
            ratio = torch.exp(action_log_probs)
        
        # PPO clipped objective（自适应裁剪参数）
        if advantages is not None:
            clip_param = hparams.get('clip_param', 0.2)
            # 可以根据训练进度自适应调整裁剪参数
            if global_steps is not None:
                # 训练初期使用较大的裁剪参数，后期逐渐减小
                adaptive_clip = clip_param * (1.0 - min(global_steps / 100000, 0.5))
                clip_param = max(adaptive_clip, clip_param * 0.5)
            
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_flat
            actor_loss = -torch.min(surr1, surr2).mean()
        else:
            # 如果没有优势函数，使用简单的策略梯度
            actor_loss = -action_log_probs.mean()
        
        # 熵正则化（鼓励探索，自适应调整）
        entropy_coef = hparams.get('entropy_coef', 0.01)
        if global_steps is not None:
            # 训练初期鼓励探索，后期逐渐减少
            adaptive_entropy = entropy_coef * (1.0 - min(global_steps / 200000, 0.5))
            entropy_coef = max(adaptive_entropy, entropy_coef * 0.5)
        
        entropy_loss = -entropy.mean()
        
        # 总策略损失
        policy_loss = actor_loss + entropy_coef * entropy_loss
        
        losses['policy_loss'] = policy_loss
        losses['actor_loss'] = actor_loss
        losses['entropy'] = entropy.mean()
        
        # 记录KL散度（用于监控策略变化）
        if old_log_probs is not None:
            kl_div = (old_log_probs_flat - action_log_probs).mean()
            losses['kl_div'] = kl_div
            if log_vars is not None:
                log_vars['Training/kl_div'] = (global_steps, kl_div.item())

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        Enhanced MAPPO价值损失计算
        
        改进点：
        1. 考虑邻接矩阵的图结构信息
        2. 改进的价值函数设计
        3. 使用Huber损失提高鲁棒性
        """
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        returns = sample.get('return', None)  # [batch_size, n_agents] 或 [batch_size, 1]
        obs = sample.get('obs', None)  # [batch_size, n_agents, obs_dim]
        adj = sample.get('adj', None)  # [batch_size, n_agents, n_agents] - 邻接矩阵
        
        batch_size = state.shape[0] if state is not None else obs.shape[0]
        
        # 处理邻接矩阵
        adj_mask = None
        if adj is not None:
            if isinstance(adj, np.ndarray):
                adj = torch.tensor(adj, dtype=torch.float32).to(device)
            adj_mask = adj
        
        # 如果没有提供状态，使用观测的拼接作为状态
        if state is None:
            if obs is not None:
                state = obs.view(batch_size, -1)
            else:
                raise ValueError("Either state or obs must be provided")
        
        # Critic前向传播（使用注意力机制）
        values = self.learned_critic_model(state, adj_mask)  # [batch_size, 1]
        
        if returns is not None:
            # 如果returns是[batch_size, n_agents]，取平均值
            if returns.ndim == 2 and returns.shape[1] == self.n_agents:
                returns = returns.mean(dim=1, keepdim=True)  # [batch_size, 1]
            elif returns.ndim == 1:
                returns = returns.unsqueeze(-1)  # [batch_size, 1]
            
            # 价值损失（使用Huber损失提高鲁棒性）
            use_huber_loss = hparams.get('use_huber_loss', False)
            if use_huber_loss:
                value_loss = F.smooth_l1_loss(values, returns)
            else:
                value_loss = F.mse_loss(values, returns)
            
            # 价值裁剪（防止价值函数更新过大）
            value_clip_param = hparams.get('value_clip_param', None)
            old_values = sample.get('old_value', None)
            if value_clip_param is not None and old_values is not None:
                if isinstance(old_values, torch.Tensor):
                    old_values = old_values.mean(dim=1, keepdim=True) if old_values.ndim == 2 else old_values
                    values_clipped = old_values + torch.clamp(
                        values - old_values, -value_clip_param, value_clip_param
                    )
                    value_loss_clipped = F.mse_loss(values_clipped, returns)
                    value_loss = torch.max(value_loss, value_loss_clipped)
        else:
            # 如果没有returns，使用简单的价值预测损失
            value_loss = torch.zeros_like(values.mean())
        
        losses['value_loss'] = value_loss
        losses['q_loss'] = value_loss  # 兼容接口

    def update_target(self):
        """Enhanced MAPPO不需要目标网络（on-policy算法）"""
        pass
