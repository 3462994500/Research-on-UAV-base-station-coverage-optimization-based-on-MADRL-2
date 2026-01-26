import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseActorCriticAgent
from modules.mappo import MAPPOActor, MAPPOCritic
from utils.torch_utils import *
from utils.hparams import hparams
from utils.device import device


class MAPPOAgent(BaseActorCriticAgent):
    """
    MAPPO智能体：实现集中式训练+分布式执行(CTDE)架构
    
    CTDE架构说明：
    =============
    
    1. 集中式训练（Centralized Training）：
       - 训练时使用全局状态(state)训练Critic网络
       - Critic网络评估全局状态价值，用于计算优势函数
       - Actor网络使用优势函数进行策略梯度更新
       - 训练过程中可以使用全局信息（如全局状态）
    
    2. 分布式执行（Decentralized Execution）：
       - 执行时每个智能体只使用自己的局部观测(obs)
       - Actor网络基于局部观测输出动作概率分布
       - 不需要全局状态或全局信息
       - 智能体之间不需要通信即可独立选择动作
       - 所有智能体共享同一个Actor网络参数（参数共享）
    
    3. 网络结构：
       - actor: 每个智能体的策略网络（共享参数）
       - critic: 集中式价值网络，使用全局状态
    """
    def __init__(self, obs_dim, act_dim, n_agents, state_dim):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hparams['hidden_dim']
        
        # 创建Actor网络（每个智能体共享参数）
        self.learned_actor_model = MAPPOActor(obs_dim, self.hidden_dim, act_dim).to(device)
        
        # 创建Critic网络（集中式，使用全局状态）
        self.learned_critic_model = MAPPOCritic(state_dim, self.hidden_dim).to(device)
        
        # 兼容接口
        self.learned_model = self.learned_actor_model

    def action(self, obs, adj=None, epsilon=0.3, action_mode='sample'):
        """
        分布式执行（Decentralized Execution）：每个智能体基于局部观测选择动作
        
        CTDE架构的分布式执行特点：
        1. 每个智能体只使用自己的局部观测(obs)计算动作
        2. 不需要全局状态或全局信息
        3. 智能体之间不需要通信即可独立选择动作
        4. 所有智能体共享同一个Actor网络参数（参数共享）
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
        
        with torch.no_grad():
            # 重塑观测为 [batch_size * n_agents, obs_dim]
            obs_flat = obs.view(-1, self.obs_dim)
            
            if action_mode == 'sample':
                # 采样动作
                actions, _, _ = self.learned_actor_model.get_action_and_log_prob(obs_flat)
                actions = actions.view(batch_size, self.n_agents)
            elif action_mode == 'greedy':
                # 贪婪动作
                _, action_probs = self.learned_actor_model(obs_flat)
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
        MAPPO策略损失计算（PPO clipped objective）
        
        集中式训练的特点：
        1. 使用全局状态训练Critic
        2. 使用优势函数更新Actor策略
        3. PPO clipped objective防止策略更新过大
        """
        obs = sample['obs']  # [batch_size, n_agents, obs_dim]
        actions = sample['action'].long()  # [batch_size, n_agents]
        old_log_probs = sample.get('old_log_prob', None)  # [batch_size, n_agents]
        advantages = sample.get('advantage', None)  # [batch_size, n_agents]
        returns = sample.get('return', None)  # [batch_size, n_agents]
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        
        batch_size = obs.shape[0]
        
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
        
        # Actor前向传播
        action_log_probs, entropy = self.learned_actor_model.evaluate_actions(obs_flat, actions_flat)
        
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
        
        # 熵正则化（鼓励探索）
        entropy_coef = hparams.get('entropy_coef', 0.01)
        entropy_loss = -entropy.mean()
        
        # 总策略损失
        policy_loss = actor_loss + entropy_coef * entropy_loss
        
        losses['policy_loss'] = policy_loss
        losses['actor_loss'] = actor_loss
        losses['entropy'] = entropy.mean()

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        MAPPO价值损失计算
        
        集中式训练：使用全局状态训练Critic
        """
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        returns = sample.get('return', None)  # [batch_size, n_agents] 或 [batch_size, 1]
        obs = sample.get('obs', None)  # [batch_size, n_agents, obs_dim]
        
        batch_size = state.shape[0] if state is not None else obs.shape[0]
        
        # 如果没有提供状态，使用观测的拼接作为状态
        if state is None:
            if obs is not None:
                state = obs.view(batch_size, -1)
            else:
                raise ValueError("Either state or obs must be provided")
        
        # Critic前向传播
        values = self.learned_critic_model(state)  # [batch_size, 1]
        
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
