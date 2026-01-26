import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseActorCriticAgent
from modules.decentral_ppo import DecentralPPOActor, DecentralPPOCritic
from utils.torch_utils import *
from utils.hparams import hparams
from utils.device import device


class DecentralPPOAgent(BaseActorCriticAgent):
    """
    完全去中心化PPO智能体：独立学习算法(Independent Learner)
    
    完全去中心化架构说明：
    ====================
    
    1. 完全去中心化（Fully Decentralized）：
       - 每个智能体都有独立的Actor和Critic网络
       - 每个智能体独立进行策略更新
       - 只使用自己的局部观测，不需要全局状态
       - 不需要集中式的全局价值函数
       - 智能体之间无需通信
    
    2. 优点：
       - 可扩展性强（增加智能体不需要改变网络结构）
       - 完全分布式，无中心点故障
       - 隐私保护好，每个智能体独立学习
       - 可应用于动态智能体数量的场景
    
    3. 缺点：
       - 训练可能不稳定（非平稳环境）
       - 无法直接利用全局信息
       - 可能出现智能体间协调困难
    
    4. 网络结构：
       - 每个智能体有独立的Actor和Critic网络
       - Actor基于局部观测输出动作概率分布
       - Critic基于局部观测评估状态价值
       - 各智能体参数独立学习，无共享参数
    
    5. 对比其他架构：
       - MAPPO: 集中式训练+分布式执行(CTDE), 共享Actor参数
       - DecentralPPO: 完全去中心化, 独立Actor和Critic
       - DQN: 集中式训练+分布式执行(CTDE), 基于Q值
    """
    
    def __init__(self, obs_dim, act_dim, n_agents):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.hidden_dim = hparams['hidden_dim']
        
        # 为每个智能体创建独立的Actor和Critic网络
        self.actor_networks = nn.ModuleList([
            DecentralPPOActor(obs_dim, self.hidden_dim, act_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        self.critic_networks = nn.ModuleList([
            DecentralPPOCritic(obs_dim, self.hidden_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        # 目标网络（用于稳定训练）
        self.target_actor_networks = nn.ModuleList([
            DecentralPPOActor(obs_dim, self.hidden_dim, act_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        self.target_critic_networks = nn.ModuleList([
            DecentralPPOCritic(obs_dim, self.hidden_dim)
            for _ in range(n_agents)
        ]).to(device)
        
        # 初始化目标网络
        for i in range(n_agents):
            self.target_actor_networks[i].load_state_dict(self.actor_networks[i].state_dict())
            self.target_critic_networks[i].load_state_dict(self.critic_networks[i].state_dict())
        
        # 兼容接口（虽然这个架构不需要集中式模型，但保留接口）
        self.learned_actor_model = self.actor_networks
        self.learned_critic_model = self.critic_networks
        self.target_actor_model = self.target_actor_networks
        self.target_critic_model = self.target_critic_networks

    def action(self, obs, adj=None, epsilon=0.3, action_mode='sample'):
        """
        完全去中心化执行：每个智能体基于自己的局部观测选择动作
        
        特点：
        1. 每个智能体使用自己的Actor网络
        2. 只需要局部观测，不需要全局信息
        3. 智能体之间完全独立，无需通信
        """
        if isinstance(obs, np.ndarray):
            # 从环境输入：[n_agents, obs_dim]
            assert obs.ndim == 2
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            batch_size = 1
        elif isinstance(obs, torch.Tensor):
            # 从回放缓冲区输入：可能是[n_agents, obs_dim]或[batch, n_agents, obs_dim]
            obs = to_cuda(obs)
            if obs.ndim == 2:
                batch_size = 1
            else:
                batch_size = obs.shape[0]
                obs = obs.view(-1, self.n_agents, self.obs_dim)
        else:
            raise TypeError("Unsupported observation type")
        
        with torch.no_grad():
            actions = []
            
            # 为每个智能体生成动作
            for agent_id in range(self.n_agents):
                if batch_size == 1:
                    # 单步交互：obs形状为[n_agents, obs_dim]
                    agent_obs = obs[agent_id:agent_id+1, :]  # [1, obs_dim]
                else:
                    # 批量处理：obs形状为[batch, n_agents, obs_dim]
                    agent_obs = obs[:, agent_id:agent_id+1, :]  # [batch, 1, obs_dim]
                    agent_obs = agent_obs.view(-1, self.obs_dim)  # [batch, obs_dim]
                
                if action_mode == 'sample':
                    # 采样动作
                    action, _, _ = self.actor_networks[agent_id].get_action_and_log_prob(agent_obs)
                elif action_mode == 'greedy':
                    # 贪婪动作
                    _, action_probs = self.actor_networks[agent_id](agent_obs)
                    action = action_probs.argmax(dim=-1)
                else:
                    raise ValueError(f"Unsupported action mode: {action_mode}")
                
                actions.append(action)
            
            # 堆叠动作 [n_agents] 或 [batch, n_agents]
            if batch_size == 1:
                actions = torch.stack(actions, dim=0).cpu().numpy()  # [n_agents]
            else:
                actions = torch.stack(actions, dim=1).cpu().numpy()  # [batch, n_agents]
        
        return actions

    def cal_p_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        完全去中心化PPO策略损失计算
        
        每个智能体独立计算自己的策略损失
        
        Args:
            sample: dict，包含所有智能体的样本数据
            losses: dict，用于存储各智能体的损失
            log_vars: dict，用于记录日志
            global_steps: int，全局步数
        """
        obs = sample['obs']  # [batch_size, n_agents, obs_dim]
        actions = sample['action'].long()  # [batch_size, n_agents]
        old_log_probs = sample.get('old_log_prob', None)  # [batch_size, n_agents]
        advantages = sample.get('advantage', None)  # [batch_size, n_agents]
        returns = sample.get('return', None)  # [batch_size, n_agents]
        
        batch_size = obs.shape[0]
        
        # 为每个智能体计算策略损失
        total_actor_loss = 0.0
        total_entropy = 0.0
        
        for agent_id in range(self.n_agents):
            # 提取该智能体的数据
            agent_obs = obs[:, agent_id, :]  # [batch_size, obs_dim]
            agent_actions = actions[:, agent_id]  # [batch_size]
            
            if old_log_probs is not None:
                agent_old_log_probs = old_log_probs[:, agent_id]  # [batch_size]
            
            if advantages is not None:
                agent_advantages = advantages[:, agent_id]  # [batch_size]
            
            # Actor前向传播
            action_log_probs, entropy = self.actor_networks[agent_id].evaluate_actions(
                agent_obs, agent_actions
            )
            
            # 计算重要性采样比率
            if old_log_probs is not None:
                ratio = torch.exp(action_log_probs - agent_old_log_probs)
            else:
                ratio = torch.exp(action_log_probs)
            
            # PPO clipped objective
            if advantages is not None:
                clip_param = hparams.get('clip_param', 0.2)
                surr1 = ratio * agent_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * agent_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
            else:
                actor_loss = -action_log_probs.mean()
            
            # 加入熵正则项
            entropy_coeff = hparams.get('entropy_coeff', 0.01)
            actor_loss = actor_loss - entropy_coeff * entropy.mean()
            
            total_actor_loss = total_actor_loss + actor_loss
            total_entropy = total_entropy + entropy.mean()
        
        # 平均每个智能体的损失
        avg_actor_loss = total_actor_loss / self.n_agents
        avg_entropy = total_entropy / self.n_agents
        
        losses['actor_loss'] = avg_actor_loss
        losses['entropy'] = avg_entropy
        
        if log_vars is not None:
            log_vars['Loss/actor_loss'] = (global_steps, avg_actor_loss.item())
            log_vars['Loss/entropy'] = (global_steps, avg_entropy.item())

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        完全去中心化PPO价值损失计算
        
        每个智能体独立计算自己的价值损失
        
        Args:
            sample: dict，包含所有智能体的样本数据
            losses: dict，用于存储各智能体的损失
            log_vars: dict，用于记录日志
            global_steps: int，全局步数
        """
        obs = sample['obs']  # [batch_size, n_agents, obs_dim]
        returns = sample.get('return', None)  # [batch_size, n_agents]
        
        if returns is None:
            return
        
        batch_size = obs.shape[0]
        
        # 为每个智能体计算价值损失
        total_critic_loss = 0.0
        
        for agent_id in range(self.n_agents):
            # 提取该智能体的数据
            agent_obs = obs[:, agent_id, :]  # [batch_size, obs_dim]
            agent_returns = returns[:, agent_id]  # [batch_size]
            
            # Critic前向传播
            values = self.critic_networks[agent_id](agent_obs).squeeze(-1)  # [batch_size]
            
            # 计算价值损失（MSE）
            critic_loss = F.mse_loss(values, agent_returns)
            
            total_critic_loss = total_critic_loss + critic_loss
        
        # 平均每个智能体的损失
        avg_critic_loss = total_critic_loss / self.n_agents
        
        losses['critic_loss'] = avg_critic_loss
        
        if log_vars is not None:
            log_vars['Loss/critic_loss'] = (global_steps, avg_critic_loss.item())

    def update_target(self):
        """更新目标网络"""
        for i in range(self.n_agents):
            self.target_actor_networks[i].load_state_dict(self.actor_networks[i].state_dict())
            self.target_critic_networks[i].load_state_dict(self.critic_networks[i].state_dict())

    def get_actor_networks(self):
        """获取所有Actor网络"""
        return self.actor_networks
    
    def get_critic_networks(self):
        """获取所有Critic网络"""
        return self.critic_networks
