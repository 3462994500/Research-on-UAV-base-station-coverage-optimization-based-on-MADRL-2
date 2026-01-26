import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseActorCriticAgent
from modules.coma import COMAActor, COMACritic
from utils.torch_utils import *
from utils.hparams import hparams
from utils.device import device


class COMAAgent(BaseActorCriticAgent):
    """
    COMA智能体：实现集中式训练+分布式执行(CTDE)架构
    
    COMA (Counterfactual Multi-Agent Policy Gradients) 核心思想：
    ============================================================
    
    1. 集中式训练（Centralized Training）：
       - 训练时使用全局状态(state)和所有智能体的动作训练Critic网络
       - Critic网络输出每个智能体的Q值 Q(s, a_i, a_{-i})
       - 使用反事实基线减少策略梯度的方差
       - 反事实基线：Q(s, a_{-i}, a_i^b)，其中a_i^b是智能体i的基线动作（平均动作）
    
    2. 分布式执行（Decentralized Execution）：
       - 执行时每个智能体只使用自己的局部观测(obs)
       - Actor网络基于局部观测输出动作概率分布
       - 不需要全局状态或全局信息
       - 智能体之间不需要通信即可独立选择动作
       - 所有智能体共享同一个Actor网络参数（参数共享）
    
    3. 反事实基线优势：
       - 减少策略梯度的方差
       - 每个智能体的优势函数：A_i = Q(s, a_i, a_{-i}) - Q(s, a_{-i}, a_i^b)
       - 基线只依赖于其他智能体的动作，不依赖于当前智能体的动作
    
    4. 网络结构：
       - actor: 每个智能体的策略网络（共享参数）
       - critic: 集中式Q网络，使用全局状态和所有智能体的动作
    """
    def __init__(self, obs_dim, act_dim, n_agents, state_dim):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hparams['hidden_dim']
        
        # 创建Actor网络（每个智能体共享参数）
        self.learned_actor_model = COMAActor(obs_dim, self.hidden_dim, act_dim).to(device)
        
        # 创建Critic网络（集中式，使用全局状态和所有智能体的动作）
        self.learned_critic_model = COMACritic(
            state_dim, obs_dim, act_dim, n_agents, self.hidden_dim
        ).to(device)
        
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
        COMA策略损失计算（使用反事实基线）
        
        反事实基线优势函数：
        A_i = Q(s, a_i, a_{-i}) - Q(s, a_{-i}, a_i^b)
        其中a_i^b是智能体i的基线动作（使用策略的平均动作）
        
        策略梯度：
        ∇θ J = E[∇θ log π_i(a_i|o_i) * A_i]
        """
        obs = sample['obs']  # [batch_size, n_agents, obs_dim]
        actions = sample['action'].long()  # [batch_size, n_agents]
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        action_probs_all = sample.get('action_probs', None)  # [batch_size, n_agents, act_dim]
        
        batch_size = obs.shape[0]
        
        # 如果没有提供状态，使用观测的拼接作为状态
        if state is None:
            state = obs.view(batch_size, -1)
        
        # 重塑输入
        obs_flat = obs.view(-1, self.obs_dim)  # [batch_size * n_agents, obs_dim]
        actions_flat = actions.view(-1)  # [batch_size * n_agents]
        
        # Actor前向传播：获取动作log概率和动作概率分布
        action_log_probs, entropy = self.learned_actor_model.evaluate_actions(obs_flat, actions_flat)
        
        # 如果没有提供动作概率，需要重新计算
        if action_probs_all is None:
            with torch.no_grad():
                _, action_probs_all = self.learned_actor_model(obs_flat)
                action_probs_all = action_probs_all.view(batch_size, self.n_agents, self.act_dim)
        
        # 计算Q值：Q(s, a_i, a_{-i})
        q_values = self.learned_critic_model(state, actions)  # [batch_size, n_agents]
        
        # 计算反事实基线：Q(s, a_{-i}, a_i^b)
        counterfactual_baselines = torch.zeros_like(q_values)  # [batch_size, n_agents]
        
        for agent_id in range(self.n_agents):
            agent_action_probs = action_probs_all[:, agent_id, :]  # [batch_size, act_dim]
            baseline = self.learned_critic_model.get_counterfactual_baseline(
                state, actions, agent_id, agent_action_probs
            )
            counterfactual_baselines[:, agent_id] = baseline
        
        # 计算优势函数（反事实基线）
        advantages = q_values - counterfactual_baselines  # [batch_size, n_agents]
        
        # 重塑优势函数
        advantages_flat = advantages.view(-1)  # [batch_size * n_agents]
        
        # 策略损失（负的策略梯度，因为要最大化）
        policy_loss = -(action_log_probs * advantages_flat).mean()
        
        # 熵正则化（鼓励探索）
        entropy_coef = hparams.get('entropy_coef', 0.01)
        entropy_loss = -entropy.mean()
        
        # 总策略损失
        total_policy_loss = policy_loss + entropy_coef * entropy_loss
        
        losses['policy_loss'] = total_policy_loss
        losses['actor_loss'] = policy_loss
        losses['entropy'] = entropy.mean()
        losses['advantage_mean'] = advantages.mean()
        losses['advantage_std'] = advantages.std()

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        COMA价值损失计算
        
        使用TD目标训练Critic网络
        """
        state = sample.get('state', None)  # [batch_size, state_dim] - 全局状态
        actions = sample.get('action', None).long()  # [batch_size, n_agents]
        returns = sample.get('return', None)  # [batch_size, n_agents] - 每个智能体的回报
        obs = sample.get('obs', None)  # [batch_size, n_agents, obs_dim]
        
        batch_size = state.shape[0] if state is not None else obs.shape[0]
        
        # 如果没有提供状态，使用观测的拼接作为状态
        if state is None:
            if obs is not None:
                state = obs.view(batch_size, -1)
            else:
                raise ValueError("Either state or obs must be provided")
        
        # Critic前向传播：计算Q值
        q_values = self.learned_critic_model(state, actions)  # [batch_size, n_agents]
        
        if returns is not None:
            # 价值损失（MSE）
            value_loss = F.mse_loss(q_values, returns)
        else:
            # 如果没有returns，使用简单的价值预测损失
            value_loss = torch.zeros_like(q_values.mean())
        
        losses['value_loss'] = value_loss
        losses['q_loss'] = value_loss  # 兼容接口
        losses['q_mean'] = q_values.mean()
        losses['q_std'] = q_values.std()

    def update_target(self):
        """COMA不需要目标网络（on-policy算法）"""
        pass
