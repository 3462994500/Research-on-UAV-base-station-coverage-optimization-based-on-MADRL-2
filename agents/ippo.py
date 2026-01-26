import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseActorCriticAgent
from modules.ippo import IPPOActor, IPPOCritic
from utils.torch_utils import *
from utils.hparams import hparams
from utils.device import device


class IPPOAgent(BaseActorCriticAgent):
    """
    Independent PPO Agent (IPPO)
    - 不使用集中式训练（无CTDE）
    - 每个智能体使用局部观测训练自己的价值估计（独立训练逻辑）
    - 支持分布式执行（每个智能体独立决策）
    """
    def __init__(self, obs_dim, act_dim, n_agents, state_dim=None):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.hidden_dim = hparams['hidden_dim']

        # Actor 和 Critic 都基于局部观测
        # 在独立PPO中，网络可以共享参数或不共享；这里采用参数共享的单一网络用于简洁实现
        self.learned_actor_model = IPPOActor(obs_dim, self.hidden_dim, act_dim).to(device)
        self.learned_critic_model = IPPOCritic(obs_dim, self.hidden_dim).to(device)

        # 兼容旧接口
        self.learned_model = self.learned_actor_model

    def action(self, obs, adj=None, epsilon=0.3, action_mode='sample'):
        """
        分布式执行：每个智能体仅使用自己的局部观测
        obs: np.ndarray 或 torch.Tensor，形状 [n_agents, obs_dim]
        """
        if isinstance(obs, np.ndarray):
            assert obs.ndim == 2
            obs = torch.tensor(obs, dtype=torch.float32).to(device)
            batch_size = 1
        elif isinstance(obs, torch.Tensor):
            obs = to_cuda(obs)
            if obs.ndim == 2:
                batch_size = 1
            else:
                batch_size = obs.shape[0]
        else:
            raise TypeError("Unsupported observation type")

        with torch.no_grad():
            obs_flat = obs.view(-1, self.obs_dim)
            if action_mode == 'sample':
                actions, _, _ = self.learned_actor_model.get_action_and_log_prob(obs_flat)
            elif action_mode == 'greedy':
                _, probs = self.learned_actor_model(obs_flat)
                actions = probs.argmax(dim=-1)
            else:
                raise ValueError(f"Unsupported action mode: {action_mode}")

            actions = actions.view(-1, self.n_agents) if batch_size != 1 and actions.ndim > 1 else actions.view(self.n_agents)
            if batch_size == 1:
                actions = actions.cpu().numpy()
            else:
                actions = actions.cpu().numpy()

        return actions

    def cal_p_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        策略损失（独立PPO）: 基于局部观测进行PPO更新
        sample 中的维度：obs [B, n_agents, obs_dim], action [B, n_agents], advantage [B, n_agents]
        """
        obs = sample['obs']
        actions = sample['action'].long()
        old_log_probs = sample.get('old_log_prob', None)
        advantages = sample.get('advantage', None)

        batch_size = obs.shape[0]

        obs_flat = obs.view(-1, self.obs_dim)
        actions_flat = actions.view(-1)

        if old_log_probs is not None:
            old_log_probs_flat = old_log_probs.view(-1)

        if advantages is not None:
            advantages_flat = advantages.view(-1)

        action_log_probs, entropy = self.learned_actor_model.evaluate_actions(obs_flat, actions_flat)

        if old_log_probs is not None:
            ratio = torch.exp(action_log_probs - old_log_probs_flat)
        else:
            ratio = torch.exp(action_log_probs)

        if advantages is not None:
            clip_param = hparams.get('clip_param', 0.2)
            surr1 = ratio * advantages_flat
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages_flat
            actor_loss = -torch.min(surr1, surr2).mean()
        else:
            actor_loss = -action_log_probs.mean()

        entropy_coef = hparams.get('entropy_coef', 0.01)
        entropy_loss = -entropy.mean()
        policy_loss = actor_loss + entropy_coef * entropy_loss

        losses['policy_loss'] = policy_loss
        losses['actor_loss'] = actor_loss
        losses['entropy'] = entropy.mean()

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        价值损失（独立PPO）：使用局部观测预测每个智能体的价值
        returns: [B, n_agents]
        """
        obs = sample.get('obs', None)
        returns = sample.get('return', None)

        batch_size = obs.shape[0]
        obs_flat = obs.view(-1, self.obs_dim)

        values_flat = self.learned_critic_model(obs_flat).view(-1)
        values = values_flat.view(batch_size, self.n_agents)

        if returns is not None:
            returns_tensor = returns
            if isinstance(returns_tensor, np.ndarray):
                returns_tensor = torch.tensor(returns_tensor, dtype=torch.float32).to(device)
            # returns_tensor shape [B, n_agents]
            value_loss = F.mse_loss(values, returns_tensor)
        else:
            value_loss = torch.zeros(1, device=values.device)

        losses['value_loss'] = value_loss
        losses['q_loss'] = value_loss

    def update_target(self):
        pass
