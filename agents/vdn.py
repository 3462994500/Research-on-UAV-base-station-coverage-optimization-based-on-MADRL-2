import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.base_agent import BaseAgent
from modules.vdn import VDNAgentNetwork
from utils.numba_utils import *
from utils.torch_utils import *
from utils.hparams import hparams
from utils.device import device


class VDNAgent(BaseAgent):
    """
    VDN智能体：实现集中式训练+分布式执行(CTDE)架构
    
    VDN (Value Decomposition Networks) 核心思想：
    ============================================
    
    1. 集中式训练（Centralized Training）：
       - 训练时将各个智能体的Q值简单相加得到全局Q值
       - Q_total = sum(Q_i)，其中Q_i是智能体i的Q值
       - 通过优化全局Q值与全局TD目标的差异来更新网络参数
       - 训练过程中可以使用全局信息（如全局状态、所有智能体的观测）
    
    2. 分布式执行（Decentralized Execution）：
       - 执行时每个智能体只使用自己的局部观测(obs)
       - 不需要全局状态或全局信息
       - 智能体之间不需要通信即可独立选择动作
       - 所有智能体共享同一个网络参数（参数共享）
    
    3. 与QMIX的区别：
       - QMIX使用混合网络：Q_total = MixingNetwork(Q_i, state)
       - VDN使用简单相加：Q_total = sum(Q_i)
       - VDN更简单，但表达能力较弱
    
    4. 网络结构：
       - agent_network: 每个智能体的Q网络（共享参数）
       - 使用目标网络稳定训练（Double Q-learning）
    """
    def __init__(self, obs_dim, act_dim, n_agents, state_dim):
        nn.Module.__init__(self)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.state_dim = state_dim
        self.hidden_dim = hparams['hidden_dim']
        
        # 创建智能体网络（每个智能体共享参数）
        self.agent_network = VDNAgentNetwork(obs_dim, self.hidden_dim, act_dim).to(device)
        self.target_agent_network = VDNAgentNetwork(obs_dim, self.hidden_dim, act_dim).to(device)
        self.target_agent_network.load_state_dict(self.agent_network.state_dict())
        
        # 隐藏状态管理
        self.hidden_states = None
        self.target_hidden_states = None
        
        # 兼容接口
        self.learned_model = self.agent_network

    def reset_hidden_states(self, batch_size=1):
        """重置隐藏状态"""
        self.hidden_states = torch.zeros(batch_size * self.n_agents, self.hidden_dim).to(device)
        self.target_hidden_states = torch.zeros(batch_size * self.n_agents, self.hidden_dim).to(device)

    def get_hidden_states(self):
        """获取隐藏状态"""
        return {'hidden_states': self.hidden_states}

    def action(self, obs, adj=None, epsilon=0.3, action_mode='epsilon-greedy'):
        """
        分布式执行（Decentralized Execution）：每个智能体基于局部观测选择动作
        
        CTDE架构的分布式执行特点：
        1. 每个智能体只使用自己的局部观测(obs)计算Q值
        2. 不需要全局状态或全局信息
        3. 智能体之间不需要通信即可独立选择动作
        4. 所有智能体共享同一个网络参数（参数共享）
        """
        is_batched_input = obs.ndim == 3
        
        if isinstance(obs, np.ndarray):
            # 从环境输入
            assert obs.ndim == 2
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            batch_size = 1
        elif isinstance(obs, torch.Tensor):
            # 从回放缓冲区输入
            obs = to_cuda(obs)
            batch_size = obs.shape[0]
        else:
            raise TypeError("Unsupported observation type")
        
        # 确保隐藏状态已初始化，并且batch_size匹配
        expected_hidden_size = batch_size * self.n_agents
        if self.hidden_states is None:
            self.reset_hidden_states(batch_size)
        elif self.hidden_states.shape[0] != expected_hidden_size:
            # 如果batch_size不匹配，重置隐藏状态
            self.reset_hidden_states(batch_size)
        
        with torch.no_grad():
            # 重塑观测为 [batch_size * n_agents, obs_dim]
            obs_flat = obs.view(-1, self.obs_dim)
            
            # 前向传播
            q_values, next_hidden = self.agent_network(obs_flat, self.hidden_states)
            q_values = q_values.view(batch_size, self.n_agents, self.act_dim)
            
            # 更新隐藏状态
            self.hidden_states = next_hidden
            
            # 动作选择
            if is_batched_input:
                q_numpy = q_values.cpu().numpy()
                actions = []
                for b_i in range(batch_size):
                    q_i = q_numpy[b_i]  # [n_agents, act_dim]
                    action_i = self._sample_actions(q_i, epsilon, action_mode)
                    actions.append(action_i)
                actions = np.stack(actions, axis=0)
            else:
                q_numpy = q_values.squeeze(0).cpu().numpy()  # [n_agents, act_dim]
                actions = self._sample_actions(q_numpy, epsilon, action_mode)
                
        return actions

    def _sample_actions(self, q_values, epsilon, action_mode):
        """从Q值采样动作"""
        actions = []
        n_agents = q_values.shape[0]
        
        if action_mode == 'epsilon-greedy':
            for i in range(n_agents):
                if np.random.rand() < epsilon:
                    a = np.random.randint(self.act_dim)
                else:
                    a = q_values[i].argmax().item()
                actions.append(a)
        elif action_mode == 'greedy':
            for i in range(n_agents):
                a = q_values[i].argmax().item()
                actions.append(a)
        else:
            raise ValueError(f"Unsupported action mode: {action_mode}")
            
        return np.array(actions, dtype=np.float32).reshape([n_agents])

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """
        VDN集中式训练：使用简单的Q值相加计算损失
        
        VDN的核心思想：
        1. 将各个智能体的Q值简单相加：Q_total = sum(Q_i)
        2. 训练目标是使全局Q值接近全局TD目标
        3. 与QMIX的区别：VDN不使用混合网络，而是直接相加
        """
        obs = sample['obs']  # [batch_size, n_agents, obs_dim]
        actions = sample['action'].long()  # [batch_size, n_agents]
        rewards = sample['reward']  # [batch_size, n_agents] - 个体奖励
        next_obs = sample['next_obs']  # [batch_size, n_agents, obs_dim]
        dones = sample['done']  # [batch_size, n_agents]
        
        batch_size = obs.shape[0]
        
        # 集中式训练：每个batch独立，隐藏状态重置为0
        # 这是因为replay buffer中的样本是独立采样的
        hidden_states = torch.zeros(batch_size * self.n_agents, self.hidden_dim).to(device)
        target_hidden_states = torch.zeros(batch_size * self.n_agents, self.hidden_dim).to(device)
        
        # 重塑输入
        obs_flat = obs.view(-1, self.obs_dim)  # [batch_size * n_agents, obs_dim]
        next_obs_flat = next_obs.view(-1, self.obs_dim)  # [batch_size * n_agents, obs_dim]
        
        # 分布式执行：每个智能体基于局部观测计算Q值
        current_q, _ = self.agent_network(obs_flat, hidden_states)
        current_q = current_q.view(batch_size, self.n_agents, self.act_dim)  # [batch_size, n_agents, act_dim]
        
        # 选择动作对应的Q值
        chosen_action_qvals = torch.gather(current_q, dim=2, index=actions.unsqueeze(-1)).squeeze(-1)  # [batch_size, n_agents]
        
        # 目标Q值计算（Double Q-learning）
        with torch.no_grad():
            # 目标网络计算下一状态Q值
            target_q, _ = self.target_agent_network(next_obs_flat, target_hidden_states)
            target_q = target_q.view(batch_size, self.n_agents, self.act_dim)  # [batch_size, n_agents, act_dim]
            
            # 使用当前网络选择动作（Double Q-learning）
            _, next_hidden = self.agent_network(next_obs_flat, hidden_states)
            next_q, _ = self.agent_network(next_obs_flat, next_hidden)
            next_q = next_q.view(batch_size, self.n_agents, self.act_dim)
            
            # 选择最优动作
            max_actions = next_q.max(dim=2)[1]  # [batch_size, n_agents]
            
            # 目标网络计算目标Q值
            target_max_q = torch.gather(target_q, dim=2, index=max_actions.unsqueeze(-1)).squeeze(-1)  # [batch_size, n_agents]
            
            # 计算TD目标（个体层面）
            targets = rewards + (1 - dones.float()) * hparams['gamma'] * target_max_q  # [batch_size, n_agents]
        
        # VDN核心：简单的Q值相加
        # Q_total = sum(Q_i)
        total_q = chosen_action_qvals.sum(dim=1, keepdim=True)  # [batch_size, 1]
        total_target_q = targets.sum(dim=1, keepdim=True)  # [batch_size, 1]
        
        # 计算损失（全局Q值 vs 全局TD目标）
        q_loss = F.mse_loss(total_q, total_target_q.detach())
        losses['q_loss'] = q_loss

    def update_target(self):
        """更新目标网络"""
        if hparams['soft_update_target_network']:
            soft_update(self.agent_network, self.target_agent_network, hparams['tau'])
        else:
            hard_update(self.agent_network, self.target_agent_network)
