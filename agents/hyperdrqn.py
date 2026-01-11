import torch
import torch.nn as nn
import numpy as np

from utils.torch_utils import *
from utils.hparams import hparams
import torch.nn.functional as F
from agents.dqn import DQNAgent
from modules.hyperdrqn import HyperDRQN
from utils.device import device

class HyperDRQNAgent(DQNAgent):
    def __init__(self, in_dim, act_dim):
        nn.Module.__init__(self)
        self.in_dim = in_dim
        self.act_dim = act_dim
        self.hidden_dim = hparams['hidden_dim']
        self.num_head = hparams.get('num_head', 4)

        # 初始化双网络结构（保持变量名不变）
        self.learned_model = HyperDRQN(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            action_dim=act_dim,
            num_heads=self.num_head,
            skip_connect=hparams.get('skip_connect', True)
        )
        self.target_model = HyperDRQN(
            in_dim=in_dim,
            hidden_dim=self.hidden_dim,
            action_dim=act_dim,
            num_heads=self.num_head,
            skip_connect=hparams.get('skip_connect', True)
        )
        self.target_model.load_state_dict(self.learned_model.state_dict())

        # 隐藏状态管理（保持变量名不变）
        self.previous_critic_hidden_states = None
        self.current_critic_hidden_states = None

    def reset_hidden_states(self, n_ant):
        """保持接口一致，调整隐藏状态维度"""
        self.previous_critic_hidden_states = torch.zeros([n_ant, self.hidden_dim]).to(device)
        self.current_critic_hidden_states = torch.zeros([n_ant, self.hidden_dim]).to(device)

    def get_hidden_states(self):
        """保持原有的数据格式"""
        previous_critic_hidden = to_cpu(self.previous_critic_hidden_states.squeeze(0))
        current_critic_hidden = to_cpu(self.current_critic_hidden_states.squeeze(0))
        return {'cri_hid': previous_critic_hidden, 'next_cri_hid': current_critic_hidden}

    def action(self, obs, adj, epsilon=0.3, action_mode='epsilon-categorical'):
        """保持接口不变，内部适配新网络"""
        assert self.current_critic_hidden_states is not None
        critic_hidden = self.current_critic_hidden_states

        # 输入处理（保持变量名不变）
        is_batched = obs.ndim == 3
        if isinstance(obs, np.ndarray):
            obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
            adj = torch.FloatTensor(adj).unsqueeze(0).to(device)
        elif isinstance(obs, torch.Tensor):
            obs, adj = to_cuda(obs), to_cuda(adj)
        else:
            raise TypeError(f"Unsupported input type: {type(obs)}")

        # 前向推理（适配新网络参数）
        with torch.no_grad():
            q_values, new_hidden = self.learned_model(
                x=obs,
                adj_mask=adj,  # 使用mask参数名但保持adj输入
                hidden_state=critic_hidden
            )

        # 更新隐藏状态（保持变量名不变）
        self.previous_critic_hidden_states = self.current_critic_hidden_states
        self.current_critic_hidden_states = new_hidden.detach()

        # 动作选择（复用原有方法）
        return self._process_action(q_values, obs.shape[0], epsilon, action_mode, is_batched)

    def _process_action(self, q_values, batch_size, epsilon, mode, is_batched):
        """封装动作选择逻辑"""
        q_np = q_values.cpu().numpy()
        if is_batched:
            actions = []
            for b in range(batch_size):
                actions.append(self._sample_action_from_q_values(q_np[b], epsilon, mode))
            return np.stack(actions, axis=0)
        return self._sample_action_from_q_values(q_np.squeeze(0), epsilon, mode)

    def cal_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """保持原有计算流程，适配新网络输出"""
        obs = sample['obs']
        cri_hid = sample['cri_hid'].unsqueeze(0)  # 适配三维隐藏状态
        adj = sample['adj']
        actions = sample['action'].long()
        rewards = sample['reward']
        next_obs = sample['next_obs']
        next_cri_hid = sample['next_cri_hid'].unsqueeze(0)
        next_adj = sample['next_adj']
        dones = sample['done']

        # 当前Q值计算（保持变量名不变）
        current_q, _ = self.learned_model(obs, adj, cri_hid)

        # 目标Q值计算（保持Double DQN逻辑）
        with torch.no_grad():
            target_q, _ = self.target_model(next_obs, next_adj, next_cri_hid)
            target_q = target_q.max(dim=2)[0]

        # 计算TD目标（去除numba依赖）
        target = rewards + (1 - dones) * hparams['gamma'] * target_q

        # 计算损失（保持原有损失计算方式）
        q_loss = F.mse_loss(
            current_q.gather(2, actions.unsqueeze(-1)).squeeze(),
            target.detach()
        )
        losses['q_loss'] = q_loss

        # 日志记录（保持原有格式）
        if log_vars is not None:
            log_vars['q_mean'] = (global_steps, current_q.mean().item())
            log_vars['target_mean'] = (global_steps, target.mean().item())

    def _sample_action_from_q_values(self, q_values, epsilon, mode):
        """直接复用原有实现"""
        # 原样保持用户提供的实现
        if mode == 'epsilon-categorical':
            return self._epsilon_categorical(q_values, epsilon)
        elif mode == 'epsilon-greedy':
            return self._epsilon_greedy(q_values, epsilon)
        else:
            raise ValueError(f"Unsupported action mode: {mode}")

    def _epsilon_greedy(self, q_values, epsilon):
        """保持原有epsilon-greedy实现"""
        n_agent = q_values.shape[0]
        actions = q_values.argmax(axis=-1)
        random_mask = np.random.rand(n_agent) < epsilon
        actions[random_mask] = np.random.randint(0, self.act_dim, size=random_mask.sum())
        return actions

    def _epsilon_categorical(self, q_values, epsilon):
        """保持原有categorical实现"""
        noise = np.random.rand(*q_values.shape) * epsilon
        noisy_q = q_values + noise
        return noisy_q.argmax(axis=-1)