import os
import torch
import numpy as np
from trainers.value_based_trainer import ValueBasedTrainer
from utils.scheduler import epsilon_scheduler
from utils.hparams import hparams
from utils.replay_buffer import ReplayBuffer
from utils.tb_logger import TensorBoardLogger
from utils.torch_utils import *
from utils.class_utils import *
from utils.checkpoint_utils import *
from utils.os_utils import *
import logging
import tqdm
from utils.device import device

class VDNTrainer(ValueBasedTrainer):
    """VDN训练器：专门为VDN算法设计"""
    
    def __init__(self):
        self.work_dir = os.path.join("checkpoints", hparams['exp_name'])
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_dir = self.work_dir if 'log_dir' not in hparams else os.path.join(self.work_dir, hparams['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化环境
        self.env = get_cls_from_path(hparams['scenario_path'])()
        
        # 计算状态维度（全局状态，VDN虽然不使用，但为了兼容性保留）
        state_dim = self._get_state_dim()
        
        # 初始化VDN智能体
        self.agent = get_cls_from_path(hparams['algorithm_path'])(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            n_agents=hparams['env_num_agent'],
            state_dim=state_dim
        ).to(device)
        
        self.replay_buffer = ReplayBuffer()
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=hparams['learning_rate'])
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter_critic = 0
        self.i_episode = 0
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False
        
        self.load_from_checkpoint_if_possible()

    def _get_state_dim(self):
        """计算全局状态维度（VDN不使用，但为了兼容性保留）"""
        n_agents = hparams['env_num_agent']
        n_poi = hparams['env_num_poi']
        
        # 状态包括：所有智能体位置、所有POI位置、环境状态等
        agent_pos_dim = 3 * n_agents  # (x,y,z) * n_agents
        poi_pos_dim = 2 * n_poi      # (x,y) * n_poi
        env_state_dim = n_agents + n_poi + 10  # 预留额外维度
        
        return agent_pos_dim + poi_pos_dim + env_state_dim

    def _get_global_state(self, obs, adj=None):
        """从局部观测构建全局状态（VDN不使用，但为了兼容性保留）"""
        n_agents = obs.shape[0]
        
        # 提取智能体位置信息（假设前3个维度是位置）
        agent_positions = obs[:, :3].flatten()
        
        # 构建全局状态（这里需要根据实际环境调整）
        global_state = np.concatenate([
            agent_positions,
            np.zeros(self._get_state_dim() - len(agent_positions))  # 填充剩余维度
        ])
        
        return global_state

    def _interaction_step(self, log_vars):
        """交互步骤：收集VDN训练数据"""
        obs, adj = self.env.reset()
        self.i_episode += 1
        epsilon = epsilon_scheduler(self.i_episode)
        self.tb_logger.add_scalars({'Epsilon': (self.i_episode, epsilon)})
        
        # 重置隐藏状态（从环境输入时batch_size=1）
        if hasattr(self.agent, 'reset_hidden_states'):
            self.agent.reset_hidden_states(batch_size=1)
        
        tmp_reward_lst = []
        
        for t in range(hparams['episode_length']):
            # 构建全局状态（VDN不使用，但为了兼容性保留）
            state = self._get_global_state(obs, adj)
            
            # 分布式执行
            action = self.agent.action(
                obs, adj, 
                epsilon=epsilon, 
                action_mode=hparams['training_action_mode']
            )
            
            reward, next_obs, next_adj, done = self.env.step(action)
            
            # 构建下一个全局状态
            next_state = self._get_global_state(next_obs, next_adj)
            
            # 存储VDN训练样本
            sample = {
                'obs': torch.tensor(obs, device=device),
                'state': torch.tensor(state, device=device),
                'adj': torch.tensor(adj, device=device),
                'action': torch.tensor(action, device=device),
                'reward': torch.tensor(reward, device=device),
                'next_obs': torch.tensor(next_obs, device=device),
                'next_state': torch.tensor(next_state, device=device),
                'next_adj': torch.tensor(next_adj, device=device),
                'done': torch.tensor(done, device=device)
            }
            
            # 添加隐藏状态（如果适用）
            if hasattr(self.agent, 'get_hidden_states'):
                hidden_states = self.agent.get_hidden_states()
                sample.update({k: v.cpu() for k, v in hidden_states.items()})
            
            self.replay_buffer.push(sample)
            obs, adj = next_obs, next_adj
            tmp_reward_lst.append(sum(reward))
            
        log_vars['Interaction/episodic_reward'] = (self.i_episode, sum(tmp_reward_lst))
        
        if hasattr(self.env, "get_log_vars"):
            tmp_env_log_vars = {f"Interaction/{k}": (self.i_episode, v) for k, v in self.env.get_log_vars().items()}
            log_vars.update(tmp_env_log_vars)

    def _training_step(self, log_vars):
        """训练步骤：VDN训练"""
        if not self.i_episode % hparams['training_interval'] == 0:
            return
            
        for _ in range(hparams['training_times']):
            self.i_iter_critic += 1
            batched_sample = self.replay_buffer.sample(hparams['batch_size'])
            
            if batched_sample is None:
                break
                
            # 确保样本在GPU上
            batched_sample = {k: v.to(device) for k, v in batched_sample.items()}
            
            losses = {}
            self.agent.cal_q_loss(batched_sample, losses, log_vars=log_vars, global_steps=self.i_iter_critic)
            
            total_loss = sum(losses.values())
            self.optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.agent.parameters(), hparams.get('grad_norm_clip', 10.0))
            
            self.optimizer.step()
            
            # 记录损失
            for loss_name, loss in losses.items():
                log_vars[f'Training/{loss_name}'] = (self.i_iter_critic, loss.item())
                
            log_vars['Training/q_grad'] = (self.i_iter_critic, get_grad_norm(self.agent, l=2))
            
            # 更新目标网络
            if self.i_iter_critic % hparams.get('target_update_interval', 100) == 0:
                self.agent.update_target()
