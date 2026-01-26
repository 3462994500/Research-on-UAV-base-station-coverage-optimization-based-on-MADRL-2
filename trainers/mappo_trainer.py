import os
import torch
import numpy as np
from trainers.base_trainer import BaseTrainer
from utils.hparams import hparams
from utils.tb_logger import TensorBoardLogger
from utils.torch_utils import *
from utils.class_utils import *
from utils.checkpoint_utils import *
from utils.os_utils import *
import logging
import tqdm
from utils.device import device
from collections import defaultdict


class MAPPOTrainer(BaseTrainer):
    """
    MAPPO训练器：实现集中式训练+分布式执行的PPO算法
    
    On-policy算法特点：
    1. 收集完整轨迹后立即训练
    2. 使用GAE计算优势函数
    3. 使用PPO clipped objective防止策略更新过大
    4. 支持多轮训练（多个epoch）
    """
    
    def __init__(self):
        self.work_dir = os.path.join("checkpoints", hparams['exp_name'])
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_dir = self.work_dir if 'log_dir' not in hparams else os.path.join(self.work_dir, hparams['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化环境
        self.env = get_cls_from_path(hparams['scenario_path'])()
        
        # 计算状态维度（全局状态）
        state_dim = self._get_state_dim()
        
        # 初始化MAPPO智能体
        self.agent = get_cls_from_path(hparams['algorithm_path'])(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            n_agents=hparams['env_num_agent'],
            state_dim=state_dim
        ).to(device)
        
        # On-policy算法不需要replay buffer，而是使用轨迹缓冲区
        self.trajectory_buffer = []
        
        # 创建优化器（Actor和Critic可以共享或分开）
        actor_params = list(self.agent.learned_actor_model.parameters())
        critic_params = list(self.agent.learned_critic_model.parameters())
        self.optimizer = torch.optim.Adam(
            actor_params + critic_params, 
            lr=hparams['learning_rate']
        )
        
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter = 0
        self.i_episode = 0
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False
        
        self.load_from_checkpoint_if_possible()

    def _get_state_dim(self):
        """计算全局状态维度"""
        n_agents = hparams['env_num_agent']
        n_poi = hparams['env_num_poi']
        
        # 状态包括：所有智能体位置、所有POI位置、环境状态等
        agent_pos_dim = 3 * n_agents  # (x,y,z) * n_agents
        poi_pos_dim = 2 * n_poi      # (x,y) * n_poi
        env_state_dim = n_agents + n_poi + 10  # 预留额外维度
        
        return agent_pos_dim + poi_pos_dim + env_state_dim

    def _get_global_state(self, obs, adj=None):
        """从局部观测构建全局状态"""
        n_agents = obs.shape[0]
        
        # 提取智能体位置信息（从观测中提取位置相关特征）
        # 根据continuous_mcs3D.py中的观测结构，位置信息在特定位置
        # 这里简化处理：使用观测的拼接作为全局状态
        global_state = obs.flatten()
        
        # 如果维度不够，进行填充或截断
        target_dim = self._get_state_dim()
        if len(global_state) < target_dim:
            padding = np.zeros(target_dim - len(global_state))
            global_state = np.concatenate([global_state, padding])
        elif len(global_state) > target_dim:
            global_state = global_state[:target_dim]
        
        return global_state

    def _compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        """
        计算GAE（Generalized Advantage Estimation）
        
        Args:
            rewards: [T, n_agents] 奖励序列
            values: [T, n_agents] 价值估计序列
            dones: [T, n_agents] 终止标志序列
            next_value: [n_agents] 最后一步的价值估计
            gamma: 折扣因子
            lam: GAE参数
            
        Returns:
            advantages: [T, n_agents] 优势函数
            returns: [T, n_agents] 回报
        """
        T = len(rewards)
        n_agents = rewards[0].shape[0]
        
        advantages = np.zeros((T, n_agents))
        returns = np.zeros((T, n_agents))
        
        gae = 0
        next_value = next_value.flatten() if next_value.ndim > 1 else next_value
        
        # 从后往前计算GAE
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1].flatten() if values[t + 1].ndim > 1 else values[t + 1]
            
            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t].flatten()
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            
            # 计算returns
            returns[t] = advantages[t] + values[t].flatten()
        
        return advantages, returns

    @property
    def i_iter_dict(self):
        return {'i_iter': self.i_iter, 'i_episode': self.i_episode}

    def _load_i_iter_dict(self, i_iter_dict):
        self.i_iter = i_iter_dict['i_iter']
        self.i_episode = i_iter_dict['i_episode']

    def _load_checkpoint(self, checkpoint):
        self.agent.load_state_dict(checkpoint['agent'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self._load_i_iter_dict(checkpoint['i_iter_dict'])
        logging.info("Checkpoint loaded successfully!")

    def load_from_checkpoint_if_possible(self):
        ckpt, ckpt_path = get_last_checkpoint(self.work_dir)
        if ckpt is None:
            logging.info("No checkpoint found, learn the agent from scratch!")
        else:
            logging.info(f"Latest checkpoint found at {ckpt_path}, try loading...")
            try:
                self._load_checkpoint(checkpoint=ckpt)
            except:
                logging.warning("Checkpoint loading failed, now learn from scratch!")

    def save_checkpoint(self):
        # 保存前先删除冗余的旧检查点
        all_ckpt_path = get_all_ckpts(self.work_dir)
        if len(all_ckpt_path) >= hparams['num_max_keep_ckpt'] - 1:
            ckpt_to_delete = all_ckpt_path[hparams['num_max_keep_ckpt'] - 1:]
            remove_files(ckpt_to_delete)
        ckpt_path = os.path.join(self.work_dir, f"model_ckpt_episodes_{self.i_episode}.ckpt")
        checkpoint = {}
        checkpoint['agent'] = self.agent.state_dict()
        checkpoint['optimizer'] = self.optimizer.state_dict()
        checkpoint['i_iter_dict'] = self.i_iter_dict
        torch.save(checkpoint, ckpt_path)
        if self.save_best_ckpt:
            ckpt_path = os.path.join(self.work_dir, f"model_ckpt_best.ckpt")
            torch.save(checkpoint, ckpt_path)

    def _interaction_step(self, log_vars):
        """交互步骤：收集MAPPO训练轨迹"""
        obs, adj = self.env.reset()
        self.i_episode += 1
        
        # 清空轨迹缓冲区
        self.trajectory_buffer = []
        
        tmp_reward_lst = []
        episode_length = hparams['episode_length']
        
        # 收集完整轨迹
        for t in range(episode_length):
            # 构建全局状态
            state = self._get_global_state(obs, adj)
            
            # 分布式执行：采样动作
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                obs_flat = obs_tensor.view(-1, self.env.obs_dim)  # [n_agents, obs_dim]
                
                # 获取动作和log概率
                actions, action_log_probs, _ = self.agent.learned_actor_model.get_action_and_log_prob(obs_flat)
                # actions: [n_agents], action_log_probs: [n_agents]
                actions = actions.cpu().numpy()  # [n_agents]
                action_log_probs = action_log_probs.cpu().numpy()  # [n_agents]
                
                # 获取价值估计
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                values = self.agent.learned_critic_model(state_tensor)
                values = values.squeeze().cpu().numpy()
            
            # 执行动作
            reward, next_obs, next_adj, done = self.env.step(actions)
            
            # 构建下一个全局状态
            next_state = self._get_global_state(next_obs, next_adj)
            
            # 存储轨迹数据
            trajectory_step = {
                'obs': obs.copy(),
                'state': state.copy(),
                'action': actions.copy(),
                'reward': reward.copy(),
                'action_log_prob': action_log_probs.copy(),
                'value': values.copy(),
                'done': done.copy(),
                'next_obs': next_obs.copy(),
                'next_state': next_state.copy()
            }
            self.trajectory_buffer.append(trajectory_step)
            
            obs, adj = next_obs, next_adj
            tmp_reward_lst.append(sum(reward))
        
        # 计算最后一步的价值（用于GAE计算）
        with torch.no_grad():
            final_state = self._get_global_state(obs, adj)
            final_state_tensor = torch.tensor(final_state, dtype=torch.float32).unsqueeze(0).to(device)
            final_value = self.agent.learned_critic_model(final_state_tensor)
            final_value = final_value.squeeze().cpu().numpy()
        
        log_vars['Interaction/episodic_reward'] = (self.i_episode, sum(tmp_reward_lst))
        
        if hasattr(self.env, "get_log_vars"):
            tmp_env_log_vars = {f"Interaction/{k}": (self.i_episode, v) for k, v in self.env.get_log_vars().items()}
            log_vars.update(tmp_env_log_vars)

    def _training_step(self, log_vars):
        """训练步骤：MAPPO训练（使用收集的轨迹）"""
        if not self.i_episode % hparams['training_interval'] == 0:
            return
        
        if len(self.trajectory_buffer) == 0:
            return
        
        # 提取轨迹数据
        episode_length = len(self.trajectory_buffer)
        n_agents = self.env.n_agent
        
        rewards = np.array([step['reward'] for step in self.trajectory_buffer])  # [T, n_agents]
        values = np.array([step['value'] for step in self.trajectory_buffer])  # [T, n_agents] 或 [T, 1]
        dones = np.array([step['done'] for step in self.trajectory_buffer])  # [T, n_agents]
        old_log_probs = np.array([step['action_log_prob'] for step in self.trajectory_buffer])  # [T, n_agents]
        
        # 处理values维度
        if values.ndim == 1:
            values = values.reshape(-1, 1).repeat(n_agents, axis=1)
        elif values.shape[1] == 1:
            values = values.repeat(n_agents, axis=1)
        
        # 计算最后一步的价值
        final_state = self.trajectory_buffer[-1]['next_state']
        with torch.no_grad():
            final_state_tensor = torch.tensor(final_state, dtype=torch.float32).unsqueeze(0).to(device)
            final_value = self.agent.learned_critic_model(final_state_tensor)
            final_value = final_value.squeeze().cpu().numpy()
            if final_value.ndim == 0:
                final_value = np.array([final_value] * n_agents)
            elif final_value.shape[0] == 1:
                final_value = final_value.repeat(n_agents)
        
        # 计算GAE和returns
        gamma = hparams.get('gamma', 0.99)
        lam = hparams.get('gae_lambda', 0.95)
        advantages, returns = self._compute_gae(rewards, values, dones, final_value, gamma, lam)
        
        # 归一化优势函数（可选）
        # 注意：在多智能体场景中，应该对每个智能体分别归一化，或者对所有智能体一起归一化
        if hparams.get('normalize_advantages', True):
            # 展平所有优势值进行归一化
            advantages_flat = advantages.flatten()
            advantages_mean = advantages_flat.mean()
            advantages_std = advantages_flat.std()
            # 避免除零，并确保稳定性
            if advantages_std > 1e-8:
                advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
            else:
                # 如果标准差太小，不进行归一化
                advantages = advantages - advantages_mean
        
        # 准备训练数据
        obs_batch = np.array([step['obs'] for step in self.trajectory_buffer])  # [T, n_agents, obs_dim]
        state_batch = np.array([step['state'] for step in self.trajectory_buffer])  # [T, state_dim]
        action_batch = np.array([step['action'] for step in self.trajectory_buffer])  # [T, n_agents]
        
        # 转换为tensor
        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(device)
        state_tensor = torch.tensor(state_batch, dtype=torch.float32).to(device)
        action_tensor = torch.tensor(action_batch, dtype=torch.long).to(device)
        old_log_prob_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        return_tensor = torch.tensor(returns, dtype=torch.float32).to(device)
        
        # 多轮训练（PPO epochs）
        num_epochs = hparams.get('ppo_epochs', 4)
        batch_size = hparams.get('batch_size', episode_length)
        
        for epoch in range(num_epochs):
            # 随机打乱数据
            indices = np.arange(episode_length)
            np.random.shuffle(indices)
            
            # 分批训练
            for start_idx in range(0, episode_length, batch_size):
                end_idx = min(start_idx + batch_size, episode_length)
                batch_indices = indices[start_idx:end_idx]
                
                batch_obs = obs_tensor[batch_indices]
                batch_state = state_tensor[batch_indices]
                batch_action = action_tensor[batch_indices]
                batch_old_log_prob = old_log_prob_tensor[batch_indices]
                batch_advantage = advantage_tensor[batch_indices]
                batch_return = return_tensor[batch_indices]
                
                # 计算损失
                losses = {}
                
                # 策略损失
                sample = {
                    'obs': batch_obs,
                    'state': batch_state,
                    'action': batch_action,
                    'old_log_prob': batch_old_log_prob,
                    'advantage': batch_advantage,
                    'return': batch_return
                }
                self.agent.cal_p_loss(sample, losses, log_vars=log_vars, global_steps=self.i_iter)
                
                # 价值损失
                self.agent.cal_q_loss(sample, losses, log_vars=log_vars, global_steps=self.i_iter)
                
                # 总损失
                policy_loss = losses.get('policy_loss', 0)
                value_loss = losses.get('value_loss', 0)
                total_loss = policy_loss + hparams.get('value_loss_coef', 0.5) * value_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), hparams.get('grad_norm_clip', 0.5))
                
                self.optimizer.step()
                self.i_iter += 1
                
                # 记录损失
                for loss_name, loss in losses.items():
                    log_vars[f'Training/{loss_name}'] = (self.i_iter, loss.item())
                log_vars['Training/total_loss'] = (self.i_iter, total_loss.item())
                log_vars['Training/policy_grad'] = (self.i_iter, get_grad_norm(self.agent.learned_actor_model, l=2))
                log_vars['Training/value_grad'] = (self.i_iter, get_grad_norm(self.agent.learned_critic_model, l=2))

    def _testing_step(self, log_vars):
        """测试步骤"""
        if not self.i_episode % hparams['testing_interval'] == 0:
            return
        episodic_reward_lst = []
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {}
        for _ in tqdm.tqdm(range(1, hparams['testing_episodes'] + 1), desc='Testing Episodes: '):
            obs, adj = self.env.reset()
            tmp_reward_lst = []
            for t in range(hparams['episode_length']):
                # 贪婪动作选择
                action = self.agent.action(obs, adj=None, epsilon=0.0, action_mode='greedy')
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj
                tmp_reward_lst.append(sum(reward))
            episodic_reward_lst.append(sum(tmp_reward_lst))
            if hasattr(self.env, "get_log_vars"):
                tmp_env_log_vars = self.env.get_log_vars()
                for k, v in tmp_env_log_vars.items():
                    if k not in episodic_env_log_vars.keys():
                        episodic_env_log_vars[k] = []
                    episodic_env_log_vars[k].append(v)
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {f"Testing/{k}": (self.i_episode, np.mean(v)) for k, v in
                                     episodic_env_log_vars.items()}
            log_vars.update(episodic_env_log_vars)
        episodic_reward_mean = np.mean(episodic_reward_lst)
        episodic_reward_std = np.std(episodic_reward_lst)
        log_vars['Testing/mean_episodic_reward'] = (self.i_episode, episodic_reward_mean)
        log_vars['Testing/std_episodic_reward'] = (self.i_episode, episodic_reward_std)

        logging.info(
            f"Episode {self.i_episode} evaluation reward: mean {episodic_reward_mean},"
            f" std {episodic_reward_std}")
        if episodic_reward_mean > self.best_eval_reward:
            self.save_best_ckpt = True
            logging.info(
                f"Best evaluation reward update: {self.best_eval_reward} ==> {episodic_reward_mean}")
            self.best_eval_reward = episodic_reward_mean
        else:
            self.save_best_ckpt = False
        self.save_checkpoint()

    def run(self):
        """主训练循环"""
        start_episode = self.i_episode
        for _ in tqdm.tqdm(range(start_episode, hparams['num_episodes'] + 1), desc='Training Episode: '):
            log_vars = {}
            # Interaction Phase
            self._interaction_step(log_vars=log_vars)
            # Training Phase
            self._training_step(log_vars=log_vars)
            # Testing Phase
            self._testing_step(log_vars=log_vars)
            self.tb_logger.add_scalars(log_vars)
