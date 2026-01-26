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


class IPPOTrainer(BaseTrainer):
    """
    IPPO 训练器：独立 PPO（不使用 CTDE），分布式执行
    每个智能体基于局部观测训练自己的价值估计（实现为共享网络的简化独立 PPO）
    """

    def __init__(self):
        self.work_dir = os.path.join("checkpoints", hparams['exp_name'])
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_dir = self.work_dir if 'log_dir' not in hparams else os.path.join(self.work_dir, hparams['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)

        # 初始化环境
        self.env = get_cls_from_path(hparams['scenario_path'])()

        # 初始化 IPPO 智能体
        self.agent = get_cls_from_path(hparams['algorithm_path'])(
            obs_dim=self.env.obs_dim,
            act_dim=self.env.act_dim,
            n_agents=hparams['env_num_agent'],
        ).to(device)

        # 轨迹缓冲区
        self.trajectory_buffer = []

        # 优化器
        actor_params = list(self.agent.learned_actor_model.parameters())
        critic_params = list(self.agent.learned_critic_model.parameters())
        # self.optimizer = torch.optim.Adam(actor_params + critic_params, lr=hparams['learning_rate'])
        lr = float(hparams.get('learning_rate', 1e-3))
        self.optimizer = torch.optim.Adam(actor_params + critic_params, lr=lr)
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter = 0
        self.i_episode = 0
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False

        self.load_from_checkpoint_if_possible()

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

    @property
    def i_iter_dict(self):
        return {'i_iter': self.i_iter, 'i_episode': self.i_episode}

    def _load_i_iter_dict(self, i_iter_dict):
        self.i_iter = i_iter_dict['i_iter']
        self.i_episode = i_iter_dict['i_episode']

    def _interaction_step(self, log_vars):
        obs, adj = self.env.reset()
        self.i_episode += 1
        self.trajectory_buffer = []

        tmp_reward_lst = []
        episode_length = hparams['episode_length']

        for t in range(episode_length):
            # 采样动作
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
                obs_flat = obs_tensor.view(-1, self.env.obs_dim)  # [n_agents, obs_dim]

                actions, action_log_probs, _ = self.agent.learned_actor_model.get_action_and_log_prob(obs_flat)
                actions = actions.cpu().numpy()
                action_log_probs = action_log_probs.cpu().numpy()

                # 价值估计（每个智能体）
                values = self.agent.learned_critic_model(obs_flat)
                values = values.squeeze().cpu().numpy()  # [n_agents]

            reward, next_obs, next_adj, done = self.env.step(actions)

            trajectory_step = {
                'obs': obs.copy(),
                'action': actions.copy(),
                'reward': reward.copy(),
                'action_log_prob': action_log_probs.copy(),
                'value': values.copy(),
                'done': done.copy(),
                'next_obs': next_obs.copy()
            }
            self.trajectory_buffer.append(trajectory_step)

            obs, adj = next_obs, next_adj
            tmp_reward_lst.append(sum(reward))

        # 最后一步的价值
        with torch.no_grad():
            final_obs = torch.tensor(obs, dtype=torch.float32).to(device)
            final_obs_flat = final_obs.view(-1, self.env.obs_dim)
            final_value = self.agent.learned_critic_model(final_obs_flat).squeeze().cpu().numpy()

        log_vars['Interaction/episodic_reward'] = (self.i_episode, sum(tmp_reward_lst))
        if hasattr(self.env, "get_log_vars"):
            tmp_env_log_vars = {f"Interaction/{k}": (self.i_episode, v) for k, v in self.env.get_log_vars().items()}
            log_vars.update(tmp_env_log_vars)

    def _compute_gae(self, rewards, values, dones, next_value, gamma=0.99, lam=0.95):
        T = len(rewards)
        n_agents = rewards[0].shape[0]
        advantages = np.zeros((T, n_agents))
        returns = np.zeros((T, n_agents))
        gae = np.zeros(n_agents)

        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]

            delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + gamma * lam * next_non_terminal * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]

        return advantages, returns

    def _training_step(self, log_vars):
        if not self.i_episode % hparams['training_interval'] == 0:
            return
        if len(self.trajectory_buffer) == 0:
            return

        episode_length = len(self.trajectory_buffer)
        n_agents = self.env.n_agent

        rewards = np.array([step['reward'] for step in self.trajectory_buffer])  # [T, n_agents]
        values = np.array([step['value'] for step in self.trajectory_buffer])  # [T, n_agents]
        dones = np.array([step['done'] for step in self.trajectory_buffer])
        old_log_probs = np.array([step['action_log_prob'] for step in self.trajectory_buffer])

        # 计算最后一步价值
        final_obs = self.trajectory_buffer[-1]['next_obs']
        with torch.no_grad():
            final_obs_tensor = torch.tensor(final_obs, dtype=torch.float32).to(device)
            final_obs_flat = final_obs_tensor.view(-1, self.env.obs_dim)
            final_value = self.agent.learned_critic_model(final_obs_flat).squeeze().cpu().numpy()

        gamma = hparams.get('gamma', 0.99)
        lam = hparams.get('gae_lambda', 0.95)
        advantages, returns = self._compute_gae(rewards, values, dones, final_value, gamma, lam)

        if hparams.get('normalize_advantages', True):
            advantages_flat = advantages.flatten()
            advantages_mean = advantages_flat.mean()
            advantages_std = advantages_flat.std()
            if advantages_std > 1e-8:
                advantages = (advantages - advantages_mean) / (advantages_std + 1e-8)
            else:
                advantages = advantages - advantages_mean

        obs_batch = np.array([step['obs'] for step in self.trajectory_buffer])  # [T, n_agents, obs_dim]
        action_batch = np.array([step['action'] for step in self.trajectory_buffer])  # [T, n_agents]

        obs_tensor = torch.tensor(obs_batch, dtype=torch.float32).to(device)
        action_tensor = torch.tensor(action_batch, dtype=torch.long).to(device)
        old_log_prob_tensor = torch.tensor(old_log_probs, dtype=torch.float32).to(device)
        advantage_tensor = torch.tensor(advantages, dtype=torch.float32).to(device)
        return_tensor = torch.tensor(returns, dtype=torch.float32).to(device)

        num_epochs = hparams.get('ppo_epochs', 4)
        batch_size = hparams.get('batch_size', episode_length)

        for epoch in range(num_epochs):
            indices = np.arange(episode_length)
            np.random.shuffle(indices)
            for start_idx in range(0, episode_length, batch_size):
                end_idx = min(start_idx + batch_size, episode_length)
                batch_indices = indices[start_idx:end_idx]

                batch_obs = obs_tensor[batch_indices]
                batch_action = action_tensor[batch_indices]
                batch_old_log_prob = old_log_prob_tensor[batch_indices]
                batch_advantage = advantage_tensor[batch_indices]
                batch_return = return_tensor[batch_indices]

                losses = {}
                sample = {
                    'obs': batch_obs,
                    'action': batch_action,
                    'old_log_prob': batch_old_log_prob,
                    'advantage': batch_advantage,
                    'return': batch_return
                }

                self.agent.cal_p_loss(sample, losses, log_vars=log_vars, global_steps=self.i_iter)
                self.agent.cal_q_loss(sample, losses, log_vars=log_vars, global_steps=self.i_iter)

                policy_loss = losses.get('policy_loss', 0)
                value_loss = losses.get('value_loss', 0)
                total_loss = policy_loss + hparams.get('value_loss_coef', 0.5) * value_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), hparams.get('grad_norm_clip', 0.5))
                self.optimizer.step()
                self.i_iter += 1

                for loss_name, loss in losses.items():
                    log_vars[f'Training/{loss_name}'] = (self.i_iter, loss.item())
                log_vars['Training/total_loss'] = (self.i_iter, total_loss.item())

    def _testing_step(self, log_vars):
        if not self.i_episode % hparams['testing_interval'] == 0:
            return
        episodic_reward_lst = []
        for _ in tqdm.tqdm(range(1, hparams['testing_episodes'] + 1), desc='Testing Episodes: '):
            obs, adj = self.env.reset()
            tmp_reward_lst = []
            for t in range(hparams['episode_length']):
                action = self.agent.action(obs, adj=None, epsilon=0.0, action_mode='greedy')
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj
                tmp_reward_lst.append(sum(reward))
            episodic_reward_lst.append(sum(tmp_reward_lst))

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
        start_episode = self.i_episode
        for _ in tqdm.tqdm(range(start_episode, hparams['num_episodes'] + 1), desc='Training Episode: '):
            log_vars = {}
            self._interaction_step(log_vars=log_vars)
            self._training_step(log_vars=log_vars)
            self._testing_step(log_vars=log_vars)
            self.tb_logger.add_scalars(log_vars)
