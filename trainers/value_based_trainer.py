import os
import torch
import numpy as np
from trainers.base_trainer import BaseTrainer

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
from importlib import reload

import copy
class ValueBasedTrainer(BaseTrainer):
    """
    This is the Main Controller for training a *Value-based* DRL algorithm. 基于值
    """

    def __init__(self):
        self.work_dir = os.path.join("checkpoints", hparams['exp_name'])
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_dir = self.work_dir if 'log_dir' not in hparams else os.path.join(self.work_dir, hparams['log_dir'])
        os.makedirs(self.log_dir, exist_ok=True)

        self.env = get_cls_from_path(hparams['scenario_path'])()
        # 将 agent 转移到 GPU
        self.agent = get_cls_from_path(hparams['algorithm_path'])(self.env.obs_dim, self.env.act_dim).to(device)

        self.replay_buffer = ReplayBuffer()
        # 确保优化器的参数在 GPU 上
        self.optimizer = torch.optim.Adam(self.agent.learned_model.parameters(), lr=hparams['learning_rate'])
        self.tb_logger = TensorBoardLogger(self.log_dir)

        self.i_iter_critic = 0
        self.i_episode = 0
        # Note that if word_dir already has config.yaml, it might override your manual setting!
        # So delete the old config.yaml when you want to do some modifications.
        # 请注意，如果word_dir已经有config.yaml，它可能会覆盖您的手动设置！
        # 因此，当您想进行一些修改时，请删除旧的config.yaml。
        self.load_from_checkpoint_if_possible()
        self.best_eval_reward = -1e15
        self.save_best_ckpt = False

    @property
    def i_iter_dict(self):
        return {'i_critic': self.i_iter_critic, 'i_episode': self.i_episode}

    def _load_i_iter_dict(self, i_iter_dict):
        self.i_iter_critic = i_iter_dict['i_critic']
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
            logging.info(f"Latest checkpoint found at f{ckpt_path}, try loading...")
            try:
                self._load_checkpoint(checkpoint=ckpt)
            except:
                logging.warning("Checkpoint loading failed, now learn from scratch!")

    def save_checkpoint(self):
        # before save checkpoint, first delete redundant old checkpoints
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
        obs, adj = self.env.reset()
        self.i_episode += 1
        epsilon = epsilon_scheduler(self.i_episode)
        self.tb_logger.add_scalars({'Epsilon': (self.i_episode, epsilon)})
        if hasattr(self.agent, 'reset_hidden_states'):
            self.agent.reset_hidden_states(obs.shape[0])
        elif hasattr(self.agent, 'hyperdrqn_reset_hidden_states'):
            self.agent.hyperdrqn_reset_hidden_states(obs.shape[0])
        tmp_reward_lst = []
        # scene = []
        for t in range(hparams['episode_length']):  # episode_length: 100
            action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['training_action_mode'])
            # print('SFD', action)
            reward, next_obs, next_adj, done = self.env.step(action)
            # scene_data = {
            #     'uavs': copy.deepcopy(self.env.agents.pos),  # 深拷贝当前时刻的pos_agent值
            #     'pois': copy.deepcopy(self.env.pos_poi),  # 深拷贝当前时刻的pos_poi值
            #     'adj': copy.deepcopy(adj),  # 深拷贝当前时刻的adj值
            #     'robs': copy.deepcopy(self.env.robs),  # 深拷贝当前时刻的robs值
            #     'rcov': copy.deepcopy(self.env.rcov),  # 深拷贝当前时刻的rcov值
            #     'collide_adj': copy.deepcopy(self.env.collide_adj),  # 深拷贝当前时刻的rcov值
            #     'agent_cover': copy.deepcopy(self.env.agent_cover),  # 深拷贝当前时刻的rcov值
            # }
            # scene.append(scene_data)
            # 将观测、动作、奖励等转移到 GPU
            sample = {
                'obs': torch.tensor(obs, device=device),
                'adj': torch.tensor(adj, device=device),
                'action': torch.tensor(action, device=device),
                'reward': torch.tensor(reward, device=device),
                'next_obs': torch.tensor(next_obs, device=device),
                'next_adj': torch.tensor(next_adj, device=device),
                'done': torch.tensor(done, device=device)
            }
            if hasattr(self.agent, 'get_hidden_states'):
                hidden_states = self.agent.get_hidden_states()
                # 将隐藏状态转移到 GPU
                sample.update({k: v.to(device) for k, v in hidden_states.items()})
            # 处理多层隐藏状态
            elif hasattr(self.agent, 'hyperdrqn_get_hidden_states'):
                hidden_states = self.agent.hyperdrqn_get_hidden_states()
                sample['cri_hid'] = [h.cpu() for h in hidden_states['cri_hid']]  # 存储为CPU张量列表
                sample['next_cri_hid'] = [h.cpu() for h in hidden_states['next_cri_hid']]
            self.replay_buffer.push(sample)
            obs, adj = next_obs, next_adj
            tmp_reward_lst.append(sum(reward))
            # print('rew', sum(reward))
        # plot_coverage_heatmap(scene)
        # visualize_3d(scene)
        log_vars['Interaction/episodic_reward'] = (self.i_episode, sum(tmp_reward_lst))
        # print('tmp_reward_lst',sum(tmp_reward_lst))
        if hasattr(self.env, "get_log_vars"):
            tmp_env_log_vars = {f"Interaction/{k}": (self.i_episode, v) for k, v in self.env.get_log_vars().items()}
            log_vars.update(tmp_env_log_vars)

    def _training_step(self, log_vars):
        if not self.i_episode % hparams['training_interval'] == 0:  # training_interval: 1
            return
        for _ in range(hparams['training_times']):  # training_times: 4
            self.i_iter_critic += 1
            batched_sample = self.replay_buffer.sample(hparams['batch_size'])
            if batched_sample is None:
                # The replay buffer has not store enough sample.
                break
            losses = {}
            # 确保样本张量在 GPU 上
            batched_sample = {k: v.to(device) for k, v in batched_sample.items()}
            self.agent.cal_q_loss(batched_sample, losses, log_vars=log_vars, global_steps=self.i_iter_critic)
            total_loss = sum(losses.values())
            self.optimizer.zero_grad()
            total_loss.backward()
            for loss_name, loss in losses.items():
                log_vars[f'Training/{loss_name}'] = (self.i_iter_critic, loss.item())
            log_vars['Training/q_grad'] = (self.i_iter_critic, get_grad_norm(self.agent.learned_model, l=2))
            self.optimizer.step()

            if self.i_iter_critic % 5 == 0:
                self.agent.update_target()

    def _testing_step(self, log_vars):
        if not self.i_episode % hparams['testing_interval'] == 0:  # testing_interval: 100
            return
        episodic_reward_lst = []
        if hasattr(self.env, "get_log_vars"):
            episodic_env_log_vars = {}
        for _ in tqdm.tqdm(range(1, hparams['testing_episodes'] + 1), desc='Testing Episodes: '):  # testing_episodes: 20,[1,20]
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            elif hasattr(self.agent, 'hyperdrqn_reset_hidden_states'):
                self.agent.hyperdrqn_reset_hidden_states(obs.shape[0])
            tmp_reward_lst = []
            for t in range(hparams['episode_length']):  # episode_length: 100
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
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
        # Record the total reward obtain by all agents at each time step
        episodic_reward_mean = np.mean(episodic_reward_lst)
        episodic_reward_std = np.std(episodic_reward_lst)
        log_vars['Testing/mean_episodic_reward'] = (self.i_episode, episodic_reward_mean)
        log_vars['Testing/std_episodic_reward'] = (self.i_episode, episodic_reward_std)

        logging.info(
            f"Episode {self.i_episode} evaluation reward: mean {episodic_reward_mean},"
            f" std {episodic_reward_std}")
        # Save checkpoint when each testing phase is end.
        if episodic_reward_mean > self.best_eval_reward:
            self.save_best_ckpt = True
            logging.info(
                f"Best evaluation reward update: {self.best_eval_reward} ==> {episodic_reward_mean}")
            self.best_eval_reward = episodic_reward_mean
        else:
            self.save_best_ckpt = False
        self.save_checkpoint()

    def run_training_loop(self):
        start_episode = self.i_episode
        for _ in tqdm.tqdm(range(start_episode, hparams['num_episodes'] + 1), desc='Training Episode: '):#num_episodes: 100000,[start_episode, hparams['num_episodes']]
            log_vars = {}  # e.g. {'Training/q_loss':(16000, 0.999)}
            # Interaction Phase
            self._interaction_step(log_vars=log_vars)
            # Training Phase
            self._training_step(log_vars=log_vars)
            # Testing Phase
            self._testing_step(log_vars=log_vars)
            self.tb_logger.add_scalars(log_vars)

    def run_display_loop(self):
        while True:
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            for t in range(hparams['episode_length']):
                self.env.render()
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj
                if done:
                    break

    def run_eval_loop(self):
        #对模型进行20个回合的测试，每个回合100步，记录每个回合的平均步数
        hparams['eval_episodes'] = hparams['testing_episodes']
        hparams['eval_result_name'] = 'out.csv'
        rew_array = np.zeros(shape=[hparams['eval_episodes']])
        for i_episode in tqdm.tqdm(range(0, hparams['eval_episodes']), desc='Eval Episodes: '):
            tmp_reward_lst = []
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            elif hasattr(self.agent, 'hyperdrqn_reset_hidden_states'):
                self.agent.hyperdrqn_reset_hidden_states(obs.shape[0])
            for t in range(hparams['episode_length']):
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done = self.env.step(action)
                obs, adj = next_obs, next_adj
                tmp_reward_lst.append(sum(reward))

            rew_array[i_episode] = sum(tmp_reward_lst) / hparams['episode_length']
        np.savetxt(os.path.join(self.work_dir, hparams['eval_result_name']), rew_array, delimiter=',')
        mean, std = rew_array.mean(), rew_array.std()
        logging.info(f"Evaluation complete, reward mean {mean}, std {std} .")
        logging.info(f"Evaluation result is saved at {os.path.join(self.work_dir, hparams['eval_result_name'])}.")

    def run_CFE_loop(self):
        #对模型进行1个回合的评估，每个回合100步，记录每步数的C,F,E,R,CFE
        hparams['CFE_result_name']='CFE.csv'
        all_steps_data = []
        for i_episode in tqdm.tqdm(range(0, hparams['CFE_episodes']), desc='CFE Episodes: '):
            obs, adj = self.env.reset()
            if hasattr(self.agent, 'reset_hidden_states'):
                self.agent.reset_hidden_states(obs.shape[0])
            for t in range(hparams['step_episode_length']):
                epsilon = hparams['min_epsilon']
                action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                reward, next_obs, next_adj, done, out = self.env.step(action, stepout=True)
                obs, adj = next_obs, next_adj
                # 计算当前步的奖励总和
                # step_reward = sum(reward)
                # 获取当前步的其他指标
                step_coverage = out['coverage_index']
                step_fairness = out['fairness_index']
                step_energy = out['energy_index']
                step_sum_rate = out['sum_rate']
                step_collide = out['collide']
                all_steps_data.append([step_coverage, step_fairness, step_energy, step_sum_rate, step_collide])
            # 将列表转换为 NumPy 数组
        result_array = np.array(all_steps_data)
        np.savetxt(os.path.join(self.work_dir, hparams['CFE_result_name']), result_array, delimiter=',')
        mean_reward = result_array[:, 0].mean()
        std_reward = result_array[:, 0].std()
        logging.info(f"Evaluation complete, reward mean {mean_reward}, std {std_reward} .")
        logging.info(f"Evaluation result is saved at {os.path.join(self.work_dir, hparams['CFE_result_name'])}.")

    def run_user_loop(self):#结果错，不知错在哪
        #对模型进行10个回合的评估，每个回合100步，记录每步数的C,F,E,R,CFE
        hparams['user_result_name']='user.csv'
        all_episode_data = []
        for user_speed in range(0, 11, 1):
            hparams['max_poi_speed'] = user_speed
            self.env = get_cls_from_path(hparams['scenario_path'])()
            # 每次创建新的环境实例
            reward_list = []
            coverage = []
            fairness = []
            energy = []
            sum_rate = []
            collide = []
            for i_episode in tqdm.tqdm(range(0, hparams['user_episodes']), desc='user Episodes: '):
                obs, adj = self.env.reset()
                if hasattr(self.agent, 'reset_hidden_states'):
                    self.agent.reset_hidden_states(obs.shape[0])
                for t in range(hparams['user_episode_length']):
                    epsilon = hparams['min_epsilon']
                    action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                    reward, next_obs, next_adj, done, out = self.env.step(action, stepout=True)
                    obs, adj = next_obs, next_adj
                    reward_list.append(sum(reward))
                coverage.append(out['coverage_index'])
                fairness.append(out['fairness_index'])
                energy.append(out['energy_index'])
                sum_rate.append(out['sum_rate'])
                collide.append(out['collide'])
            reward_episode = sum(reward_list) / hparams['user_episodes']
            coverage_episode = sum(coverage) / hparams['user_episodes']
            fairness_episode = sum(fairness) / hparams['user_episodes']
            energy_episode = sum(energy) / hparams['user_episodes']
            sum_rate_episode = sum(sum_rate) / hparams['user_episodes']
            collide_episode = sum(collide) / hparams['user_episodes']
            all_episode_data.append([user_speed, reward_episode, coverage_episode, fairness_episode, energy_episode, sum_rate_episode, collide_episode])
            # 将列表转换为 NumPy 数组
        result_array = np.array(all_episode_data)
        np.savetxt(os.path.join(self.work_dir, hparams['user_result_name']), result_array, delimiter=',')
        logging.info(f"Evaluation result is saved at {os.path.join(self.work_dir, hparams['user_result_name'])}.")


    def run_user_num_loop(self):#结果错，不知错在哪
        #对模型进行10个回合的评估，每个回合100步，记录每步数的C,F,E,R,CFE
        hparams['user_num_result_name'] ='user_num.csv'
        all_episode_data = []
        for user_num in range(50, 200, 10):
            hparams['env_num_poi'] = user_num
            self.env = get_cls_from_path(hparams['scenario_path'])()
            # 每次创建新的环境实例
            reward_list = []
            coverage = []
            fairness = []
            energy = []
            sum_rate = []
            collide = []
            for i_episode in tqdm.tqdm(range(0, hparams['user_episodes']), desc='user Episodes: '):
                obs, adj = self.env.reset()
                if hasattr(self.agent, 'reset_hidden_states'):
                    self.agent.reset_hidden_states(obs.shape[0])
                for t in range(hparams['user_episode_length']):
                    epsilon = hparams['min_epsilon']
                    action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                    reward, next_obs, next_adj, done, out = self.env.step(action, stepout=True)
                    obs, adj = next_obs, next_adj
                    reward_list.append(sum(reward))
                coverage.append(out['coverage_index'])
                fairness.append(out['fairness_index'])
                energy.append(out['energy_index'])
                sum_rate.append(out['sum_rate'])
                collide.append(out['collide'])
            reward_episode = sum(reward_list) / hparams['user_episodes']
            coverage_episode = sum(coverage) / hparams['user_episodes']
            fairness_episode = sum(fairness) / hparams['user_episodes']
            energy_episode = sum(energy) / hparams['user_episodes']
            sum_rate_episode = sum(sum_rate) / hparams['user_episodes']
            collide_episode = sum(collide) / hparams['user_episodes']
            all_episode_data.append([user_num, reward_episode, coverage_episode, fairness_episode, energy_episode, sum_rate_episode, collide_episode])
            # 将列表转换为 NumPy 数组
        result_array = np.array(all_episode_data)
        np.savetxt(os.path.join(self.work_dir, hparams['user_num_result_name']), result_array, delimiter=',')
        logging.info(f"Evaluation result is saved at {os.path.join(self.work_dir, hparams['user_num_result_name'])}.")
    def run_SNR_loop(self):
        #对模型进行10个回合的评估，每个回合100步，记录每步数的C,F,E,R,CFE
        hparams['SNR_result_name']='SNR.csv'
        all_episode_data = []
        outage_probability = 0
        for i in range(0, 11, 1):
            hparams['outage_probability'] = outage_probability
            self.env = get_cls_from_path(hparams['scenario_path'])()
            # 每次创建新的环境实例
            reward_list = []
            coverage = []
            fairness = []
            energy = []
            sum_rate = []
            collide = []
            for i_episode in tqdm.tqdm(range(0, hparams['user_episodes']), desc='user Episodes: '):
                obs, adj = self.env.reset()
                if hasattr(self.agent, 'reset_hidden_states'):
                    self.agent.reset_hidden_states(obs.shape[0])
                for t in range(hparams['user_episode_length']):
                    epsilon = hparams['min_epsilon']
                    action = self.agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
                    reward, next_obs, next_adj, done, out = self.env.step(action, stepout=True)
                    obs, adj = next_obs, next_adj
                    reward_list.append(sum(reward))
                coverage.append(out['coverage_index'])
                fairness.append(out['fairness_index'])
                energy.append(out['energy_index'])
                sum_rate.append(out['sum_rate'])
                collide.append(out['collide'])
            reward_episode = sum(reward_list) / hparams['user_episodes']
            coverage_episode = sum(coverage) / hparams['user_episodes']
            fairness_episode = sum(fairness) / hparams['user_episodes']
            energy_episode = sum(energy) / hparams['user_episodes']
            sum_rate_episode = sum(sum_rate) / hparams['user_episodes']
            collide_episode = sum(collide) / hparams['user_episodes']
            all_episode_data.append([outage_probability, reward_episode, coverage_episode, fairness_episode, energy_episode, sum_rate_episode, collide_episode])
            outage_probability += 0.1
            # 将列表转换为 NumPy 数组
        result_array = np.array(all_episode_data)
        np.savetxt(os.path.join(self.work_dir, hparams['SNR_result_name']), result_array, delimiter=',')
        logging.info(f"Evaluation result is saved at {os.path.join(self.work_dir, hparams['SNR_result_name'])}.")


    def run(self):
        if hparams['display']:
            self.run_display_loop()
        elif hparams['evaluate']:
            self.run_eval_loop()
        elif hparams['CFE']:
            self.run_CFE_loop()
        elif hparams['user']:
            self.run_user_loop()
        elif hparams['user_num']:
            self.run_user_num_loop()
        elif hparams['SNR']:
            self.run_SNR_loop()
        else:
            self.run_training_loop()