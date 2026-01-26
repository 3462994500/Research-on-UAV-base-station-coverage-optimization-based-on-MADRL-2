# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# import torch
# import numpy as np

# from utils.class_utils import get_cls_from_path
# from utils.device import device
import os
import logging
# from utils.hparams import set_hparams, hparams
from utils.class_utils import get_cls_from_path


if __name__ == '__main__':
    # use --config=<path to *.yaml> to control all training related settings
    # use --exp_name=<str> to set the path to store log and checkpoints
    # 使用--config=<path to*.yaml>来控制所有与训练相关的设置
    # 使用--exp_name=<str>设置存储日志和检查点的路径

# # 1.训练
    logging.basicConfig(level=logging.INFO)
    from utils.hparams import set_hparams, hparams
    set_hparams()

    trainer = get_cls_from_path(hparams['trainer_path'])()
    trainer.run()

#'''
# #2.tensorboard画网络结构
#     from utils.hparams import set_hparams, hparams
#     set_hparams()
#     env = get_cls_from_path(hparams['scenario_path'])()
#     agent = get_cls_from_path(hparams['algorithm_path'])(env.obs_dim, env.act_dim).to(device)
# '''


# #3.print网络结构
#    from modules.ac_drgn import ActorDRGNNetwork
#     set_hparams()
#     env = get_cls_from_path(hparams['scenario_path'])()
#     network_struct = get_cls_from_path(hparams['algorithm_path'])(env.obs_dim, env.act_dim).to(device)
#     print(network_struct)

#4.测试环境env能不能用 hparams['env_use_pixel_obs']=1、env_use_square_obs: false情况下，可以通
    # from utils.hparams import set_hparams, hparams
    # set_hparams()
    # env = get_cls_from_path(hparams['scenario_path'])()
    # agent = get_cls_from_path(hparams['algorithm_path'])(env.obs_dim, env.act_dim).to(device)
    # obs, adj = env.reset()
    # epsilon = hparams['min_epsilon']
    # action = agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
    # reward, next_obs, next_adj, done = env.step(action)
 #   print('reward, next_obs, next_adj, done',reward, next_obs, next_adj, done)

# 5.画环境env
    # from utils.hparams import set_hparams, hparams
    # set_hparams()
    # env = get_cls_from_path(hparams['scenario_path'])()
    # agent = get_cls_from_path(hparams['algorithm_path'])(env.obs_dim, env.act_dim).to(device)
    # obs, adj = env.reset()
    # epsilon = hparams['min_epsilon']
    # action = agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
    # reward, next_obs, next_adj, done = env.step(action)
# #表 不同方法的指标
#     import random
#     import numpy as np
#     from utils.hparams import set_hparams, hparams
#     from tabulate import tabulate
#     from utils.device import device
#     import os
#     import torch
#     set_hparams()
#     trainer = get_cls_from_path(hparams['trainer_path'])()
#     from scenarios.continuous_mcs3D import (
#         MB_greedy_policy, MB_GA_policy, suiji, potential_field_policy, KMeans_policy, APSO_policy
#     )
#
#     # 算法配置字典（名称 -> 策略函数及参数）
#     POLICY_CONFIG = {
#         # 'KMeans_policy': {'func': KMeans_policy, 'args': {}},
#         # 'APSO_policy': {'func': APSO_policy, 'args': {}},
#
#         # 'Random': {'func': suiji, 'args': {}},
#         # 'Potential': {'func': potential_field_policy, 'args': {}},
#         # 'MB-GA': {'func': MB_GA_policy, 'args': {'max_iter': 20}},
#         # 'MB-Greedy': {'func': MB_greedy_policy, 'args': {}},
#         'Ours': {}
#     }
#
#
#     def run_experiment(policy_name: str, seed: int, episode_length: int = 100):
#         """单次实验运行
#
#         Args:
#             policy_name: 算法名称
#             seed: 随机种子
#             episode_length: 实验步长
#
#         Returns:
#             metrics: 包含五个指标的字典
#         """
#         # 设置随机种子
#         random.seed(seed)
#         np.random.seed(seed)
#
#         # 初始化环境
#         env = get_cls_from_path('scenarios.continuous_mcs3D.ContinuousMCS3D')()
#
#
#         if hasattr(env, 'seed'):
#             env.seed(seed)
#         obs, adj = env.reset()
#
#         # 获取策略配置
#         config = POLICY_CONFIG[policy_name]
#         if policy_name == 'Ours':
#
#             agent = get_cls_from_path(hparams['algorithm_path'])(env.obs_dim, env.act_dim).to(device)
#             # ========== 新增模型加载逻辑 ==========
#             ckpt_dir = os.path.join("checkpoints", hparams['exp_name'])
#             best_ckpt_path = os.path.join(ckpt_dir, "model_ckpt_best.ckpt")
#             if os.path.exists(best_ckpt_path):
#                 checkpoint = torch.load(best_ckpt_path, map_location=device)
#                 agent.load_state_dict(checkpoint['agent'])
#                 logging.info(f"Loaded pretrained model from {best_ckpt_path}")
#             else:
#                 raise ValueError("Pretrained model not found. Train first!")
#             # ========== 加载逻辑结束 ==========
#             if hasattr(agent, 'reset_hidden_states'):
#                 agent.reset_hidden_states(obs.shape[0])
#             elif hasattr(agent, 'hyperdrqn_reset_hidden_states'):
#                 agent.hyperdrqn_reset_hidden_states(obs.shape[0])
#         else:
#             policy_func = config['func']
#             policy_args = config['args']
#
#         # 运行实验
#         rewards = []
#         for _ in range(episode_length):
#             if policy_name == 'Ours':
#                 epsilon = hparams['min_epsilon']
#                 action = agent.action(obs, adj, epsilon=epsilon, action_mode=hparams['testing_action_mode'])
#             else:
#                 action = policy_func(env, obs, adj, **policy_args)
#             reward, next_obs, next_adj, done = env.step(action)
#             obs, adj = next_obs, next_adj
#             rewards.append(sum(reward))
#             # if done:
#             #     break
#         # 收集指标
#         log_vars = env.get_log_vars()
#         return {
#             'C': log_vars['coverage_index'],
#             'F': log_vars['fairness_index'],
#             'E': log_vars['energy_index'],
#             'R': log_vars['sum_rate'],
#             'Co': log_vars['collide'],
#             'Reward': sum(rewards)
#         }
#
#
#     def format_throughput(value, std):
#         """智能单位转换函数"""
#         units = ['', 'K', 'M', 'B', 'T']
#         unit_index = 0
#         scaled_value = value
#         scaled_std = std
#
#         # 自动寻找最佳单位
#         while abs(scaled_value) >= 1000 and unit_index < len(units) - 1:
#             unit_index += 1
#             scaled_value /= 1000
#             scaled_std /= 1000
#
#         # 动态精度控制
#         if scaled_std > 100:
#             precision = 0
#         elif scaled_std > 10:
#             precision = 1
#         else:
#             precision = 2
#
#         # 特殊处理十亿级单位
#         if units[unit_index] == 'B':
#             return (f"{scaled_value:.{precision}f}B ± {scaled_std:.{precision}f}B",
#                     f"({value / 1e9:.2f} Billion)")
#
#         return f"{scaled_value:.{precision}f}{units[unit_index]} ± {scaled_std:.{precision}f}{units[unit_index]}", "(not Billion)"
#
#
#     def evaluate_algorithms(num_runs: int = 10, episode_length: int = 100):
#         """多算法性能评估
#
#         Args:
#             num_runs: 每个算法的运行次数
#             episode_length: 实验步长
#
#         Returns:
#             展示格式化的性能对比表格
#         """
#         # 存储结果 {算法: {指标: [值列表]}}
#         results = {algo: {m: [] for m in ['C', 'F', 'E', 'R', 'Co', 'Reward']}
#                    for algo in POLICY_CONFIG}
#
#         # 运行所有实验
#         for algo_name in POLICY_CONFIG:
#             print(f"Testing {algo_name}...")
#             for run_id in range(num_runs):
#                 seed = 1000 + run_id  # 可配置的种子序列
#                 metrics = run_experiment(algo_name, seed, episode_length)
#                 # metrics = run_experiment(algo_name, episode_length)
#                 for metric, value in metrics.items():
#                     results[algo_name][metric].append(value)
#
#         # 构建结果表格
#         table = []
#         footnotes = []  # 单位说明存储
#         for algo, metrics in results.items():
#             row = [algo]
#             for metric in ['C', 'F', 'E', 'R', 'Co', 'Reward']:
#                 values = np.array(metrics[metric])
#                 mean, std = values.mean(), values.std()
#
#                 if metric == 'R':
#                     # 智能格式化吞吐量
#                     # print(format_throughput(mean, std))
#                     formatted, note = format_throughput(mean, std)
#                     if note not in footnotes:
#                         footnotes.append(note)
#                     row.append(formatted)
#                 else:
#                     row.append(f"{mean:.4f} ± {std:.4f}")
#             table.append(row)
#
#             # 打印专业表格
#         headers = ["Algorithm", "Coverage", "Fairness", "Energy", "Throughput", "collide", "Reward"]
#         print(tabulate(table, headers=headers, tablefmt='grid', stralign='center'))
#
#         # 添加单位注释
#         if footnotes:
#             print("\nMeasurement Units:")
#             for note in footnotes:
#                 print(f"* {note}")
#
#
#     if __name__ == "__main__":
#         # 运行评估（5次实验取平均）
#
#         evaluate_algorithms(num_runs=10)
