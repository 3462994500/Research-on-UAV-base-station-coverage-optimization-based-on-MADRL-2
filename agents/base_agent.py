import torch.nn as nn


class BaseAgent(nn.Module):
    def __init__(self):
        self.learned_model = None
        self.target_model = None

    def action(self, sample, epsilon, action_mode):
        raise NotImplementedError

    def cal_q_loss(self, sample, losses, log_vars):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError


class BaseActorCriticAgent(nn.Module):
    def __init__(self):
        self.learned_actor_model = None
        self.target_actor_model = None
        self.learned_critic_model = None
        self.target_critic_model = None

    def action(self, sample, epsilon, action_mode):
        raise NotImplementedError

    def cal_p_loss(self, sample, losses, log_vars):
        raise NotImplementedError

    def cal_q_loss(self, sample, losses, log_vars):
        raise NotImplementedError

    def update_target(self):
        raise NotImplementedError


class CentralizedAgent(nn.Module):
    """集中式训练+分布式执行架构的基类"""
    def __init__(self):
        super().__init__()
        self.central_model = None  # 中央全局模型
        self.decentralized_models = None  # 分布式执行模型（可选）
        
    def central_action(self, global_obs, global_adj, epsilon=0.3, action_mode='epsilon-greedy'):
        """中央控制器基于全局观测生成所有智能体的动作"""
        raise NotImplementedError
        
    def decentralized_action(self, local_obs, local_adj, epsilon=0.3, action_mode='epsilon-greedy'):
        """分布式执行时每个智能体基于局部观测生成动作"""
        raise NotImplementedError
        
    def cal_central_q_loss(self, sample, losses, log_vars=None, global_steps=None):
        """集中式训练的损失计算"""
        raise NotImplementedError
        
    def update_central_target(self):
        """更新中央目标网络"""
        raise NotImplementedError