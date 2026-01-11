import math
import random

import networkx as nx
import numba

import numpy as np
import copy
from sko.GA import GA

from scenarios.continuous_uav_base.agents_3D import AgentGroup3D
from utils.hparams import hparams
import utils.wireness_communication as wc



N_AGENT = hparams['env_num_agent']
N_POI = hparams['env_num_poi']
N_MAJOR = hparams['env_num_major_point']
MAJOR_POINT_MIN_DISTANCE_PERCENT = hparams['env_major_point_min_distant_percent']
N_OBSTACLE_RECT = hparams['env_num_rect_obstacle']
N_OBSTACLE_CIRCLE = hparams['env_num_circle_obstacle']
NUM_GRIDS = hparams['env_num_grid']
COMM_RANGE = hparams['env_comm_range']
COLLIDE_RANGE = hparams['env_collide_range']
#自己定义的参数
env_theta = hparams['env_theta']
ZMAX = hparams['env_zmax']
ZMIN = hparams['env_zmin']
theta = np.deg2rad(env_theta)
OBS_RANGE_MAX = math.ceil(ZMAX / np.tan(theta))
SUOF = hparams['env_suofang']
num_angle_sections = 8  # 角度分区数量，可根据实际情况调整
num_radius_sections = 5  # 半径分区数量，可根据实际情况调整
# 假设这些参数原本在hparams或者其他配置文件中，现在提取出来方便灵活调整


#wireness_communication
RTH = hparams['RTH']
# non-pixel configs
MAX_NEIGHBOR_AGENTS = hparams['env_num_max_neighbor_agent']
MAX_NEIGHBOR_POI = hparams['env_num_max_neighbor_poi']
SQUARE_OBS = hparams['env_use_square_obs']
PIXEL_OBS = hparams['env_use_pixel_obs']
STATIC_POI = hparams['env_static_poi']
ENABLE_OBSTACLE = hparams['env_enable_obstacle']
OBS_TEST = hparams['env_observation_test']

rmax, rmin = wc.get_r_max_min()



@numba.njit
def cal_two_point_distance(x1, y1, x2, y2):  #计算平面两点距离
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2 ) ** 0.5


@numba.njit
def decimal2binary_arr(x, length=10): # 返回numpy数组->10位int类型的位置信息的二进制编码
    h = []
    for _ in range(length):
        h.append(x % 2)
        x = x // 2
    return np.array(h)


@numba.njit
def in_rectangle(x, y, min_x, min_y, height, width):#判断坐标(x,y)是否在矩形区域内，min_x, min_y,矩形的左下角点坐标，高，宽
    return x > min_x and x < min_x + height and y > min_y and y < min_y + width


@numba.njit
def in_circle(x, y, x_core, y_core, radius):#判断坐标(x,y)是否在圆周区域内，圆心(x_core, y_core),半径radius
    distance_to_core = ((x - x_core) ** 2 + (y - y_core) ** 2) ** 0.5
    return distance_to_core < radius




@numba.njit
def generate_n_agent_not_in_map(n):#生成不在地图map中的n个agent坐标，返回agent地图
    out = []
    for _ in range(n):
        while True:
            x = random.random() * 0.8 + 0.1#范围[0.1,0.9]
            y = random.random() * 0.8 + 0.1
            z = ZMIN/ZMAX
            if [x, y, z] not in out: #没障碍，也没agent
                out.append([x, y, z])
                break
    return np.array(out, dtype=np.float32)

@numba.njit
def clip(x, x_min, x_max):
    if x > x_max:
        return x_max
    if x < x_min:
        return x_min
    return x


def generate_poi_not_in_map(total_poi_num):
    """生成具有密度分层的POI坐标（确保总数准确）"""
    layers = [
        {'center': (0.4, 0.6), 'ratio': 0.4, 'std': 0.03},
        {'center': (0.7, 0.3), 'ratio': 0.3, 'std': 0.06},
        {'center': (0.5, 0.5), 'ratio': 0.2, 'std': 0.1},
        {'ratio': 0.1}
    ]

    all_poi = []
    remaining = total_poi_num

    # 前三个分层生成
    for layer in layers[:-1]:
        if 'center' in layer:
            num = int(round(total_poi_num * layer['ratio']))
            num = min(num, remaining)  # 确保不超过剩余数量
            valid_count = 0
            while valid_count < num:
                x = np.random.normal(layer['center'][0], layer['std'])
                y = np.random.normal(layer['center'][1], layer['std'])
                if 0 < x < 1 and 0 < y < 1:
                    all_poi.append([x, y])
                    valid_count += 1
            remaining -= num

    # 最后一层补充剩余数量
    uniform_num = max(remaining, 0)
    points = np.random.uniform(0, 1, (uniform_num, 2))
    all_poi.extend(points.tolist())

    return np.array(all_poi[:total_poi_num])  # 确保输出数量准确



class AgentUAVMBS3D(AgentGroup3D):
    def __init__(self, n_agent, pos=None):
        self.mass = 2 #kg
        super(AgentUAVMBS3D, self).__init__(n_agent, pos=pos, vel=None, mass=self.mass)
        self.n_agent = n_agent
        self.max_a = 5/(SUOF*NUM_GRIDS)  # 5m/s^2 V = V0+at
        self.max_vel = 20/(SUOF*NUM_GRIDS)  # 15m/s 速度有0,5,10,15,20.那么格子0,1,2,3,4，时长100步，边长200格 20/(SUOF*NUM_GRIDS)
        self.hover_energy_cost = 0.5
        self.move_energy_cost = 0.5
        self.energy_consumption = np.ones([n_agent]) * self.hover_energy_cost
        self.action2a = self.max_a * np.array([
            #全加速
            [1, 0, 0],  # action 0 denotes 1 unit acceleration front
            [-1, 0,0],  # back 1
            [0, 1, 0],  # rightward 2
            [0, -1, 0],  # leftward 3
            [0, 0, 1], # upward 4
            [0, 0, -1],  # downward 5

            #半加速
            [1/2, 0, 0],  # action 0 denotes 1 unit force front 6
            [-1/2, 0, 0],  # back 7
            [0, 1/2, 0],  # rightward 8
            [0, -1/2, 0],  # leftward 9
            [0, 0, 1/2],  # upward 10
            [0, 0, -1/2],  # downward 11
            [0, 0, 0]  # no throttle action 12
        ])
        self.n_action = self.action2a.shape[0]

    def deploy_action(self, actions, time_step=1):
        assert isinstance(actions, np.ndarray)
        actions = actions.reshape([-1]).astype(int)
        assert actions.size == self.n_agent
        acceleration = self.action2a[actions]  # [n_agent, 3] 现在是三维向量
        self.vel = self.vel + acceleration * time_step  # v=v0+at
        vel_norm = np.linalg.norm(self.vel, axis=-1, keepdims=True)
        vel_less_than_threshold_mask = np.array(vel_norm < self.max_vel, dtype=np.float32)
        vel_greater_than_threshold_mask = np.array(vel_norm > self.max_vel, dtype=np.float32)
        vel_greater_than_threshold_mask = vel_greater_than_threshold_mask * (
                self.max_vel * np.ones_like(vel_norm) / (vel_norm + 1e-5))
        clip_vel_mask = vel_less_than_threshold_mask + vel_greater_than_threshold_mask
        self.vel = self.vel * clip_vel_mask
        clipped_vel_norm = (self.vel ** 2).sum(axis=-1, keepdims=True) ** 0.5
        self.energy_consumption = np.ones([self.n_agent]) * self.hover_energy_cost + (
                    clipped_vel_norm / self.max_vel * self.move_energy_cost).reshape([-1])
        collision_penalty = np.zeros([self.n_agent], dtype=np.float32)
        # previous_pos = self.pos
        self.pos = self.pos + self.vel * time_step
        for i, (x_i, y_i, z_i) in enumerate(self.pos):
            if x_i > 1 or x_i < 0 or y_i > 1 or y_i < 0 or z_i > 1 or z_i < ZMIN/ZMAX:
                # self.pos[i] = previous_pos[i]
                collision_penalty[i] = 1
        self.pos[:, :2] = np.clip(self.pos[:, :2], 0, 1)
        # 对第三列（索引为2，即后一列）进行裁剪，范围是0.5到1
        self.pos[:, 2] = np.clip(self.pos[:, 2], ZMIN / ZMAX, 1)
        return collision_penalty

class ContinuousMCS3D:
    def __init__(self, n_agent=N_AGENT, world_scale_2D=(NUM_GRIDS, NUM_GRIDS),
                 world_scale_3D=(NUM_GRIDS, NUM_GRIDS, ZMAX-ZMIN), heuristic_reward=True):

        self.cov_ability = None
        self.world_scale_2D = np.array(world_scale_2D, dtype=np.int32)
        self.world_scale_3D = np.array(world_scale_3D, dtype=np.int32)
        self.n_agent = n_agent
        self.pixel_obs = PIXEL_OBS
        self.static_poi = STATIC_POI
        self.square_obs = SQUARE_OBS
        self.heuristic_reward = heuristic_reward
        self.obs_test = OBS_TEST  # For Observation Testing Case
        self.agents = None
        self.agent_map = np.zeros(world_scale_3D)
        self.num_major_points = N_MAJOR
        self.num_poi = hparams['env_num_poi']
        self.major_point_pos = np.zeros([self.num_major_points, 2])
        self.poi_pos = np.zeros([self.num_poi, 2])
        self.poi_map = np.zeros(world_scale_2D)
        self.poi_vel = np.zeros((self.num_poi, 2))


        # self.max_local_indices_len = 0
        # for offset_x in range(-OBS_RANGE_MAX, OBS_RANGE_MAX):#边长2*obs_range的方形区域
        #     for offset_y in range(-OBS_RANGE_MAX, OBS_RANGE_MAX):
        #         if (offset_x ** 2 + offset_y ** 2) ** 0.5 <= OBS_RANGE_MAX:
        #             self.max_local_indices_len +=1

        self.obs_dim = (num_angle_sections * num_radius_sections) + 1 + + 5 + 5 + 3


        self.act_dim = None

        # self.distance_mat = np.ones((N_AGENT, N_AGENT))
        self.comm_distance_mat = np.ones((N_AGENT, N_AGENT))
        self.collide_adj = np.ones((N_AGENT, N_AGENT))
        # 初始化用于记录上一时刻相关指标的变量
        # self.prev_c = 0.0  # 上一时刻覆盖比例
        # self.prev_f = 0.0  # 上一时刻公平指数
        # self.prev_energy = 0.0  # 上一时刻能耗平均值
        self.sum_r_i_pre = [0]*N_AGENT

        self.poi_coverage_history = []
        self.energy_consumption_history = []
        self.num_neighbor_history = []
        self.sum_r_history = []
        self.collide_agent_num_history = []
        self.timeslot = 0
        self.reset()
        self.renderer = None

    def reset(self):
        self.num_poi = hparams['env_num_poi']
        self.timeslot = 0
        self.rcov = np.zeros(N_AGENT, dtype=np.float32)
        self.robs = np.zeros(N_AGENT, dtype=np.float32)
        self.agent_cover = np.ones(N_AGENT, dtype=np.float32) #每个智能体覆盖到多少poi
        self.individual_pre = [0]
        self.group_pre = [0]
        self.delta_list = np.zeros((N_AGENT, self.num_poi), dtype=int) # [n_agent,n_poi]agent和poi的关系
        self.local_indices_x = [0]*N_AGENT
        self.local_indices_y = [0]*N_AGENT
        self.agent_pos_xd = np.full((self.n_agent, self.n_agent), 1, dtype=np.float32)
        self.interference = np.zeros(N_AGENT, dtype=np.float32)
        self.gain = np.zeros(N_AGENT, dtype=np.float32)
        self.max_r = np.zeros(N_AGENT)  # 新增的总速率存储
        # self.history_pos = []  # 用于存储每一时刻的self.pos_agent
        # self.pre_act = np.fill([self.n_agent], -1)
        self.poi_coverage_rate = np.zeros(self.num_poi, dtype=np.float32)

        poi_pos = generate_poi_not_in_map(total_poi_num=self.num_poi)
        self.poi_pos = poi_pos
        self.refresh_poi_map()

        agent_pos = generate_n_agent_not_in_map(self.n_agent)
        self.agents = AgentUAVMBS3D(self.n_agent, agent_pos)
        self.act_dim = self.agents.n_action
        self.refresh_agent_map()
        # POI 0表示没有东西，1表示被观测，2表示被覆盖，

        self.poi_coverage_history = []
        self.energy_consumption_history = []
        self.num_neighbor_history = []
        self.sum_r_history = []
        self.collide_agent_num_history = []

        self.get_robs_rcov()
        delta_list = self.delta_list

        self.agent_cover = np.sum(delta_list == 2, axis=1)
        self.agent_obverse = np.sum(delta_list == 1, axis=1) #每个智能体纯观测不覆盖到多少poi
        self.poi_is_covered = np.sum(self.delta_list == 2, axis=0)

        z_values = np.arange(ZMIN, ZMAX+1, 1)
        self.cov_ability = wc.get_ability(z_values)
        self.cov_ability = np.array(self.cov_ability)
        self.ability_utilization_ratio = np.zeros(N_AGENT)

        # 初始化用户速度，初始速度为 0
        self.poi_vel = np.zeros((self.num_poi, 2))
        # 最大速度
        self.max_poi_speed = hparams['max_poi_speed']/(SUOF * NUM_GRIDS)
        adj = self.get_adj() #无人机的邻接矩阵 adj.shape=(agent_num,agent_num)
        obs = self.get_obs() #无人机的观测矩阵 obs.shape=(agent_num,obs_dim)
        return obs, adj

    def refresh_poi_map(self):
        self.poi_map = np.zeros(self.world_scale_2D, dtype=np.int32)
        x_pois = np.clip(self.poi_pos[:, 0] * NUM_GRIDS, 0, NUM_GRIDS - 1).astype(#地图索引范围[OBS_RANGE, NUM_GRIDS + OBS_RANGE - 1]
            np.int32)
        y_pois = np.clip(self.poi_pos[:, 1] * NUM_GRIDS, 0, NUM_GRIDS - 1).astype(
            np.int32)
        for x, y in zip(x_pois, y_pois):
            self.poi_map[x, y] += 1 #每个位置上 poi 的数量情况
        self.pos_poi = np.concatenate([x_pois.reshape([self.num_poi, 1]),
                                   y_pois.reshape([self.num_poi, 1])], axis=-1)# [n_poi, 2]

    def refresh_agent_map(self):
        self.agent_map = np.zeros(self.world_scale_3D, dtype=np.int32)
        x_agents = np.clip(self.agents.x * NUM_GRIDS, 0, NUM_GRIDS - 1).astype(
            np.int32)
        y_agents = np.clip(self.agents.y * NUM_GRIDS, 0, NUM_GRIDS - 1).astype(
            np.int32)
        z_agents = np.clip(self.agents.z * ZMAX, ZMIN, ZMAX).astype(
            np.int32)
        for x, y, z in zip(x_agents, y_agents, z_agents):
            self.agent_map[x, y, z-ZMIN-1] += 1 #agent不会出现在同一位置
        self.pos_agent = np.concatenate([x_agents.reshape([-1, 1]),
                                     y_agents.reshape([-1, 1]),
                                     z_agents.reshape([-1, 1])], axis=-1)  # [n_agent, 3]


    def get_mazes(self):
        if not self.pixel_obs:
            print("warning: you are acquiring agent map in no-pixel-obs mode!")
        return self.poi_map, self.agent_map

    def get_pos(self):#返回poi,agent,矩形障碍物，圆形障碍物的坐标
        return self.poi_pos, self.agents.pos

    def get_obs(self):
        obs = []

        for i, (x, y, z) in enumerate(self.pos_agent):#x_agents.shape=(agent_num,) 挨个取无人机位置
            #状态编码，数值归一化

            obs_i = []

            #分区上的用户数量——>帮助无人机选择方向
            delta_pos_poi = []
            for deta_j, deta_poi in enumerate(self.delta_list[i]):
                if deta_poi == 1 or deta_poi == 2:
                    # 保存 POI 坐标及其对应的索引 deta_j
                    delta_pos_poi.append((self.pos_poi[deta_j], deta_j))

            # 初始化统计数组
            user_count_section = np.zeros((num_angle_sections, num_radius_sections))
            # 遍历每个 POI 及其索引
            for (poi_x, poi_y), deta_j in delta_pos_poi:
                dx = poi_x - x  # 无人机当前 x 坐标
                dy = poi_y - y  # 无人机当前 y 坐标
                distance = math.sqrt(dx ** 2 + dy ** 2)

                # 计算极角并调整到 [0, 2π)
                angle = math.atan2(dy, dx)
                if angle < 0:
                    angle += 2 * math.pi

                # 计算分区索引
                angle_idx = int(angle / (2 * math.pi / num_angle_sections))
                radius_idx = int(distance / (self.robs[i] / num_radius_sections))
                radius_idx = min(radius_idx, num_radius_sections - 1)  # 确保不越界

                # 统计用户数量
                user_count_section[angle_idx, radius_idx] += 1

            if user_count_section.sum() != 0:
                user_count_section = user_count_section / user_count_section.sum()

            # 将统计结果展平并添加到观测信息
            obs_i.extend([
                user_count_section.flatten(),  # 保留原有的用户数量统计
            ])

            # obs密度
            obs_area = 0 if self.robs[i] == 0 else self.agent_obverse[i]/(math.pi*(self.robs[i]**2))
            obs_i.append([round(obs_area,4)])

            # cov密度
            cov_area = 0 if self.rcov[i] == 0 else self.agent_cover[i]/(math.pi*(self.rcov[i]**2))
            obs_i.append([round(cov_area,4)])

            # UAV Position (float [x_pos, y_pos, z_pos])
            obs_i.append([round(self.agents.x[i],4)])
            obs_i.append([round(self.agents.y[i],4)])
            obs_i.append([round(self.agents.z[i],4)])

            # UAV Velocity (float [x_vel, y_vel, z_vel])有-1表方向，归一化了的[-1,1]，好像是小数，还是数组？ array([0., 0., 0.], dtype=float32) 3
            vel = self.agents.vel[i] / self.agents.max_vel #坐标比例 vel.shape=(3,)
            obs_i.append(vel)

            # UAV energy
            guiyi_energy = (self.agents.energy_consumption[i]-self.agents.hover_energy_cost)/(1-self.agents.hover_energy_cost)
            obs_i.append([guiyi_energy])

            # UAV ID (binary)
            binary_id = np.array(decimal2binary_arr(int(i), length=5))#binary_id.shape=(5,) 最多63个无人机
            obs_i.append(binary_id)

            # # 通信范围内与其他无人机归一化相对距离N_AGENTS self.uav_dis[i]——>避免干扰，增大相对距离
            # for j, pos_xd_i in enumerate(self.comm_distance_mat[i]):
            #     obs_i.append([round(pos_xd_i, 4)])

            # print('obs_i',obs_i)
            obs_i = np.concatenate(obs_i)
            obs.append(obs_i)
        obs = np.array(obs)

        return obs.reshape([self.n_agent, self.obs_dim])

    def get_adj(self):
        """
        获取基于信道感知的邻接矩阵，考虑空对空信道特性和中断概率。
        """
        # 获取归一化位置并计算三维距离矩阵
        pos_agent_i = self.agents.pos.reshape([self.n_agent, 1, 3]).repeat(self.n_agent, axis=1)
        pos_agent_j = self.agents.pos.reshape([1, self.n_agent, 3]).repeat(self.n_agent, axis=0)
        env_scale = NUM_GRIDS * SUOF
        distance_mat = np.sqrt(((pos_agent_i - pos_agent_j) ** 2).sum(axis=-1)) * env_scale
        distance_mat = np.maximum(distance_mat, 1e-10)

        # 信道参数和SNR计算（保持原有模型不变）
        FC = 2e9
        C = 3e8
        PT = 1
        N0 = 1e-16
        B = 1e6
        U_LOS = 11.55

        with np.errstate(divide='ignore'):
            fspl = 20 * np.log10(4 * np.pi * FC * distance_mat / C)
        pl_dB = fspl + U_LOS

        pr = PT / (10 ** (pl_dB / 10))
        noise_power = N0 * B
        snr = pr / noise_power

        # 生成基础邻接矩阵（SNR阈值判断）
        original_adj = (snr >= 10).astype(np.float32)
        outage_prob = hparams['outage_probability']

        # 应用中断概率逻辑
        if outage_prob > 0:
            # 生成随机掩码，概率性断开连接
            connection_prob = 1 - outage_prob
            mask = (np.random.rand(*original_adj.shape) < connection_prob).astype(np.float32)
            adj = original_adj * mask
        else:
            adj = original_adj.copy()

        # 强制保证自连接
        np.fill_diagonal(adj, 1)

        # 碰撞检测矩阵（保持原有逻辑）
        self.collide_adj = (distance_mat <= COLLIDE_RANGE * SUOF).astype(np.float32)
        np.fill_diagonal(self.collide_adj, 0)
        return adj.reshape([self.n_agent, self.n_agent])
    # 用户分配算法-优先级竞争机制
    def get_robs_rcov(self):
        agent_pos = np.concatenate([(self.agents.x * NUM_GRIDS).reshape([-1, 1]),
                                    (self.agents.y * NUM_GRIDS).reshape([-1, 1]),
                                    (self.agents.z * ZMAX).reshape([-1, 1])], axis=-1)  # [n_agent, 3]
        poi_pos = self.poi_pos * NUM_GRIDS

        robs = np.round(agent_pos[:, 2] / np.tan(theta)).astype(int)
        rth = RTH

        # 初始化数据结构
        delta_list = np.zeros((N_AGENT, self.num_poi))
        is_already_covered = [False] * self.num_poi
        cov_counts = np.zeros(N_AGENT, dtype=int)  # 每个无人机的覆盖计数

        # 预计算所有距离矩阵（无人机-POI）
        dx = agent_pos[:, 0, np.newaxis] - poi_pos[:, 0]
        dy = agent_pos[:, 1, np.newaxis] - poi_pos[:, 1]
        dist_matrix = np.sqrt(dx ** 2 + dy ** 2)

        # 主处理循环：按POI选择最优无人机
        for m in range(self.num_poi):
            if is_already_covered[m]:
                continue
            best_i = -1
            best_r = -np.inf

            # 寻找能覆盖当前POI且速率最大的无人机
            for i in range(N_AGENT):
                if dist_matrix[i, m] > robs[i]:
                    continue  # 不在观测范围内

                # 计算若选中此无人机时的覆盖数（当前覆盖数+1）
                current_cov_num = cov_counts[i] + 1
                r, _, _ = wc.get_r(
                    cov_num=current_cov_num,
                    dist_i_m=dist_matrix[i, m],
                    pos_agent_z=agent_pos[i][2],
                )
                if r >= rth and r > best_r:
                    best_r = r
                    best_i = i

            # 更新关联状态
            if best_i != -1:
                delta_list[best_i, m] = 2
                cov_counts[best_i] += 1
                is_already_covered[m] = True
                # 其他观测到此POI的无人机设为1
                for i in range(N_AGENT):
                    if i != best_i and dist_matrix[i, m] <= robs[i]:
                        delta_list[i, m] = 1
            else:
                # 无满足条件的无人机，仅标记观测
                for i in range(N_AGENT):
                    if dist_matrix[i, m] <= robs[i]:
                        delta_list[i, m] = 1

        # 计算各无人机的覆盖半径rcov
        for i in range(N_AGENT):
            covered_pois = np.where(delta_list[i] == 2)[0]
            if covered_pois.size > 0:
                self.rcov[i] = np.max(dist_matrix[i, covered_pois])
            else:
                self.rcov[i] = 0.0

        # 计算总速率和损耗
        for i in range(N_AGENT):
            r_i = 0.0
            for m in range(self.num_poi):
                if delta_list[i, m] == 2:
                    r, _, _ = wc.get_r(
                        cov_num=cov_counts[i],
                        dist_i_m=dist_matrix[i, m],
                        pos_agent_z=agent_pos[i][2],
                    )
                    r_i += r
                    # l_i += l_n_m
            self.max_r[i] = r_i
            # self.max_l[i] = l_i

        # 保存结果到对象属性
        self.delta_list = delta_list
        self.robs = robs

    def step(self, actions, stepout = False):
        self.rcov = [0] * N_AGENT  # [n_agent,]每个agent的覆盖范围
        self.robs = [0]*N_AGENT  # [n_agent,]每个agent观测范围
        self.delta_list = np.zeros((N_AGENT, self.num_poi), dtype=int)  # [n_agent,n_poi]agent和poi的关系

        self.gain = [0] * N_AGENT
        self.interference = [0] * N_AGENT

        # print('self.act',self.act)
        collision_penalty = self.agents.deploy_action(actions)

        self.energy_consumption_history.append(self.agents.energy_consumption)
        if self.pixel_obs:
            self.refresh_agent_map()
        adj = self.get_adj()
        adj_graph = nx.from_numpy_matrix(adj)
        sub_graphs = tuple(
            adj_graph.subgraph(c).nodes() for c in nx.connected_components(adj_graph)
        )
        reward = []
        id2group_dic = {}
        for i_group, group in enumerate(sub_graphs):
            for id in group:
                id2group_dic[id] = i_group
        self.get_robs_rcov()

        # self.delta_list = get_delta_list(self.pos_agent, self.pos_poi, self.robs, self.rcov)
        delta_list = self.delta_list
        self.poi_is_covered = np.sum(self.delta_list == 2, axis=0)
        # self.poi_is_obversed = np.sum(self.delta_list == 1, axis=0)#观测不覆盖
        self.agent_cover = np.sum(delta_list == 2, axis=1) #每个智能体覆盖到多少poi
        self.agent_obverse = np.sum(delta_list == 1, axis=1) #每个智能体纯观测不覆盖到多少poi
        self.sum_r_history.append(sum(self.max_r))
        self.poi_coverage_history.append(self.poi_is_covered)
        self.poi_coverage_rate = np.sum(self.poi_coverage_history, axis=0)/(self.timeslot+1)
        agent_cover, agent_obverse = self.agent_cover, self.agent_obverse

        if self.heuristic_reward:
            for i in range(self.n_agent):
                sum_rate_rew = 0
                i_group = id2group_dic[i]
                group = sub_graphs[i_group]
                local_covered = np.sum([agent_cover[agent_id] for agent_id in group])  # Total covered by the group
                individual_covered = agent_cover[i]
                individual_covered_rew = -1 if individual_covered == 0 else individual_covered
                group_scale = len(group)
                group_covered_rew = 0 if group_scale == 1 else 0.1 * (local_covered - individual_covered) / (group_scale - 1)

                individual_now = [k for k, value in enumerate(delta_list[i]) if value == 2]
                count = 0
                for item in individual_now:
                    if self.poi_coverage_rate[item] <= 0.6:
                        # count += 1
                        count += (1-self.poi_coverage_rate[item])
                individual_fairness_rew = count

                group_now = []
                for j in group:
                    if j == i:
                        continue
                    column = delta_list[:, j]
                    if 2 in column:
                        group_now.append(j)
                count = 0
                for item in group_now:
                    if self.poi_coverage_rate[item] <= 0.6:
                        count += (1-self.poi_coverage_rate[item])
                group_fairness_rew = 0.1 * count

                if self.max_r[i] != 0:
                    sum_rate_rew = (self.max_r[i] - rmin) / (rmax - rmin)

                rew = individual_covered_rew + group_covered_rew + individual_fairness_rew + group_fairness_rew + sum_rate_rew
                # rew = (individual_covered_rew + group_covered_rew) * f
                reward.append(rew)
        else:
            num_covered = self.poi_is_covered.sum()
            current_c = num_covered / self.num_poi
            _, f, _, _ = self.cal_episodic_coverage_and_fairness()
            global_rew = f * current_c / self.agents.energy_consumption.mean()
            reward = [global_rew] * self.n_agent

        rew = np.array(reward)
        collide_agent_num = self.collide_adj.sum(axis=1)
        rew -= (collide_agent_num + collision_penalty) * 10
        rew /= self.agents.energy_consumption

        rew = rew.reshape([self.n_agent, ])
        done = np.zeros([self.n_agent, ])

        if not self.static_poi:
            # 生成服从高斯分布的随机偏移量，与上一时刻速度相关
            noise = np.random.normal(0, 0.01, size=(self.num_poi, 2))

            # 更新速度
            self.poi_vel = self.poi_vel + noise

            # 计算速度的大小（模长）
            speed_magnitude = np.linalg.norm(self.poi_vel, axis=1)

            # 找出速度超过最大速度的用户索引
            exceed_indices = speed_magnitude > self.max_poi_speed

            # 对速度超过最大速度的用户进行处理
            if np.any(exceed_indices):
                # 归一化速度向量
                normalized_vel = self.poi_vel[exceed_indices] / speed_magnitude[exceed_indices, np.newaxis]
                # 将速度限制为最大速度
                self.poi_vel[exceed_indices] = normalized_vel * self.max_poi_speed

            # 更新位置
            self.poi_pos = self.poi_pos + self.poi_vel
            # 将位置限制在 [0, 1] 范围内
            self.poi_pos = np.clip(self.poi_pos, a_min=0, a_max=1)

            self.refresh_poi_map()
        result = ((collide_agent_num != 0) | (collision_penalty != 0)).astype(int)
        # print('collision_penalty',collision_penalty)
        # print('result',result)
        self.collide_agent_num_history.append(sum(result))
        obs = self.get_obs()
        self.sum_r_i_pre = self.max_r.copy()
        # self.individual_pre = individual_now
        # self.group_pre = group_now
        self.timeslot += 1
        if stepout:
            output = self.get_log_vars()
            return rew, obs, adj, done, output
        return rew, obs, adj, done

    def cal_episodic_coverage_and_fairness(self):
        try:
            w_t_k = np.stack(self.poi_coverage_history)  # [T, n_poi]
        except:
            # First time
            return -1, -1, -1, -1
        for i in range(1, len(w_t_k)):
            w_t_k[i] = w_t_k[i] + w_t_k[i - 1]
        max_time = w_t_k.shape[0]
        time_matrix = np.arange(1, max_time + 1).reshape([max_time, 1]).repeat(self.num_poi, axis=-1)  # [T, n_poi]
        c_t_k = w_t_k / time_matrix
        c_t = np.mean(c_t_k, axis=-1)
        final_averaged_coverage_score = c_t[-1]
        f_t = (np.sum(c_t_k, axis=-1) ** 2) / (self.num_poi * np.sum(c_t_k ** 2, axis=-1) + 1e-10)
        final_achieved_fairness_index = f_t[-1]
        return final_averaged_coverage_score, final_achieved_fairness_index, c_t, f_t

    def cal_episodic_mean_energy_consumption(self):
        return np.array(self.energy_consumption_history).mean()

    def cal_episodic_mean_num_neighbors(self):
        return np.array(self.num_neighbor_history).mean()

    def from_copy(self, other):
        for k, v in self.__dict__.items():
            self.__dict__[k] = copy.deepcopy(other.__dict__[k])

    def get_log_vars(self):
        """
        get vars to be logged in a episode
        """
        coverage_index, fairness_index, _, _ = self.cal_episodic_coverage_and_fairness()
        energy_index = self.cal_episodic_mean_energy_consumption()
        sum_rate = sum(self.sum_r_history)#bit/s
        collide = sum(self.collide_agent_num_history)
        return {'coverage_index': coverage_index, 'fairness_index': fairness_index, 'energy_index': energy_index, 'sum_rate':sum_rate,'collide':collide}


def MB_greedy_policy(env, obs, adj):
    # It is a model-based individual greedy policy.
    # 我们不联合选择动作，因为它需要对大小为 5^100 的联合动作空间进行计数We don't choose actions jointly since it need to numerate joint action space with a size of 5^100
    # For each agent choosing action, we assume other agent choose stay there, then find the action with highest reward
    actions = []
    backup_env = copy.deepcopy(env)
    for agent_i in range(env.n_agent):
        rewards_for_each_actions = np.array([-999] * env.act_dim)
        imagine_env = copy.deepcopy(backup_env)
        for action_i in range(env.act_dim):
            dummy_actions = [env.act_dim - 1] * env.n_agent  # means stop
            dummy_actions[agent_i] = action_i
            rews, _, _, _ = imagine_env.step(np.array(dummy_actions))
            imagine_env.from_copy(backup_env)
            reward = rews[agent_i]
            rewards_for_each_actions[action_i] = reward
        best_action_agent_i = rewards_for_each_actions.argmax()
        actions.append(best_action_agent_i)
    return np.array(actions)


def MB_GA_policy(env, obs, adj, max_iter=20):
    # 创建环境副本用于模拟
    backup_env = copy.deepcopy(env)
    imagine_env = copy.deepcopy(env)

    def loss_func(actions):
        # 执行动作并获取奖励（注意step返回的奖励是第一个元素）
        rew, _, _, _ = imagine_env.step(actions)
        # 重置模拟环境状态
        imagine_env.from_copy(backup_env)
        # 最小化负的总奖励
        return -sum(rew)

    # 初始化遗传算法，调整动作上限为 env.act_dim - 1
    ga = GA(func=loss_func,
            n_dim=env.n_agent,
            size_pop=50,
            max_iter=max_iter,
            lb=[0] * env.n_agent,
            ub=[env.act_dim - 1] * env.n_agent,  # 使用env.act_dim获取动作空间
            precision=1)

    # 运行遗传算法并获取最优解
    best_actions, _ = ga.run()

    return np.array(best_actions, dtype=np.int32)  # 返回整型动作数组

def suiji(env, obs, adj):
    selected_actions = np.random.randint(0, env.act_dim, env.n_agent)
    return selected_actions

def potential_field_policy(env, obs, adj):
    actions = []
    attraction_gain = 1.5  # 用户吸引力增益系数
    repulsion_gain = 0.8  # 无人机间斥力增益
    boundary_gain = 0.3  # 边界斥力增益
    height_adjust_threshold = 3  # 高度调整阈值

    for i in range(env.n_agent):
        # 获取基础状态信息
        agent_pos = env.agents.pos[i]
        current_z = agent_pos[2]

        # ================== 用户引力计算 ==================
        user_force = np.zeros(2)
        # 获取观测范围内的用户索引（覆盖和观测状态）
        observed_users = np.where(env.delta_list[i] >= 1)[0]
        if len(observed_users) > 0:
            # 计算用户群几何中心
            user_centroid = np.mean(env.poi_pos[observed_users], axis=0)
            # 计算指向用户的单位方向向量
            direction = user_centroid - agent_pos[:2]
            norm = np.linalg.norm(direction)
            if norm > 1e-5:
                user_force = direction / norm * attraction_gain

        # ================== 无人机斥力计算 ==================
        repel_force = np.zeros(2)
        # 获取通信范围内的无人机（排除自身）
        neighbors = np.where(adj[i] == 1)[0]
        neighbors = neighbors[neighbors != i]

        for j in neighbors:
            neighbor_pos = env.agents.pos[j]
            # 计算相对位置向量
            diff = agent_pos[:2] - neighbor_pos[:2]
            distance = np.linalg.norm(diff)
            if distance < 1e-5:
                continue
            # 斥力与距离平方成反比
            repel_force += (diff / distance) * repulsion_gain / (distance ** 2 + 1e-5)

        # ================== 边界斥力计算 ==================
        boundary_force = np.zeros(2)
        # X轴边界处理
        if agent_pos[0] < 0.1:
            boundary_force[0] += boundary_gain / (agent_pos[0] + 1e-5)
        elif agent_pos[0] > 0.9:
            boundary_force[0] -= boundary_gain / (1.0 - agent_pos[0] + 1e-5)
        # Y轴边界处理
        if agent_pos[1] < 0.1:
            boundary_force[1] += boundary_gain / (agent_pos[1] + 1e-5)
        elif agent_pos[1] > 0.9:
            boundary_force[1] -= boundary_gain / (1.0 - agent_pos[1] + 1e-5)

        # ================== 高度调整策略 ==================
        z_force = 0
        # 计算覆盖效能指标
        coverage_efficiency = env.agent_cover[i] / (np.pi * (env.rcov[i] ** 2) + 1e-5)
        # 效能低下时降低高度增加覆盖密度
        if coverage_efficiency < 0.4 and current_z > ZMIN / ZMAX + 0.1:
            z_force = -0.5
        # 用户数过少时主动降低高度
        elif env.agent_cover[i] < height_adjust_threshold and current_z > ZMIN / ZMAX + 0.15:
            z_force = -0.8
        # 正常情况缓慢爬升扩大覆盖
        else:
            z_force = 0.2 if current_z < 0.8 else 0

        # ================== 合力向量合成 ==================
        total_force = np.array([
            user_force[0] + repel_force[0] + boundary_force[0],
            user_force[1] + repel_force[1] + boundary_force[1],
            z_force
        ])

        # ================== 动作选择策略 ==================
        action_candidates = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1], [0.5, 0, 0], [-0.5, 0, 0],
            [0, 0.5, 0], [0, -0.5, 0], [0, 0, 0.5], [0, 0, -0.5],
            [0, 0, 0]
        ]

        # 寻找最匹配的离散动作
        best_match = 12  # 默认不动
        max_similarity = -np.inf
        for idx, acc in enumerate(action_candidates):
            similarity = np.dot(total_force, acc)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = idx

        actions.append(best_match)

    return np.array(actions, dtype=np.int32)



from sko.PSO import PSO

def APSO_policy(env, obs, adj, max_iter=15, n_particles=30):
    """
    自适应粒子群优化算法实现无人机协同覆盖
    核心特点：
    - 动态惯性权重调整策略
    - 三维速度约束机制
    - 能量效率优化目标函数
    """
    backup_env = copy.deepcopy(env)

    def fitness_func(actions_flat):
        actions = actions_flat.reshape(-1, env.n_agent).astype(int)
        total_rewards = []

        for action_set in actions:
            imagine_env = copy.deepcopy(backup_env)
            rewards, _, _, _ = imagine_env.step(action_set)
            imagine_env.from_copy(backup_env)

            energy_penalty = np.sum(imagine_env.agents.energy_consumption)
            total_rewards.append(-np.mean(rewards) + 0.3 * energy_penalty)

        return np.array(total_rewards)

    # 初始化PSO时需指定速度范围
    pso = PSO(func=fitness_func,
              dim=env.n_agent,
              pop=n_particles,
              max_iter=max_iter,
              lb=[0] * env.n_agent,
              ub=[env.act_dim - 1] * env.n_agent,
              w=0.8,
              c1=1.5,
              c2=2.0)

    # 修改参数自适应函数
    def adapt_params(particle, generation):
        # 惯性权重线性递减[1,2](@ref)
        particle.w = 0.9 - 0.5 * (generation / max_iter)
        # 学习因子非线性调整[2](@ref)
        particle.c1 = 1.5 * (1 - generation / max_iter) ** 2
        particle.c2 = 2.0 * (generation / max_iter) ** 0.5

    # 注册回调时需要指定触发时机
    pso.register(operator_name='before_iter', operator=adapt_params)  # 修正后的注册方式

    pso.run()
    best_actions = pso.gbest_x.astype(int)
    return best_actions.astype(int)


def KMeans_policy(env, obs, adj, max_iter=10):
    # 获取POI和无人机位置（确保为numpy数组）
    poi_pos = np.array(env.poi_pos, dtype=np.float32)  # [N_POI, 2]
    agent_pos = env.agents.pos[:, :2].astype(np.float32)  # [N_AGENT, 2]

    # 改进的初始中心选择（确保返回浮点数组）
    def select_initial_centroids():
        if len(poi_pos) == 0:
            return np.random.rand(env.n_agent, 2).astype(np.float32)

        density = np.zeros(len(poi_pos), dtype=np.float32)
        for i in range(len(poi_pos)):
            distances = np.linalg.norm(poi_pos - poi_pos[i], axis=1)
            density[i] = np.sum(np.exp(-(distances ** 2) / (2 * 0.1 ** 2)))

        top_poi = poi_pos[np.argsort(-density)[:min(env.n_agent * 2, len(poi_pos))]]

        centroids = []
        if len(top_poi) > 0:
            centroids.append(top_poi[0])
            for _ in range(1, env.n_agent):
                if len(centroids) >= len(top_poi):
                    break
                dists = np.min(np.linalg.norm(top_poi[:, None] - centroids, axis=2), axis=1)
                next_idx = np.argmax(dists)
                centroids.append(top_poi[next_idx])

        # 不足时用随机点补充
        while len(centroids) < env.n_agent:
            centroids.append(np.random.rand(2))
        return np.array(centroids[:env.n_agent], dtype=np.float32)

    # 初始化中心并确保数值类型
    centroids = select_initial_centroids()
    prev_centroids = np.full_like(centroids, np.inf)  # 初始化为无穷大
    iteration = 0

    while iteration < max_iter:
        # 类型检查与转换
        centroids = centroids.astype(np.float32)
        prev_centroids = prev_centroids.astype(np.float32)

        # 终止条件：中心点变化小于阈值或达到最大迭代
        if np.allclose(centroids, prev_centroids, atol=1e-4):
            break

        # 分配无人机到聚类中心
        clusters = [[] for _ in range(env.n_agent)]
        for i, pos in enumerate(env.agents.pos):
            pos_2d = pos[:2].astype(np.float32)
            dists = np.linalg.norm(centroids - pos_2d, axis=1)
            cluster_idx = np.argmin(dists)
            clusters[cluster_idx].append(i)

        # 更新聚类中心（含空簇处理）
        new_centroids = []
        for cluster in clusters:
            if len(cluster) == 0:
                new_centroids.append(centroids[np.random.choice(len(centroids))])
                continue

            cluster_pos = agent_pos[cluster]
            poi_dists = np.linalg.norm(poi_pos[:, None] - cluster_pos, axis=2)
            if poi_dists.size == 0:
                weights = np.ones(len(cluster_pos))
            else:
                min_dists = np.min(poi_dists, axis=0)
                weights = np.exp(-(min_dists ** 2) / (2 * 0.1 ** 2))

            new_center = np.average(cluster_pos, axis=0, weights=weights)
            new_centroids.append(new_center)

        prev_centroids = centroids
        centroids = np.array(new_centroids, dtype=np.float32)
        iteration += 1

    # 生成动作指令（添加边界检查）
    actions = []
    for i, agent in enumerate(env.agents.pos):
        agent_pos_2d = agent[:2].astype(np.float32)
        dists = np.linalg.norm(centroids - agent_pos_2d, axis=1)
        target_idx = np.argmin(dists)
        target = centroids[target_idx]

        # 计算方向向量（限制在[-1,1]范围）
        direction = target - agent_pos_2d
        direction = np.clip(direction, -1.0, 1.0)

        # 动作选择（添加高度调整项）
        action_vec = np.zeros(3, dtype=np.float32)
        action_vec[:2] = direction
        action_vec[2] = (ZMIN - agent[2]) / ZMAX  # 高度调整项

        # 找到最匹配的离散动作
        similarities = env.agents.action2a @ action_vec
        action = np.argmax(similarities)
        actions.append(action)

    return np.array(actions, dtype=np.int32)


# def quantize_action(cont_actions,action_dim):
#     """将连续动作空间映射到离散动作集合"""
#     # 添加高斯扰动增强探索性
#     noise = np.random.normal(0, 0.3, cont_actions.shape)
#     cont_actions = np.clip(cont_actions + noise, 0, action_dim - 1)
#     # 基于概率的柔性量化
#     prob = cont_actions - np.floor(cont_actions)
#     return np.where(np.random.rand(*cont_actions.shape) < prob,
#                     np.ceil(cont_actions),
#                     np.floor(cont_actions)).astype(int)


# from sklearn.cluster import KMeans
#
#
# def kmeans_strategy(env, obs, adj):
#     """基于k-means的无人机协同覆盖策略"""
#     # 获取环境参数
#     n_agent = env.n_agent
#     poi_pos = env.poi_pos  # 获取所有POI坐标[N_POI, 2]
#
#     # 动态调整聚类权重（根据历史覆盖频率）
#     weights = 1.0 - env.poi_coverage_rate  # 优先覆盖低覆盖率的POI
#     weights = np.clip(weights, 0.1, 1.0)  # 防止权重过小
#
#     # 执行加权k-means聚类
#     kmeans = KMeans(n_clusters=n_agent, n_init=10)
#     kmeans.fit(poi_pos, sample_weight=weights)
#
#     # 获取簇中心并添加高度信息
#     cluster_centers = kmeans.cluster_centers_
#     cluster_sizes = np.bincount(kmeans.labels_, minlength=n_agent)
#
#     # 生成目标点（包含三维坐标）
#     target_pos = np.zeros((n_agent, 3))
#     for i in range(n_agent):
#         # 根据簇密度调整高度（簇越大飞行高度越低）
#         density = cluster_sizes[i] / (np.pi * (env.rcov[i] ** 2) + 1e-5)
#         optimal_z = np.clip(ZMAX - 0.3 * (density - 5), ZMIN, ZMAX) / ZMAX  # 归一化
#         target_pos[i] = [cluster_centers[i][0], cluster_centers[i][1], optimal_z]
#
#     # 计算移动方向并选择动作
#     actions = []
#     for i in range(n_agent):
#         current_pos = env.agents.pos[i]
#         target = target_pos[i]
#
#         # 计算三维方向向量
#         direction = target - current_pos
#         dir_normalized = direction / (np.linalg.norm(direction) + 1e-5)
#
#         # 选择最接近的离散动作（考虑加速度方向）
#         action_candidates = env.agents.action2a
#         similarities = np.dot(action_candidates, dir_normalized)
#         best_action = np.argmax(similarities)
#
#         # 避障调整：检测附近无人机
#         neighbor_mask = (env.comm_distance_mat[i] < 0.2) & (env.comm_distance_mat[i] > 0)
#         if neighbor_mask.any():
#             repulsion = np.mean(env.agents.pos[neighbor_mask] - current_pos, axis=0)
#             repulsion_dir = repulsion / (np.linalg.norm(repulsion) + 1e-5)
#             best_action = select_avoidance_action(best_action, repulsion_dir, env.agents.action2a)
#
#         actions.append(best_action)
#
#     return np.array(actions)
#
#
# def select_avoidance_action(original_action, repulsion_dir, action_set):
#     """避障动作选择策略"""
#     # 计算排斥方向与各动作的相似度
#     similarities = np.dot(action_set, repulsion_dir)
#     avoidance_action = np.argmax(similarities)
#
#     # 在原始动作和避障动作间加权选择
#     return avoidance_action if np.random.rand() < 0.7 else original_action
#
#
# def dynamic_height_adjustment(cluster_size, current_rcov):
#     """根据簇大小和当前覆盖半径动态调整高度"""
#     # 覆盖半径与高度的关系模型（可根据实际调整）
#     target_rcov = np.sqrt(cluster_size / np.pi) * 0.1  # 假设每单位面积需要覆盖0.1个POI
#     optimal_z = target_rcov * np.tan(theta)  # 根据覆盖半径反推高度
#     return np.clip(optimal_z, ZMIN, ZMAX) / ZMAX  # 归一化

