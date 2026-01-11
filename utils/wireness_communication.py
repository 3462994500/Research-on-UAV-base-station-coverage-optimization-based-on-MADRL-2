from utils.hparams import hparams
import math
import scipy.constants as constant
import numpy as np

# SUOF = hparams['env_suofang']
# env_theta = hparams['env_theta']
# env_theta = np.deg2rad(env_theta)
# rth = hparams['RTH']
# ZMAX = hparams['env_zmax']
# ZMIN = hparams['env_zmin']

SUOF = 10
env_theta = np.deg2rad(60)
rth = 200000
ZMAX = 5
ZMIN = 15

#number of UAVs
# N_AGENT = hparams['env_num_agent']
# N_POI = hparams['env_num_poi']


# Carrier Frequency 2G Hz
FC = 2e9

# Speed of Light  3 * 10^8 m/s
C = constant.c

# Power Spectral Density 10-16 W/Hz
N0 = 1e-16

# Bandwidth 1mHz
B = 1e6

#Transmission power of an UAV 1w
PT = 1

# Noise
STD = B * N0
# STD = 1e-11

# Environmental Variable 1
a = 12.08

# Environmental Variable 2
b = 0.11

# Additional path loss for LineOfSight dB #U_LOS = 10**(3/10)
U_LOS = 3
# Additional path loss for NonLineOfSight
U_NLOS = 23


def calculate_channel_gain( r, z ):
    """
    计算信道增益相关参数。
    根据给定的banjin距离 r 和高度 z 等参数，按照信道增益的计算公式进行计算并返回结果。
    """
    # print(z)
    r = r * SUOF
    z = z * SUOF
    d = math.sqrt(r ** 2 + z ** 2)

    lfs = 20 * np.log10(4 * np.pi * FC * d / C)
    Llos = lfs + U_LOS  # dB
    Lnlos = lfs + U_NLOS
    theta = np.arcsin(z / d)
    # print(theta)
    Plos = 1 / (1 + a * np.exp(-b * ((180 / np.pi) * theta - a)))
    Pnlos = 1 - Plos
    l_n_m = Plos * Llos + Pnlos * Lnlos  # dB
    g = (10) ** -(l_n_m / 10)
    return g,l_n_m


def get_r(cov_num, dist_i_m, pos_agent_z):
    """
    更新后的速率计算函数
    n: 当前无人机索引
    cov_num: 当前覆盖的POI数量
    rcov: 当前测试的覆盖半径
    pos_agent: 所有无人机位置数组
    pos_poi_m: 当前测试的POI位置
    k_list: 各无人机当前覆盖的POI数量列表
    PT: 发射功率数组
    """
    # 计算信道增益
    g_n_m,l_n_m = calculate_channel_gain(dist_i_m, pos_agent_z)  # 使用无人机高度

    # 计算干扰
    # I = 0
    # agent_num = len(pos_agent)
    # for j in range(agent_num):
    #     if j == n:
    #         continue
    #     if k_list[j] == 0:
    #         continue

        # # 计算无人机间距离
        # d = np.sqrt(
        #     (pos_poi_m[0] - pos_agent[j][0]) ** 2 +
        #     (pos_poi_m[1] - pos_agent[j][1]) ** 2
        # )

        # # 计算干扰距离
        # g_i = calculate_channel_gain(d, pos_agent[j][2])
        # I += (PT[j] / k_list[j]) * g_i  # 防止除以零

    # 计算SINR
    B_t_n = B / cov_num  # 带宽分配
    pr = (PT / cov_num) * g_n_m
    # SINR = pr / (I + STD)
    SNR = pr / (STD)

    # 计算速率
    r = B_t_n * np.log2(1 + SNR)
    return r, g_n_m,l_n_m

def get_ability(z_values):
    max_cov_num_list = []
    for z in z_values:
        # 初始化最大 cov_num
        max_cov_num = 0
        # 无人机坐标
        # dist_i_m = z / np.tan(env_theta)#最小覆盖能力
        dist_i_m = 0  #最大覆盖能力
        # 遍历 cov_num 从 1 到某个最大值（例如 100）
        for cov_num in range(1, 101):
            # pos_poi_m = [10, 10]
            # 计算 r
            r, _, _ = get_r(cov_num, dist_i_m, z)
            flag = 0
            # 如果 r 大于 1500，更新最大 cov_num
            if r > rth:
                max_cov_num = cov_num
                flag = 1
            if flag == 0:
                break

        # 将最大 cov_num 添加到列表中
        max_cov_num_list.append(max_cov_num)

    return max_cov_num_list

def get_r_max_min():
    rmax, _, _ = get_r(1, 0, ZMIN)
    dist_i_m = ZMAX / np.tan(env_theta)
    rmin = 0
    for cov_num in range(1, 101):
        # pos_poi_m = [10, 10]
        # 计算 r
        r, _, _ = get_r(cov_num, dist_i_m, ZMAX)
        flag = 0
        # 如果 r 大于 1500，更新最大 cov_num
        if r > rth:
            rmin = r
            flag = 1
        if flag == 0:
            break
    return rmax, rmin

# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.ticker import MaxNLocator
# from matplotlib.font_manager import FontProperties

# # 定义Euclid字体路径（用户指定路径）
# euclid_font_path = r'C:\Windows\Fonts\euclid.ttf'  # Euclid字体
# # 定义宋体路径
# simsun_font_path = r'C:\Windows\Fonts\simsun.ttc'  # 宋体

# # 创建中文字体属性（宋体，25号）
# cn_font = FontProperties(fname=simsun_font_path, size=15)
# # 创建Euclid字体属性（8号，用于刻度等外文文本）
# euclid_font = FontProperties(fname=euclid_font_path, size=15)

# # 设置学术风格参数
# plt.style.use('seaborn-whitegrid')
# plt.rcParams.update({
#     'figure.dpi': 300,
#     'axes.linewidth': 0.8,
#     'grid.color': '#DDDDDD',
#     'axes.unicode_minus': False,  # 处理负号显示
#     "mathtext.fontset": 'custom',  # 使用自定义字体渲染数学公式
#     "mathtext.rm": euclid_font.get_name(),  # 数学公式中的常规文本
#     "mathtext.it": euclid_font.get_name() + ':italic',  # 数学公式中的斜体文本
#     "mathtext.bf": euclid_font.get_name() + ':bold',  # 数学公式中的粗体文本
# })

# # 创建画布
# fig, ax = plt.subplots(figsize=(6, 4))

# # 定义 z 的范围（假设env_theta已在其他地方定义）
# z_values = np.arange(5, 16, 1)
# OBS_RANGE = z_values / np.tan(env_theta)

# # 遍历 z 值（假设get_ability函数已定义）
# max_cov_num_list = get_ability(z_values)
# z_values = z_values * SUOF  # 假设SUOF已定义

# print('z_values', z_values)
# print('R', OBS_RANGE * 10)
# print('max_cov_num_list', max_cov_num_list)

# # 坐标轴边界计算
# x_pad = 5  # 横坐标左右边距
# y_pad = 1  # 纵坐标上下边距
# x_min = z_values.min() - x_pad
# x_max = z_values.max() + x_pad
# y_min = 6  # 从0开始
# y_max = max(max_cov_num_list) + y_pad

# # 绘制主曲线
# line = ax.plot(z_values, max_cov_num_list,
#               marker='D',
#               markersize=7,
#               markerfacecolor='white',
#               markeredgecolor='#1f77b4',
#               markeredgewidth=1.2,
#               linewidth=1.8,
#               color='#1f77b4',
#               linestyle='-',
#               label='Pt=1w')

# # 坐标轴设置（中文使用宋体）
# ax.set_xlabel('无人机高度（米）', fontproperties=cn_font)
# ax.set_ylabel('最大覆盖能力（个）', fontproperties=cn_font)

# # 强制纵坐标为整数
# ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# # 横坐标刻度间隔设置
# ax.xaxis.set_major_locator(plt.MultipleLocator(10))  # 每20米一个主刻度

# # 网格线设置
# ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.8)
# ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.5)

# # 边框控制
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['left'].set_position(('outward', 5))
# ax.spines['bottom'].set_position(('outward', 5))

# # # 数据标签优化（使用与刻度一致的字体大小）
# # for x, y in zip(z_values, max_cov_num_list):
# #     offset = 0.6 if y == max(max_cov_num_list) else 0.8  # 最高点标签下移
# #     ax.text(x, y + offset, f'{int(y)}',
# #            ha='center', va='bottom',
# #            fontproperties=euclid_font,  # 显式指定Euclid字体
# #            fontsize=15,  # 统一字体大小为20
# #            color='#1f77b4',
# #            bbox=dict(boxstyle='round,pad=0.2',
# #                     facecolor='white',
# #                     edgecolor='lightgray',
# #                     linewidth=0.5))

# # 图例设置（带下标，中文使用宋体）
# legend = ax.legend(
#     [f'发射功率$P_T$:1w'],  # 使用数学表达式实现下标
#     loc='upper right',
#     frameon=True,
#     framealpha=0.95,
#     prop=cn_font,
#     handlelength=0  # 隐藏图例图标
# )
# legend.get_title().set_fontweight('normal')

# # 设置坐标轴刻度字体为Euclid
# plt.xticks(fontproperties=euclid_font)
# plt.yticks(fontproperties=euclid_font)

# # 紧凑布局
# plt.tight_layout()

# # 保存输出
# plt.savefig('图8 无人机覆盖能力随高度变化曲线.pdf', bbox_inches='tight')
# plt.savefig('图8 无人机覆盖能力随高度变化曲线.png', bbox_inches='tight')
# plt.show()
# ##########################################

# # 绘制折线图
# z_values = z_values * SUOF
# plt.plot(z_values, max_cov_num_list, marker='o', color='blue', label='Pt=1w')
#
# # 添加图例（tabel）
# plt.legend(title='', loc='upper left', bbox_to_anchor=(0.75, 1), frameon=True, shadow=True, fancybox=True)
# plt.xlabel('z (m)')
# plt.ylabel('Max cov_num')
# plt.title('Max cov_num vs z')
# plt.grid(True)
# plt.show()

# def get_poi_r(agent_cover, pos_agent, pos_poi, delta_list, pt):
#     """
#       Calculate rate for uav and user
#     """
#     # print('pos_agent', pos_agent) 没归一化
#     # print('pos_poi', pos_poi) 没归一化
#     agent_num = len(pos_agent)
#     poi_num = len(pos_poi)
#     r_agent = [0] * agent_num
#     i_agent = [0] * agent_num
#     i_poi = [0] * poi_num
#     sinr_poi = [0] * poi_num
#
#     for agent_i, agent_poi in enumerate(delta_list):
#         r_i = 0
#         i_i = 0
#         for poi_j, lianxi in enumerate(agent_poi):
#             i_j = 0
#             sinr = 0
#             if lianxi == 2:
#                 # Calculate channel_gain between uav and user caused by other uavs
#                 l = math.sqrt(((pos_poi[poi_j][0] - pos_agent[agent_i][0]) * SUOF) ** 2 + ((pos_poi[poi_j][1] - pos_agent[agent_i][1]) * SUOF) ** 2)
#                 g_n_m = calculate_channel_gain(l, pos_agent[agent_i][-1])
#
#                 # Calculate interference between uav and user caused by other uavs
#                 i_j_o = 0
#                 agent_num = len(pos_agent)
#                 for other_uav_index in range(agent_num):
#                     if other_uav_index == agent_i:
#                         continue
#                     k_i = agent_cover[other_uav_index]
#                     if k_i == 0:
#                         continue
#                     else:
#                         x = math.sqrt(((pos_poi[poi_j][0]-pos_agent[other_uav_index][0])*SUOF)**2 + ((pos_poi[poi_j][1]-pos_agent[other_uav_index][1])*SUOF)**2)
#                         g_i = calculate_channel_gain( x, pos_agent[other_uav_index][-1] )
#                         i_j_o += (pt[other_uav_index] / k_i) * g_i
#                 i_j = i_j_o
#                 b_t_n = B / agent_cover[agent_i]
#                 pr = (pt[agent_i]/agent_cover[agent_i]) * g_n_m
#                 sinr = pr / (i_j + STD)
#                 # if SINR < 0:
#                 #     print("SINR的值:", SINR,pr,I,cov_num,g_n_m)
#                 r_i += b_t_n * math.log2(1 + sinr)
#
#             # print('poi_j', poi_j)
#             i_poi[poi_j] = i_j
#             i_i += i_j
#             sinr_poi[poi_j] = sinr
#         r_agent[agent_i] = r_i
#         i_agent[agent_i] = i_i
#
#     return r_agent, i_agent, i_poi, sinr_poi
#
#


#test
# pos_agent = np.array([[1, 1,5], [1, 1,6], [1, 1,7],[1, 1,8],[1, 1,9],[1, 1,10],[1, 1,11],[1, 1,12],[1, 1,13],[1, 1,14],[1, 1,15],[1, 1,16],[1, 1,17],[1, 1,18],[1, 1,19],[1, 1,20]])
# # print(pos_agent)
# cov_num = 6
#
# for n in range(16):
#     print('n',n)
#     # pos_poi_m = [1,1]
#     r, G,l_n_m = get_r(cov_num, 0, pos_agent[n][2])
#     print(r*cov_num,G,l_n_m)

# # First calculate all channel gains and transmit powers of all combination of uav and users
# k_list = []#[N,1] 无人机的关联用户数量
# transpowers = []#[N,2]    无人机的[发射功率,关联用户数量]
# channel_gain_matrix = []#[N,M]      无人机与用户之间的信道增益
#
# for uav_index in range(N):
#     k_n = calculate_k(uav_index)
#     k_list.append(k_n)
#     pn = 0 if k_n == 0 else PT
#     transpowers.append(pn)
#     channel_gain_list = [calculate_channel_gain(uav_index, user_index) \
#                          for user_index in range(M)]
#     channel_gain_matrix.append(channel_gain_list)
#
#
# print(k_list,transpowers,channel_gain_matrix,get_r(0,0))

