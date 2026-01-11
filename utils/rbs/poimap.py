# import matplotlib.pyplot as plt
# import numpy as np
#
# # 地图大小
# map_size = 50
#
# # 存储圆的圆心和半径
# circles = []
#
# # 获取用户输入的圆心坐标和半径
# while True:
#     user_input = input("请输入圆心的 x 坐标 (0 - 49) 或输入 'end' 结束: ")
#     if user_input.lower() == 'end':
#         break
#     center_x = int(user_input)
#     center_y = int(input("请输入圆心的 y 坐标 (0 - 49): "))
#     GZ = int(input("请输入Z: "))
#     radius = GZ / np.tan(np.radians(60))
#     circles.append((center_x, center_y, radius))
#
# # 创建地图
# fig, ax = plt.subplots()
# ax.set_xlim(0, map_size)
# ax.set_ylim(0, map_size)
# ax.set_aspect('equal')
#
# # 绘制所有圆
# for circle_info in circles:
#     center_x, center_y, radius = circle_info
#     circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
#     ax.add_artist(circle)
#
# # 归一化坐标显示函数
# def on_click(event):
#     if event.xdata is not None and event.ydata is not None:
#         x_norm = event.xdata / map_size
#         y_norm = event.ydata / map_size
#         print(f" [{x_norm:.8f}, {y_norm:.8f}],")
#
# # 绑定点击事件
# fig.canvas.mpl_connect('button_press_event', on_click)
#
# plt.show()


import matplotlib.pyplot as plt
import numpy as np

# # 地图大小
# map_size = 50
#
# # 获取用户输入的圆心坐标和半径
# center_x = int(input("请输入圆心的 x 坐标 (0 - 49): "))
# center_y = int(input("请输入圆心的 y 坐标 (0 - 49): "))
# GZ = int(input("请输入Z: "))
# radius = GZ / np.tan(np.radians(60))
# # 创建地图
# fig, ax = plt.subplots()
# ax.set_xlim(0, map_size)
# ax.set_ylim(0, map_size)
# ax.set_aspect('equal')
#
# # 绘制圆
# circle = plt.Circle((center_x, center_y), radius, color='r', fill=False)
# ax.add_artist(circle)
#
# # 归一化坐标显示函数
# def on_click(event):
#     if event.xdata is not None and event.ydata is not None:
#         x_norm = event.xdata / map_size
#         y_norm = event.ydata / map_size
#         print(f" [{x_norm:.8f}, {y_norm:.8f}],")
#
# # 绑定点击事件
# fig.canvas.mpl_connect('button_press_event', on_click)
#
# plt.show()

# import numpy as np
# import random
# import matplotlib.pyplot as plt

# 设置随机种子以保证结果可重复
# np.random.seed(42)
# random.seed(42)

# 用户地图，随机密度
# import numpy as np
# import matplotlib.pyplot as plt
#
# def generate_poi_not_in_map(num_main_points, total_poi_num):
#     # 生成归一化的主要点坐标
#     main_points_x = np.random.uniform(0.2, 0.8, num_main_points)
#     main_points_y = np.random.uniform(0.2, 0.8, num_main_points)
#     main_points = np.column_stack((main_points_x, main_points_y))
#
#     # 随机分配 POI 数量给每个主要点
#     poi_counts = np.random.multinomial(total_poi_num, np.ones(num_main_points) / num_main_points)
#     all_poi_points = []
#     # 为每个主要点随机设置不同的标准差，控制散开程度
#     std_devs = np.random.uniform(0.05, 0.25, num_main_points)  # 这里调整标准差范围，因为是归一化坐标
#     for i, point in enumerate(main_points):
#         num_poi = poi_counts[i]
#         std_dev = std_devs[i]
#         for _ in range(num_poi):
#             while True:
#                 poi_x = np.random.normal(point[0], std_dev)
#                 poi_y = np.random.normal(point[1], std_dev)
#                 if 0.02 < poi_x < 1 and 0 < poi_y < 0.98:
#                     all_poi_points.append([poi_x, poi_y])
#                     break
#     all_poi_points = np.array(all_poi_points)
#     return main_points, all_poi_points
#
#
# def generate_3d_aware_users(total_users, map_size=100):
#     """
#     生成具有三维空间密度特征的用户分布
#     参数：
#     total_users : 总用户数
#     map_size : 地图尺寸(米)
#
#     返回：
#     users : 用户坐标数组[[x,y],...] (0-map_size)
#     """
#     np.random.seed(42)
#
#     # 创建密度梯度网格
#     x = np.linspace(0, map_size, 100)
#     y = np.linspace(0, map_size, 100)
#     X, Y = np.meshgrid(x, y)
#
#     # 生成三组空间密度模式
#     density = (np.exp(-((X - 30) ** 2 + (Y - 30) ** 2) / 800) * 0.4 +  # 高密度区需要低空覆盖
#                np.exp(-((X - 70) ** 2 + (Y - 70) ** 2) / 1800) * 0.3 +  # 中密度区
#                0.3 / (1 + np.exp(-(X + Y) / 50)))  # 线性梯度区
#
#     # 生成用户位置
#     prob = density.flatten()
#     prob /= prob.sum()  # 归一化概率
#
#     indices = np.random.choice(np.arange(len(prob)), size=total_users, p=prob)
#     users = np.column_stack([X.flatten()[indices],
#                              Y.flatten()[indices]]).astype(int)
#
#     # 添加随机扰动
#     users = users + np.random.uniform(-0.5, 0.5, users.shape)
#
#     return np.clip(users, 0, map_size)
#
# # 生成用户地图
# n_major = 3  # 主用户点数量
# n_poi = 100   # 用户点数量
# pos_major, poi_pos = generate_poi_not_in_map(n_major, n_poi)
#
# # 绘制用户地图
# plt.figure(figsize=(8, 8))
# plt.scatter(pos_major[:, 0], pos_major[:, 1], c='red', marker='s', label='Major Users')
# plt.scatter(poi_pos[:, 0], poi_pos[:, 1], c='blue', marker='o', label='POI Users')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('User Map')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # 生成用户地图并叠加热力图
# plt.figure(figsize=(10, 8))
#
# # 生成核密度估计热力图
# from scipy.stats import gaussian_kde
# kde = gaussian_kde(poi_pos.T)  # 转置为(2, N)格式
# xgrid = np.linspace(0, 1, 100)
# ygrid = np.linspace(0, 1, 100)
# X, Y = np.meshgrid(xgrid, ygrid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
#
# # 绘制热力图（使用半透明效果）
# heatmap = plt.imshow(Z,
#                     cmap='YlGnBu',
#                     aspect='auto',
#                     extent=[0, 1, 0, 1],
#                     origin='lower',
#                     alpha=0.6)
#
# # 添加散点图
# plt.scatter(pos_major[:, 0], pos_major[:, 1],
#            c='red', marker='s',
#            edgecolor='white',
#            s=80, label='Major Users')
# plt.scatter(poi_pos[:, 0], poi_pos[:, 1],
#            c='blue', marker='o',
#            alpha=0.6,
#            edgecolor='white',
#            s=40, label='POI Users')
#
# # 添加装饰元素
# plt.colorbar(heatmap, label='User density', shrink=0.8)
# plt.title('User distribution heat map', fontsize=14, pad=20)
# plt.xlabel('X Coordinate', fontsize=12)
# plt.ylabel('Y Coordinate', fontsize=12)
# plt.legend(loc='upper right', framealpha=0.9)
#
# # 添加等高线提升可读性
# contour = plt.contour(X, Y, Z,
#                      colors='black',
#                      linewidths=0.5,
#                      levels=5)
# plt.clabel(contour, inline=True, fontsize=8)
#
# # 设置科学计数法显示
# plt.colorbar(heatmap, format='%.1e')
# # 优化显示效果
# plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# plt.show()

# import numpy as np
#
#
# def generate_3d_users(total_users):
#     """
#     生成适应三维无人机覆盖的归一化用户坐标
#     返回：np.array([[x,y], ...]) 范围[0,1]
#     """
#     # 创建三层空间分布
#     ratios = [0.4, 0.35, 0.25]  # 各层用户比例
#
#     # 第一层：高密度核心区 (需要低空大覆盖半径)
#     core = np.random.normal(loc=[0.5, 0.5], scale=[0.08, 0.06],
#                             size=(int(total_users * ratios[0]), 2))
#
#     # 第二层：中密度环形区 (需要中空中等覆盖)
#     theta = np.random.uniform(0, 2 * np.pi, int(total_users * ratios[1]))
#     r = np.random.normal(0.25, 0.05, len(theta)) + 0.15
#     ring = np.column_stack([0.5 + r * np.cos(theta),
#                             0.5 + r * np.sin(theta)])
#
#     # 第三层：低密度边缘区 (适合高空小覆盖)
#     edge = np.random.uniform(0.1, 0.9, (int(total_users * ratios[2]), 2))
#     edge = edge[(np.abs(edge - 0.5) > 0.35).any(axis=1)]  # 排除中心区域
#
#     # 合并并限制范围
#     users = np.concatenate([core, ring, edge])
#     np.random.shuffle(users)
#     return np.clip(users, 0, 1)
#
#
# # 生成示例
# users = generate_3d_users(100)
#
# # 可视化
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(8, 8))
# plt.scatter(users[:, 0], users[:, 1], s=10, alpha=0.6)
# plt.title("3D-Optimized User Distribution")
# plt.grid(True, alpha=0.3)
# plt.show()
#
# # 三维优势可视化方法
# import plotly.graph_objects as go
#
# def plot_3d_coverage(users, z_values, coverage_radii):
#     fig = go.Figure()
#
#     # 用户点层
#     fig.add_trace(go.Scatter3d(
#         x=users[:, 0], y=users[:, 1], z=np.zeros(len(users)),
#         mode='markers',
#         marker=dict(size=2, color='blue', opacity=0.4)
#     ))
#
#     # 覆盖曲面层
#     for z, r in zip(z_values, coverage_radii):
#         u = np.linspace(0, 2 * np.pi, 50)
#         v = np.linspace(0, np.pi, 50)
#         x = r * np.outer(np.cos(u), np.sin(v)) + 0.5
#         y = r * np.outer(np.sin(u), np.sin(v)) + 0.5
#         z_grid = z / 20 * np.ones_like(x)  # 归一化高度
#
#         fig.add_trace(go.Surface(
#             x=x, y=y, z=z_grid,
#             colorscale='Reds',
#             opacity=0.1,
#             showscale=False
#         ))
#
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(range=[0, 1], title='X'),
#             yaxis=dict(range=[0, 1], title='Y'),
#             zaxis=dict(title='Normalized Altitude'),
#             camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
#         ),
#         margin=dict(l=0, r=0, b=0, t=30)
#     )
#     fig.show()
#
#
# # 归一化参数（假设最大高度20m对应覆盖半径6）
# z_norm = np.linspace(0, 1, 11)
# coverage_norm = np.array([17, 15, 13, 12, 11, 11, 10, 9, 9, 8, 8, 7, 7, 7, 6, 6]) / 17
#
# plot_3d_coverage(users, z_norm, coverage_norm)

#todo 1 now

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# 设置中文字体
font_path = r'C:\Windows\Fonts\simsun.ttc'  # 宋体路径
euclid_font_path = r'C:\Windows\Fonts\euclid.ttf'  # Euclid字体

cn_font = FontProperties(fname=font_path, size=23)  # 8pt中文
euclid_font = FontProperties(fname=euclid_font_path, size=23)  # 8pt Euclid

plt.rcParams['font.family'] = ['serif']
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式使用Stix字体
plt.rcParams['font.serif'] = ['SimSun', 'Euclid']  # 优先使用宋体和Euclid

def generate_poi(total_poi_num):
    """生成具有密度分层的POI坐标"""
    np.random.seed(23)
    # 创建分层参数
    layers = [
        {'center': (0.4, 0.6), 'ratio': 0.4, 'std': 0.03},  # 超高密度层
        {'center': (0.7, 0.3), 'ratio': 0.3, 'std': 0.06},  # 高密度层
        {'center': (0.5, 0.5), 'ratio': 0.2, 'std': 0.1},  # 中密度层
        {'ratio': 0.1}  # 均匀分布层
    ]

    all_poi = []

    for layer in layers:
        if 'center' in layer:
            # 生成高斯分布层
            num = int(total_poi_num * layer['ratio'])
            for _ in range(num):
                while True:
                    x = np.random.normal(layer['center'][0], layer['std'])
                    y = np.random.normal(layer['center'][1], layer['std'])
                    if 0 < x < 1 and 0 < y < 1:
                        all_poi.append([x, y])
                        break
        else:
            # 生成均匀分布层
            num = int(total_poi_num * layer['ratio'])
            points = np.random.uniform(0, 1, (num, 2))
            all_poi.extend(points.tolist())

    return np.array(all_poi)


def plot_coverage(poi):
    """可视化覆盖效果"""
    plt.figure(figsize=(10, 8))

    # 绘制POI分布
    plt.scatter(poi[:, 0], poi[:, 1], s=20, c='gray', alpha=0.6, label='Users')

    # 添加无人机覆盖示例
    drones_3d = [
        {'pos': (0.4, 0.6), 'z': 5, 'color': 'red'},  # 超高密度区使用低空
        {'pos': (0.7, 0.3), 'z': 8, 'color': 'orange'},  # 高密度区使用中低空
        {'pos': (0.5, 0.5), 'z': 15, 'color': 'blue'}  # 稀疏区使用高空
    ]

    # 对应的覆盖半径索引（根据给定的高度数组）
    ZMIN = 5
    ZMAX = 15
    z_values = np.arange(ZMIN, ZMAX + 1, 1).tolist()
    env_theta = np.deg2rad(60)
    R = z_values / np.tan(env_theta)
    for drone in drones_3d:
        z_index = z_values.index(drone['z'])
        radius = R[z_index] / 50  # 归一化半径（地图范围50单位）
        circle = plt.Circle(drone['pos'], radius,
                            color=drone['color'], alpha=0.2,
                            linestyle='--', linewidth=2)
        plt.gca().add_patch(circle)
        plt.plot(*drone['pos'], 'x', color=drone['color'], markersize=12)

    # 设置坐标轴标签（使用中文字体）
    plt.xlabel('归一化X坐标', fontproperties=cn_font)
    plt.ylabel('归一化Y坐标', fontproperties=cn_font)

    # 设置坐标轴刻度标签（重点修改部分）
    ax = plt.gca()
    x_ticks = np.arange(0, 1.1, 0.2)
    y_ticks = np.arange(0, 1.1, 0.2)

    # 关键修改：将x轴的第一个刻度标签设为空字符串，避免与y轴的0.0重复
    x_tick_labels = ['' if i == 0 else f'{x:.1f}' for i, x in enumerate(x_ticks)]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontproperties=euclid_font)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{y:.1f}' for y in y_ticks], fontproperties=euclid_font)


    # 设置图例（使用中文标签）
    legend_labels = ['用户', '超高密度', '高密度', '中等密度']

    plt.legend(handles=[
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10),
        plt.Line2D([0], [0], color='red', alpha=0.2, linewidth=10),
        plt.Line2D([0], [0], color='orange', alpha=0.2, linewidth=10),
        plt.Line2D([0], [0], color='blue', alpha=0.2, linewidth=10)
    ], labels=legend_labels, prop=cn_font, loc='upper right')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)

    plt.savefig("图2 用户地图设计.png",
                dpi=600,
                bbox_inches='tight',
                facecolor='white',  # 保证背景为纯白[9](@ref)
                transparent=False)  # 关闭透明背景[5](@ref)
    plt.savefig("图2 用户地图设计.pdf",
                format='pdf',
                bbox_inches='tight')  # 矢量格式适合论文[1,10](@ref)
    plt.show()


# 生成POI数据
poi = generate_poi(100)

# 无人机高度对应的覆盖半径（归一化前）
# z_values = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# R = [2.88675135, 3.46410162, 4.04145188, 4.61880215, 5.19615242, 5.77350269,
#      6.35085296, 6.92820323, 7.5055535, 8.08290377, 8.66025404, 9.23760431,
#      9.81495458, 10.39230485, 10.96965511, 11.54700538]

plt.rcParams.update({
    'font.family': 'serif',
    'figure.dpi': 600,
    'axes.linewidth': 0.8,
    # 'grid.color': '#DDDDDD',
    "mathtext.fontset": 'stix',
    'axes.unicode_minus': False,  # 处理负号显示
    'xtick.labelsize': 25,
    'ytick.labelsize': 25
})

# plt.rcParams.update({
#     'font.family': 'Times New Roman',
#     'font.size': 12,  # 统一字号[9](@ref)
#     'axes.titlesize': 14,  # 标题字号
#     'axes.labelsize': 12,  # 坐标轴标签字号
#     'xtick.labelsize': 10,
#     'ytick.labelsize': 10,
#     'figure.dpi': 600  # 全局DPI设置[9](@ref)
# })

# 可视化
plot_coverage(poi)

# # 生成用户地图并叠加热力图
# plt.figure(figsize=(10, 8))
#
# # 生成核密度估计热力图
# from scipy.stats import gaussian_kde
# kde = gaussian_kde(poi.T)  # 转置为(2, N)格式
# xgrid = np.linspace(0, 1, 100)
# ygrid = np.linspace(0, 1, 100)
# X, Y = np.meshgrid(xgrid, ygrid)
# Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
#
# # 绘制热力图（使用半透明效果）
# heatmap = plt.imshow(Z,
#                     cmap='viridis',
#                     aspect='auto',
#                     extent=[0, 1, 0, 1],
#                     origin='lower',
#                     alpha=0.6)
#
# # 添加散点图
#
# plt.scatter(poi[:, 0], poi[:, 1],
#            c='blue', marker='o',
#            alpha=0.6,
#            edgecolor='white',
#            s=40, label='POI Users')
#
# # 添加装饰元素
# plt.colorbar(heatmap, label='User density', shrink=0.8)
# plt.title('User distribution heat map', fontsize=14, pad=20)
# plt.xlabel('X Coordinate', fontsize=12)
# plt.ylabel('Y Coordinate', fontsize=12)
# plt.legend(loc='upper right', framealpha=0.9)
#
# # 添加等高线提升可读性
# # contour = plt.contour(X, Y, Z,
# #                      colors='black',
# #                      linewidths=0.5,
# #                      levels=5)
# # plt.clabel(contour, inline=True, fontsize=8)
#
# # 设置科学计数法显示
# # plt.colorbar(heatmap, format='%.1e')
# # 优化显示效果
# # plt.grid(True, linestyle='--', alpha=0.3)
# plt.tight_layout()
# # 在热力图生成代码末尾添加：
# plt.tight_layout()  # 自动调整布局[4](@ref)
# plt.savefig("user_density_heatmap.png",
#            dpi=600,
#            bbox_inches='tight',
#            facecolor='white',
#            pad_inches=0.1)  # 增加0.1英寸边距[4](@ref)
# plt.savefig("user_density_heatmap.svg",
#            format='svg',
#            bbox_inches='tight')  # 矢量格式便于后期编辑[10](@ref)
# plt.show()


#todo

# import numpy as np
# import matplotlib.pyplot as plt
#
#
# def generate_3d_aware_poi(total_poi_num=100):
#     """生成适合三维覆盖验证的用户分布"""
#     np.random.seed(42)  # 保证可重复性
#
#     # 创建三层不同密度的区域
#     poi_coords = []
#
#     # 1. 高密度区域（需要低高度覆盖）
#     centers = np.array([[0.3, 0.3], [0.7, 0.7], [0.3, 0.7]])  # 三个高密度中心
#     for center in centers:
#         # 每个中心生成约25个点，标准差小
#         num = int(total_poi_num * 0.25)
#         x = np.random.normal(center[0], 0.03, num)
#         y = np.random.normal(center[1], 0.03, num)
#         poi_coords.extend(np.column_stack((x, y)))
#
#     # 2. 中等密度区域（需要中等高度覆盖）
#     x = np.random.uniform(0.2, 0.8, int(total_poi_num * 0.3))
#     y = np.random.uniform(0.2, 0.8, int(total_poi_num * 0.3))
#     poi_coords.extend(np.column_stack((x, y)))
#
#     # 3. 低密度区域（需要高空覆盖）
#     x = np.random.normal(0.5, 0.3, int(total_poi_num * 0.2))
#     y = np.random.normal(0.5, 0.3, int(total_poi_num * 0.2))
#     valid = (x > 0.02) & (x < 0.98) & (y > 0.02) & (y < 0.98)
#     poi_coords.extend(np.column_stack((x[valid], y[valid])))
#
#     return np.array(poi_coords)[:total_poi_num]  # 确保总数正确
#
#
# # 生成用户位置
# poi_coords = generate_3d_aware_poi(100)
#
# # 可视化
# plt.figure(figsize=(10, 8))
# plt.scatter(poi_coords[:, 0], poi_coords[:, 1], s=20, alpha=0.6, c='green', label='Users')
#
#
# # 添加覆盖范围示例
# def plot_coverage(ax, position, z, color):
#     """绘制不同高度的覆盖范围"""
#     R_values = [2.886, 6.928, 8.660]  # z=5,12,15对应的覆盖半径（单位：米）
#     scale_factor = 10  # 坐标缩放因子
#     normalized_R = R_values[z // 5 - 1] / (50 * scale_factor)  # 归一化计算
#
#     circle = plt.Circle(position, normalized_R,
#                         color=color, fill=False, linestyle='--', linewidth=2)
#     ax.add_patch(circle)
#     plt.scatter(*position, marker='^', color=color, s=100, label=f'Drone z={z}m')
#
#
# # 绘制三维覆盖示例
# ax = plt.gca()
# plot_coverage(ax, (0.3, 0.3), z=5, color='red')  # 低空覆盖高密度区域
# plot_coverage(ax, (0.5, 0.5), z=15, color='blue')  # 高空覆盖分散区域
#
# # 二维对比示例（单一高度）
# plot_coverage(ax, (0.7, 0.3), z=12, color='purple')
#
# plt.title('3D vs 2D Coverage Demonstration\n')
# plt.xlabel('Normalized X Coordinate')
# plt.ylabel('Normalized Y Coordinate')
# plt.legend(loc='upper right')
# plt.grid(True)
# plt.show()