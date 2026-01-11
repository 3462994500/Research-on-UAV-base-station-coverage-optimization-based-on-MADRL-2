from utils.hparams import hparams
import matplotlib.pyplot as plt
import numpy as np

NUM_GRIDS = hparams['env_num_grid']
ZMAX = hparams['env_zmax']
SUOF = hparams['env_suofang']


def plot_coverage_heatmap(scene):
    """
    生成二维覆盖热力地图，颜色深度表示位置被覆盖的累计时间
    参数：
        scene: 所有时间步的场景数据列表
        NUM_GRIDS: 地图尺寸（格子数量)
    """
    # 🌟 学术论文样式配置
    plt.style.use('seaborn')  # 基于seaborn的优雅风格
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelweight': 'bold',
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.4,
        'figure.dpi': 600  # 印刷级分辨率
    })

    # 初始化热力矩阵
    heatmap = np.zeros((NUM_GRIDS, NUM_GRIDS))
    x_centers = np.linspace(0.5, NUM_GRIDS - 0.5, NUM_GRIDS)
    y_centers = np.linspace(0.5, NUM_GRIDS - 0.5, NUM_GRIDS)
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')
    # 累计覆盖时间
    # 性能优化版（避免重复计算）
    for step_data in scene:
        # 生成所有无人机的覆盖掩模
        step_data['uavs'][:, :2] *= NUM_GRIDS
        # step_data['uavs'][:, 2] *= ZMAX
        coverage_masks = [
            (np.hypot(X - x, Y - y) <= radius)
            for (x, y, _), radius in zip(step_data['uavs'], step_data['rcov'])
            if radius > 0  # 过滤无效半径
        ]

        # 合并所有掩模（使用np.logical_or.reduce优化性能）
        if coverage_masks:
            step_coverage = np.logical_or.reduce(coverage_masks)
            heatmap += step_coverage.astype(float)

    # 🌟 创建可视化画布
    fig, ax = plt.subplots(figsize=(8, 6), tight_layout=True)

    # 🌟 热力图层（使用科学配色）
    im = ax.imshow(heatmap.T,
                   origin='lower',
                   extent=[0, NUM_GRIDS, 0, NUM_GRIDS],
                   cmap='viridis',  # 色盲友好配色
                   interpolation='gaussian',  # 高斯平滑
                   aspect='equal')

    # 🌟 专业级颜色条设置
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.03)
    cbar.set_label('Coverage Time (steps)', weight='bold')
    cbar.outline.set_visible(False)  # 去除边框

    # 🌟 POI标记（带边框的星形）
    final_pois = np.unique(scene[-1]['pois'], axis=0)  # 去重处理
    for x, y in final_pois:
        ax.scatter(x, y, s=120, marker='*',
                   color='gold', edgecolor='k', linewidth=0.5,
                   zorder=10, label='UE')

    # 🌟 坐标轴美化
    ax.set(xlim=(0, NUM_GRIDS), ylim=(0, NUM_GRIDS),
           xlabel='X Coordinate', ylabel='Y Coordinate')
    ax.tick_params(axis='both', which='both', length=0)  # 隐藏刻度线
    ax.set_xticks(np.arange(0, NUM_GRIDS + 1, 2))  # 间隔2的刻度
    ax.set_yticks(np.arange(0, NUM_GRIDS + 1, 2))
    ax.set_facecolor('#f5f5f5')  # 浅灰色背景

    # 🌟 智能图例（自动去重）
    handles, labels = ax.get_legend_handles_labels()
    legend_elements = {label: handle for handle, label in zip(handles, labels)}
    ax.legend(legend_elements.values(), legend_elements.keys(),
              loc='upper right', frameon=True,
              framealpha=0.9, handletextpad=0.5)

    # 🌟 保存矢量格式（可选）
    plt.savefig('Coverage_Heatmap.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.close()

def visualize_3d(scene, num_steps=hparams['episode_length']):
    plt.ion()  # 开启交互模式
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=27, azim=48)  # 设置仰角和方位角
    # 科研图片参数配置
    plt.rcParams.update({
        'font.family': 'Times New Roman',  # 学术字体
        'font.size': 12,  # 统一字体大小
        'axes.labelpad': 10  # 坐标轴标签间距
    })

    # 保存参数配置
    save_params = {
        'dpi': 600,
        'bbox_inches': 'tight',
        'transparent': False,
        'format': 'pdf'
    }

    # 添加交互保存功能
    save_count = 0

    def save_current_view(event):
        nonlocal save_count
        if event.inaxes == ax:  # 确保在3D坐标系内操作
            filename = f"UAV_View_{save_count}.pdf"
            plt.savefig(filename, ** save_params)
            print(f"\nSaved: {filename} (View: {ax.elev}°/{ax.azim}°)")
            save_count += 1

    # 绑定鼠标右键点击保存
    fig.canvas.mpl_connect('button_press_event', lambda event: save_current_view(event) if event.button == 3 else None)
    # 绑定键盘快捷键 (s键)
    fig.canvas.mpl_connect('key_press_event', lambda event: save_current_view(event) if event.key == 's' else None)

    # 初始化颜色方案和轨迹列表
    num_uavs = len(scene[0]['uavs'])
    colors = plt.cm.tab10(np.linspace(0, 1, num_uavs))  # 使用matplotlib默认颜色循环
    uav_trajectories = [[] for _ in range(num_uavs)]
    uav_velocities = [[] for _ in range(num_uavs)]  # 新增：存储速度历史
    max_vel = 20/(SUOF*NUM_GRIDS)   # 假设max_vel可从hparams获取
    collide_agent_num_history = []
    for step in range(num_steps):
        ax.clear()
        scene_data = scene[step]
        scene_data['uavs'][:, :2] *= NUM_GRIDS
        scene_data['uavs'][:, 2] *= ZMAX
        uav_positions = scene_data['uavs']
        uav_adj = scene_data['adj']
        uav_robs = scene_data['robs']
        uav_rcov = scene_data['rcov']
        collide_adj = scene_data['collide_adj']
        collide_agent_num = collide_adj.sum(axis=1)
        collide_agent_num_history.append(sum(collide_agent_num))
        # collide_adj = np.any(collide_adj == 1, axis=1)
        # agent_cover = scene_data['agent_cover']
        vel = scene_data['vel']

        # 更新轨迹
        for i, pos in enumerate(uav_positions):
            uav_trajectories[i].append(pos)
            uav_velocities[i].append(vel[i])  # 新增：记录速度

        # 绘制每个无人机元素
        for i, pos in enumerate(uav_positions):
            x, y, z = pos
            current_color = colors[i]

            # 无人机本体
            ax.scatter(x, y, z, color=current_color, s=50, label=f'UAV {i}' if step == 0 else "")
            # ax.text(x, y, z, f" {round(z),collide_adj[i],agent_cover[i]}", fontsize=8, color='black')  # 添加标签
            # 在无人机本体上添加居中编号（新增代码）
            ax.text(x+0.5, y+0.5, z+0.5,
                    f"{i + 1}",
                    fontsize=9,  # 稍大字号
                    color='black',  # 白色文字提高对比度
                    ha='left',  # 水平
                    va='bottom',  # 垂直
                    zorder=11  # 层级高于本体
                    )
            # 通信连接（保持灰色）
            for j in range(len(uav_positions)):
                if uav_adj[i][j] == 1 and i != j:
                    x2, y2, z2 = uav_positions[j]
                    ax.plot([x, x2], [y, y2], [z, z2], c='gray', linestyle='--', linewidth=1)

            # 地面投影和覆盖范围（使用同色系）
            ax.plot([x], [y], [0], marker='o', markersize=5, color=current_color, alpha=0.3)

            # 在可视化函数中这样调用
            plot_circle_on_ground(ax, x, y, uav_robs[i],
                                  color=current_color,
                                  alpha=0.2,
                                  linewidth=1)  # 观测半径

            plot_circle_on_ground(ax, x, y, uav_rcov[i],
                                  color=current_color,
                                  is_coverage=True)  # 覆盖半径
            # 三维轨迹（动态绘制）
            trajectory = np.array(uav_trajectories[i])
            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                        c=current_color, linewidth=2, alpha=0.7)

            # 绘制速度相关轨迹（修改部分）
            trajectory = np.array(uav_trajectories[i])
            velocities = np.array(uav_velocities[i])
            if len(trajectory) > 1:
                # 逐线段绘制，根据速度调整颜色
                for j in range(len(trajectory) - 1):
                    # 计算速度向量的模（新增行）
                    speed = np.linalg.norm(velocities[j])  # 关键修复：计算速度大小

                    normalized_speed = np.clip(speed / max_vel, 0, 1)

                    # 调整颜色亮度（保持色相，降低亮度使颜色变深）
                    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
                    rgb = current_color[:3]
                    hsv = rgb_to_hsv(rgb)
                    # 速度越快，亮度提高（hsv[2]乘数增大）
                    hsv[2] = 0.4 + 0.6 * normalized_speed
                    new_rgb = hsv_to_rgb(hsv)

                    # 绘制线段
                    ax.plot([trajectory[j, 0], trajectory[j + 1, 0]],
                            [trajectory[j, 1], trajectory[j + 1, 1]],
                            [trajectory[j, 2], trajectory[j + 1, 2]],
                            color=new_rgb, linewidth=2, alpha=0.7)

        # 绘制POI（保持红色不变）
        poi_positions = scene_data['pois']
        for x, y in poi_positions:
            ax.scatter(x, y, 0, c='red', alpha=0.5, marker='x', label='POI' if step == 0 else "")

        # 设置坐标轴和标签
        ax.set(xlim3d=(-0.5, NUM_GRIDS + 0.5),
               ylim3d=(-0.5, NUM_GRIDS + 0.5),
               zlim3d=(0, ZMAX))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if step == 0:
            # 智能图例处理（避免重复标签）
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique), loc='upper right')

        plt.pause(0.1)
    collide = sum(collide_agent_num_history)
    print(collide)
    plt.savefig("UAV_3D_Final.pdf", ** save_params)
    plt.ioff()
    plt.show()



# 辅助函数保持原样
# 观测半径用半透明细线，覆盖半径用深色粗线
def plot_circle_on_ground(ax, x, y, radius, color, alpha=0.3, linewidth=1, is_coverage=False):
    theta = np.linspace(0, 2 * np.pi, 100)  # 固定100个点保证圆滑

    # 覆盖半径加强视觉效果
    if is_coverage:
        linewidth = max(linewidth, 1)  # 线宽至少2
        alpha = max(alpha, 0.5)  # 透明度不高于0.5
        color = np.array(color) * 0.7  # 颜色加深（RGB值缩小）

    x_circle = x + radius * np.cos(theta)
    y_circle = y + radius * np.sin(theta)
    ax.plot(x_circle, y_circle, 0,
            color=color,
            alpha=alpha,
            linewidth=linewidth,
            linestyle='-')