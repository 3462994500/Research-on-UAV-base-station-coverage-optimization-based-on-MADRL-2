# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# uav_pos=[[120,13,50],[58,212,50],[83,13,50]]
# poi_pos=[[129,168],[117,182],[116,168],[117,186],[147,157],[120,136],[182,194],[ 81,180],[111,183],[137,154], [114,171],[131,178]]
#
# # 示例三维无人机轨迹数据
# trajectory_3d = [
#     (0, 0, 0),
#     (1, 2, 1),
#     (2, 3, 2),
#     (3, 5, 3),
#     (4, 4, 4),
#     (5, 6, 5),
#     (6, 5, 6),
#     (7, 7, 7)
# ]
#
# # 分离轨迹数据的x, y和z坐标
# x_coords = [point[0] for point in trajectory_3d]
# y_coords = [point[1] for point in trajectory_3d]
# z_coords = [point[2] for point in trajectory_3d]
#
# # 创建图形和3D轴
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# # 绘制三维轨迹图
# ax.plot(x_coords, y_coords, z_coords, marker='o')
#
# # 添加标题和标签
# ax.set_title('3D Drone Trajectory')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Z Coordinate')
#
# # 显示图形
# plt.show()

def get3DMap(self):
    def getCoords(map):
        x, y, z = [], [], []
        for obj in map:
            x.append(obj.current_location[0])
            y.append(obj.current_location[1])
            z.append(obj.current_location[2])
        return x, y, z

    return getCoords(self.map["uav_set"]), getCoords(self.map["ue_set"])

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d');
uav_coords, ue_coords = get3DMap()
clear_output(wait=True)
ax.set_xlim3d(0, env_dim[0])
ax.set_ylim3d(0, env_dim[1])
ax.set_zlim3d(0, MAX_ALTITUDE)
ax.scatter(*uav_coords, c="green")
ax.scatter(*ue_coords, c="red")
distance_map = get_distance_matrix()["assoc_matrix_uav"]
for uav_index in range(uav_count):
    coord_of_UAV = getUAV(uav_index).current_location
    for ue_index in distance_map[uav_index]:
        coord_of_UE = map["ue_set"][ue_index].current_location
        ax.plot([coord_of_UE[0], coord_of_UAV[0]], [coord_of_UE[1], coord_of_UAV[1]], [coord_of_UE[2], coord_of_UAV[2]],
                c="green", alpha=.1)
plt.show()

# new_env.render(ax=ax, plot_trajectories=plot_trajectories)
# # Plot resource errors by step and interation
# plot_step_graph(sum_rates_per_step, ax1, lim=step_count)