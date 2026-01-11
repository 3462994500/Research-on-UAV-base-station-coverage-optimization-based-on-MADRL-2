# import networkx as nx
# import numpy as np
#
# def generate_adjacency_matrix(n_agent):
#     # 生成一个随机的邻接矩阵，元素为0或1
#     adj_matrix = np.random.randint(0, 2, size=(n_agent, n_agent))
#     # 确保矩阵是对称的，因为关系是双向的
#     adj_matrix = np.triu(adj_matrix) + np.triu(adj_matrix, 1).T
#     # 将对角线上的元素设置为0，因为无人机不与自己通信
#     np.fill_diagonal(adj_matrix, 0)
#     return adj_matrix
#
# # 假设 n_agent = 3
# n_agent = 3
# adj = generate_adjacency_matrix(n_agent)
# print(adj)
# # 使用邻接矩阵 adj 创建一个图对象 adj_graph
# adj_graph = nx.from_numpy_matrix(adj)
# print(adj_graph)
# # 找出图 adj_graph 中的所有连通分量，并将每个连通分量的节点列表转换为元组
# sub_graphs = tuple(
#     adj_graph.subgraph(c).nodes() for c in nx.connected_components(adj_graph))  # ((0,1,2),(3,4,),...)
#
# id2group_dic = {}
# # 遍历所有的子图
# for i_group, group in enumerate(sub_graphs):
#     # 对于每个子图中的智能体
#     print(i_group, group)
#     for id in group:
#         # 将智能体的id映射到它所属的子图的索引，即组号
#         id2group_dic[id] = i_group
# print(sub_graphs, id2group_dic)
import numpy as np

data = np.array([
    0.39, 0.73, 0.91, 0.85, 0.71, 0.88, 0.78, 0.77, 0.89, 0.0, 0.0, 0.45, 0.4, 0.75,
    0.54, 0.68, 0.0, 0.77, 0.63, 0.87, 0.92, 0.55, 0.89, 0.03, 0.87, 0.06, 0.0, 0.94,
    0.26, 0.85, 0.86, 0.0, 0.87, 0.0, 0.26, 0.88, 0.37, 0.56, 0.05, 0.9, 0.87, 0.0,
    0.31, 0.63, 0.16, 0.11, 0.36, 0.9, 0.88, 0.22, 0.19, 0.21, 0.47, 0.39, 0.88, 0.64,
    0.29, 0.6, 0.16, 0.5
])

variance = np.var(data)
print("方差:", variance)




