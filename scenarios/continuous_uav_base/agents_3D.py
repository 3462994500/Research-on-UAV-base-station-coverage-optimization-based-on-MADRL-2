import numpy as np

class AgentGroup3D:
    def __init__(self, n_agent, pos=None, vel=None, mass=1):
        self.n_agent = n_agent
        if pos is not None:#xyz轴位置
            assert isinstance(pos, np.ndarray)
            assert pos.shape == (n_agent, 3)
            self.pos = pos
        else:#如果没有提供该参数，将会随机生成智能体的初始位置
            self.pos = np.random.rand(n_agent, 3)
        if vel is not None:#xy轴速度
            assert isinstance(vel, np.ndarray)
            assert vel.shape == (n_agent, 3)
        else:
            self.vel = np.zeros([n_agent, 3], dtype=np.float32)
        self.mass = mass

    @property
    def x(self):
        return self.pos[:, 0]

    @property
    def y(self):
        return self.pos[:, 1]

    @property
    def z(self):
        return self.pos[:, 2]

    @property
    def u(self):
        return self.vel[:, 0]

    @property
    def v(self):
        return self.vel[:, 1]

    @property
    def w(self):
        return self.vel[:, 2]
