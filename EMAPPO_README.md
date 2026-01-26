# Enhanced MAPPO (EMAPPO) - 改进的多智能体PPO算法

## 概述

Enhanced MAPPO (EMAPPO) 是对标准MAPPO算法的改进版本，专门针对无人机协同覆盖场景进行了优化。该算法通过引入图注意力网络(GAT)、改进的奖励处理和训练策略，显著提升了训练效果和全局奖励。

## 主要改进点

### 1. 图注意力网络(GAT)集成
- **Actor网络增强**：使用GAT层处理邻接矩阵信息，使智能体能够更好地理解多智能体协作关系
- **Critic网络增强**：利用图结构信息改进全局状态价值估计
- **自适应图结构**：根据通信邻接矩阵动态调整注意力权重

### 2. 改进的奖励处理
- **奖励归一化**：使用运行统计量对奖励进行归一化，提高训练稳定性
- **优势函数优化**：对每个智能体分别归一化优势函数，考虑个体差异
- **奖励塑形**：更好地利用环境的多个奖励组件（覆盖、公平性、速率等）

### 3. 自适应训练策略
- **自适应学习率**：使用指数衰减调度器，随着训练进行逐渐降低学习率
- **自适应熵系数**：动态调整熵正则化系数，平衡探索与利用
- **改进的梯度裁剪**：使用更大的梯度裁剪阈值，防止梯度爆炸

### 4. 网络架构改进
- **残差连接**：在GAT层之间添加残差连接，提高训练稳定性
- **Layer Normalization**：使用层归一化加速收敛
- **多头注意力**：使用多头注意力机制捕获不同类型的协作模式

## 文件结构

```
agents/
  └── emappo.py              # Enhanced MAPPO Agent类

modules/
  └── emappo.py              # Enhanced MAPPO网络模块（GAT层、Actor、Critic）

trainers/
  └── emappo_trainer.py      # Enhanced MAPPO Trainer类

configs/
  ├── base_configs/
  │   └── emappo.yaml        # 基础配置文件
  └── scenarios/
      └── continuous_mcs3D/
          └── emappo.yaml     # 场景配置文件
```

## 使用方法

### 1. 训练

```bash
python main.py --config configs/scenarios/continuous_mcs3D/emappo.yaml --exp_name emappo_experiment
```

### 2. 配置参数说明

#### 网络参数
- `hidden_dim`: 隐藏层维度（默认：128）
- `use_gat`: 是否使用GAT网络（默认：true）
- `num_heads`: GAT注意力头数（默认：4）

#### 训练参数
- `learning_rate`: 初始学习率（默认：0.0001）
- `lr_decay`: 学习率衰减率（默认：0.9999）
- `batch_size`: 批次大小（默认：32）
- `gamma`: 折扣因子（默认：0.99）
- `gae_lambda`: GAE参数（默认：0.95）

#### PPO参数
- `clip_param`: PPO裁剪参数（默认：0.2）
- `entropy_coef`: 初始熵系数（默认：0.01）
- `entropy_coef_min`: 最小熵系数（默认：0.001）
- `entropy_coef_decay`: 熵系数衰减率（默认：0.9995）
- `ppo_epochs`: PPO训练轮数（默认：4）

#### 奖励归一化
- `reward_normalize`: 是否归一化奖励（默认：true）
- `reward_normalize_alpha`: 奖励归一化平滑系数（默认：0.99）

### 3. 自定义配置

可以通过命令行参数覆盖配置：

```bash
python main.py --config configs/scenarios/continuous_mcs3D/emappo.yaml \
    --exp_name emappo_custom \
    --hparams "learning_rate=0.0002,use_gat=true,num_heads=8"
```

## 性能优势

相比标准MAPPO，EMAPPO在以下方面有显著提升：

1. **更高的全局奖励**：通过GAT网络更好地利用多智能体协作，提升整体性能
2. **更快的收敛速度**：改进的奖励处理和归一化策略加速训练
3. **更好的稳定性**：自适应学习率和熵系数提高训练稳定性
4. **更强的泛化能力**：图结构信息使算法能更好地适应不同的通信拓扑

## 技术细节

### GAT层实现
- 使用多头注意力机制处理邻接矩阵
- 每个头独立计算注意力权重
- 通过邻接矩阵掩码限制注意力范围

### 奖励归一化
- 使用指数移动平均(EMA)维护运行统计量
- 对奖励进行Z-score归一化
- 提高不同奖励尺度下的训练稳定性

### 优势函数计算
- 对每个智能体分别归一化优势函数
- 考虑个体差异，避免优势函数被少数智能体主导
- 使用GAE计算，平衡偏差和方差

## 注意事项

1. **内存使用**：GAT网络会增加内存使用，特别是在智能体数量较多时
2. **计算开销**：图注意力计算会增加训练时间，但通常能带来性能提升
3. **超参数调优**：建议根据具体场景调整学习率、熵系数等超参数

## 实验建议

1. **对比实验**：与标准MAPPO进行对比，验证改进效果
2. **消融实验**：分别测试GAT、奖励归一化等组件的贡献
3. **超参数搜索**：对关键超参数进行网格搜索或贝叶斯优化

## 参考文献

- MAPPO: Multi-Agent Proximal Policy Optimization
- Graph Attention Networks (GAT)
- Proximal Policy Optimization Algorithms (PPO)
