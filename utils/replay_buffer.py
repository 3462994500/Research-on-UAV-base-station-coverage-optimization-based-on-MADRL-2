import torch
import numpy as np
from utils.hparams import hparams
from utils.torch_utils import *
import logging
from utils.device import device
from collections import defaultdict


class ReplayBuffer:
    def __init__(self):
        self.capacity = int(hparams['buffer_capacity'])
        self.current_idx = 0
        self.num_stored_sample = 0
        self.buffer = None  # 使用字典存储所有特征
        self.layer_pattern = "_layer"  # 隐藏状态层级分隔符

    def _initialize_buffer_with_an_example_sample(self, sample):
        """根据样本结构初始化缓冲区，支持多层隐藏状态"""
        self.buffer = {}

        for key, value in sample.items():
            # 处理列表型隐藏状态 (hiddens: List[[batch, agents, hid]])
            if isinstance(value, list):
                for layer_idx, layer_value in enumerate(value):
                    self._init_buffer_entry(f"{key}{self.layer_pattern}{layer_idx}", layer_value)
            # 处理标准数据类型
            elif isinstance(value, (np.ndarray, torch.Tensor)):
                self._init_buffer_entry(key, value)
            # 处理嵌套字典结构
            elif isinstance(value, dict):
                self._init_dict_structure(key, value)
            else:
                raise TypeError(f"不支持的初始数据类型: {type(value)}")

        logging.info("\n缓冲区结构详情:")
        for k, v in self.buffer.items():
            logging.info(f"{k}, shape={v.shape}")
        logging.info("缓冲区初始化成功\n")

    def _init_buffer_entry(self, key, value):
        """初始化单个缓冲区条目"""
        # 转换并压缩维度
        tensor = torch.as_tensor(value)
        if tensor.dim() >= 3 and tensor.shape[0] == 1:  # 压缩batch维度
            tensor = tensor.squeeze(0)

        # 根据维度创建存储空间
        if tensor.dim() == 1:  # [n_agent]
            self.buffer[key] = torch.zeros(self.capacity, tensor.shape[0], dtype=torch.float32)
        elif tensor.dim() == 2:  # [n_agent, feature_dim]
            self.buffer[key] = torch.zeros(self.capacity, *tensor.shape, dtype=torch.float32)
        elif tensor.dim() == 3:  # 保留3D结构 [n_agent, seq_len, hid_dim]
            self.buffer[key] = torch.zeros(self.capacity, *tensor.shape, dtype=torch.float32)
        else:
            raise ValueError(f"不支持的张量维度 {tensor.dim()} 对于键 {key}")

    def _init_dict_structure(self, prefix, data_dict):
        """初始化字典结构数据"""
        for sub_key, sub_value in data_dict.items():
            full_key = f"{prefix}_{sub_key}"
            if isinstance(sub_value, (np.ndarray, torch.Tensor)):
                self._init_buffer_entry(full_key, sub_value)
            elif isinstance(sub_value, dict):
                self._init_dict_structure(full_key, sub_value)  # 递归处理嵌套字典
            else:
                raise TypeError(f"字典中不支持的类型 {type(sub_value)}")

    def push(self, sample):
        """存入样本数据，支持多层隐藏状态列表"""
        if self.buffer is None:
            self._initialize_buffer_with_an_example_sample(sample)

        # 预处理隐藏状态列表
        processed_sample = defaultdict(list)
        for key, value in sample.items():
            if isinstance(value, list):
                # 分离各层隐藏状态
                for layer_idx, layer_value in enumerate(value):
                    layer_key = f"{key}{self.layer_pattern}{layer_idx}"
                    processed_sample[layer_key] = layer_value
            else:
                processed_sample[key] = value

        # 存入处理后的数据
        for key, value in processed_sample.items():
            if isinstance(value, (np.ndarray, torch.Tensor)):
                tensor = torch.as_tensor(value, dtype=torch.float32)
                # 压缩单样本的batch维度
                if tensor.dim() >= 3 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
                self.buffer[key][self.current_idx] = tensor.to('cpu')  # 统一存储在CPU
            elif isinstance(value, dict):
                self._push_dict_value(key, value)
            else:
                raise TypeError(f"不支持的数据类型 {type(value)} 对于键 {key}")

        self.current_idx = (self.current_idx + 1) % self.capacity
        self.num_stored_sample = min(self.num_stored_sample + 1, self.capacity)

    def _push_dict_value(self, prefix, data_dict):
        """处理字典结构数据"""
        for sub_key, sub_value in data_dict.items():
            full_key = f"{prefix}_{sub_key}"
            if isinstance(sub_value, (np.ndarray, torch.Tensor)):
                tensor = torch.as_tensor(sub_value, dtype=torch.float32)
                self.buffer[full_key][self.current_idx] = tensor.to('cpu')
            else:
                raise TypeError(f"字典中不支持的类型 {type(sub_value)}")

    def sample(self, batch_size):
        """采样数据并重组隐藏状态结构"""
        if self.num_stored_sample < batch_size:
            return None

        # 生成随机索引
        indices = np.random.choice(self.num_stored_sample, batch_size, replace=False)

        # 重组数据结构
        batch = defaultdict(list)
        for key in self.buffer.keys():
            # 检测是否为隐藏状态层级
            if self.layer_pattern in key:
                base_key, layer_num = key.split(self.layer_pattern)
                batch[base_key].append((int(layer_num), self.buffer[key][indices]))
            else:
                batch[key] = self.buffer[key][indices]

        # 重组隐藏状态列表
        for base_key in list(batch.keys()):
            if isinstance(batch[base_key], list):
                # 按层级排序
                sorted_layers = sorted(batch[base_key], key=lambda x: x[0])
                # 合并为列表 [batch_size, num_layers, ...]
                batch[base_key] = torch.stack([layer[1] for layer in sorted_layers], dim=1).to(device)
            else:
                # 转移常规数据到设备
                batch[base_key] = batch[base_key].to(device)

        return dict(batch)

    def __len__(self):
        return self.num_stored_sample