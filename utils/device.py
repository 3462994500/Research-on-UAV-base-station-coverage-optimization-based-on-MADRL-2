# utils/device.py

import torch

# 初始化设备
device = None

def init_device():
    global device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.current_device())
    else:
        device = torch.device("cpu")
        print("Using CPU")

# 初始化设备
init_device()