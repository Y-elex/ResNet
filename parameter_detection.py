import os
import torch
from torch import nn, optim
from typing import List, Optional
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from labml import experiment
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import sys
sys.path.append('E:/python code/annotated_deep_learning_paper_implementations/labml_nn')
sys.path.append('E:/python code/annotated_deep_learning_paper_implementations')
from labml.configs import option
from experiments.cifar10 import CIFAR10Configs
from torchinfo import summary
from resnet import ResNetBase
from ptflops import get_model_complexity_info

class Configs(CIFAR10Configs):
    """
    配置类，用于设置模型和训练参数
    """
    n_blocks: List[int] = [6, 6, 6]  # ResNet的块数
    n_channels: List[int] = [16, 32, 64]  # ResNet的每个卷积块的通道数
    bottlenecks: Optional[List[int]] = [8, 16, 16]  # 每个卷积块的瓶颈层大小
    first_kernel_size: int = 3  # 第一层卷积核大小
    train_batch_size: int = 32  # 批次大小
    epochs: int = 100  # 训练的总轮数
    learning_rate: float = 2.5e-4  # 学习率
    optimizer: str = 'Adam'  # 优化器
    train_dataset: str = 'data/AffectNet/train'  # 训练数据集路径
    valid_dataset: str = 'data/AffectNet/val'  # 验证数据集路径
    pretrained_model_path: str = 'E:/python code/results/models/Resnet/best_model_AffectNet.pth'  # 预训练模型路径

# 创建ResNet模型
@option(Configs.model)
def _resnet(c: Configs):
    base = ResNetBase(c.n_blocks, c.n_channels, c.bottlenecks, img_channels=3, first_kernel_size=c.first_kernel_size)
    classification = nn.Linear(c.n_channels[-1], 8)  # 输出7个类别，假设有7个情感类别
    model = nn.Sequential(base, classification)
    return model.to(c.device)

def load_pretrained_model(model, model_path, device):
    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model = model.to(device)
    else:
        print(f"No pretrained model found at {model_path}. Starting from scratch.")
    return model

num_classes = 8
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
conf = Configs()
model = conf.model
model = model.to(device)
model = load_pretrained_model(model, conf.pretrained_model_path, device)
summary(model, input_size=(1, 3, 120, 120))
model.eval()
with torch.cuda.device(0):
    flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print(f"FLOPs: {flops}")
    print(f"Parameters: {params}")