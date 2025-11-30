import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def create_data_loaders(train_dir: str, valid_dir: str, batch_size: int):
    """
    创建训练和验证数据加载器
    :param train_dir: 训练数据集目录
    :param valid_dir: 验证数据集目录
    :param batch_size: 批次大小
    :return: 训练和验证数据加载器
    """
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # 调整图像大小以适应网络输入尺寸（假设224x224）
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet标准化
    ])

    # 训练数据集
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 验证数据集
    valid_dataset = datasets.ImageFolder(root=valid_dir, transform=transform)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader
