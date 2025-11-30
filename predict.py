import os
import torch
import pandas as pd
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
import sys
from collections import defaultdict

sys.path.append('E:/python_code/FER')
from resnet import ResNetBase

data = 'AffectNet'  # 修改为你的数据集名称

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 用于展示的 emotion_map（仅当 ImageFolder 顺序与之匹配时才准确）
emotion_map = {
    0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Sad', 5: 'Surprise', 6: 'Neutral', 7: 'Contempt'
}

# 加载模型结构
base = ResNetBase(n_blocks=[6, 6, 6], n_channels=[16, 32, 64], bottlenecks=[8, 16, 16], img_channels=3, first_kernel_size=3)
model = nn.Sequential(base, nn.Linear(64, 8))  # 输出8类
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = f'E:/python_code/results/models/Resnet/best_model_{data}.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ 模型文件不存在: {os.path.abspath(model_path)}")

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 预测函数
def predict_dataset(root_dir, split_name):
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 获取 ImageFolder 的真实类别映射 (关键！)
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    results = []
    true_labels_all = []
    pred_labels_all = []

    for i, (inputs, labels) in enumerate(tqdm(loader, desc=f"Predicting {split_name}")):
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

        img_path, _ = dataset.samples[i]
        image_name = os.path.basename(img_path)
        true_label = labels.item()
        pred_label = predicted.item()

        # 使用 ImageFolder 的真实映射获取类别名
        true_class_name = idx_to_class[true_label]
        pred_class_name = idx_to_class[pred_label]

        # 尝试用 emotion_map 获取更友好的名称（如果标签ID匹配）
        true_class_display = emotion_map.get(true_label, true_class_name)
        pred_class_display = emotion_map.get(pred_label, pred_class_name)

        results.append({
            'id': i,
            'split': split_name,
            'image_name': image_name,
            'true_label': true_label,
            'true_class': true_class_display,
            'pred_label': pred_label,
            'pred_class': pred_class_display
        })

        # 收集标签用于准确率计算（使用真实标签ID）
        true_labels_all.append(true_label)
        pred_labels_all.append(pred_label)

    return results, true_labels_all, pred_labels_all

# 主函数
if __name__ == '__main__':
    dataset_root = f'E:/python_code/Dataset/facial_emotion/AffectNet/{data}'  # 修改为你的数据根目录
    all_results = []
    all_true_labels = []
    all_pred_labels = []

    for split in ['val']:
        split_dir = os.path.join(dataset_root, split)
        if not os.path.exists(split_dir):
            print(f"⚠️ 跳过未找到的目录: {split_dir}")
            continue
        split_results, true_labels, pred_labels = predict_dataset(split_dir, split)
        all_results.extend(split_results)
        all_true_labels.extend(true_labels)
        all_pred_labels.extend(pred_labels)

    # === 计算并打印准确率 ===
    total_correct = sum(1 for t, p in zip(all_true_labels, all_pred_labels) if t == p)
    total_samples = len(all_true_labels)
    overall_acc = total_correct / total_samples if total_samples > 0 else 0.0

    print("\n" + "="*70)
    print(f"📊 整体准确率 (Overall Accuracy): {overall_acc:.4f} ({total_correct}/{total_samples})")
    print("="*70)

    # 每类准确率
    per_class_correct = defaultdict(int)
    per_class_total = defaultdict(int)
    for t, p in zip(all_true_labels, all_pred_labels):
        per_class_total[t] += 1
        if t == p:
            per_class_correct[t] += 1

    # 获取 ImageFolder 的真实映射用于打印
    temp_dataset = datasets.ImageFolder(root=os.path.join(dataset_root, 'val'), transform=transform)
    real_idx_to_class = {v: k for k, v in temp_dataset.class_to_idx.items()}

    print("\n📈 各类别表情识别准确率 (基于 ImageFolder 实际类别顺序):")
    print("-" * 70)
    for class_id in sorted(per_class_total.keys()):
        class_name = real_idx_to_class[class_id]
        total = per_class_total[class_id]
        correct = per_class_correct[class_id]
        if total > 0:
            acc = correct / total
            print(f"{class_id:>2d} ({class_name:>12}): {acc:.4f} ({correct:>5d}/{total:>5d})")
        else:
            print(f"{class_id:>2d} ({class_name:>12}): N/A      (    0/    0)")
    # ==========================

    # 保存结果到 Excel
    df = pd.DataFrame(all_results)
    output_file = f"predict_{data}_val.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n✅ 所有预测完成，结果保存在 {output_file}")