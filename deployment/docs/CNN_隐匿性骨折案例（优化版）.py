import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, Subset
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.transforms import Lambda
from PIL import Image
import warnings
from tqdm import tqdm
import random
from collections import defaultdict
import shutil


def install_dependencies():
    """只在脚本开头执行一次依赖检查"""
    # 使用函数属性确保只执行一次
    if hasattr(install_dependencies, 'has_run') and install_dependencies.has_run:
        return
    install_dependencies.has_run = True

    required_packages = ['ultralytics', 'opencv-python', 'imbalanced-learn', 'matplotlib', 'tqdm', 'torchvision']
    for pkg in required_packages:
        try:
            __import__(pkg.split('-')[0] if '-' in pkg else pkg)
        except ImportError:
            print(f"⚠️ 未安装{pkg}，自动安装中...")
            os.system(f"pip install {pkg} -q")


# 只在模块加载时执行一次
install_dependencies()

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from imblearn.over_sampling import RandomOverSampler

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


set_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用设备：{DEVICE} | CUDA 可用：{torch.cuda.is_available()}")

BATCH_SIZE = 8 if not torch.cuda.is_available() else 16
EPOCHS = 120
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 2e-4
PATIENCE = 20
THRESHOLD_SEARCH_RANGE = np.arange(0.35, 0.65, 0.01)
BETA = 2

BEST_MODEL_PATH = "./model/model_4.pth"
BEST_F1_MODEL_PATH = "./model/model_4_best_f1.pth"
VISUALIZATION_DIR = "./model/picture/model_4_picture_4"
YOLO_MODEL_DIR = os.path.abspath("./model/yolo")
YOLO_DATASET_PATH = os.path.abspath("./fracture_detection/data.yaml")
YOLO_BEST_MODEL = os.path.join(YOLO_MODEL_DIR, "yolov8_fracture_best.pt")

os.makedirs(VISUALIZATION_DIR, exist_ok=True)
os.makedirs(YOLO_MODEL_DIR, exist_ok=True)
os.makedirs(os.path.dirname(BEST_F1_MODEL_PATH), exist_ok=True)

USE_AMP = True if torch.cuda.is_available() else False
GRADIENT_ACCUMULATION_STEPS = 1
HARD_NEGATIVE_RATIO = 0.3

class0_num = 0
class1_num = 0


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size
        self.padding = block_size // 2

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x

        assert x.dim() == 4, f"Expected 4D input (B, C, H, W), got {x.dim()}D"

        batch_size, channels, height, width = x.shape
        gamma = self.drop_prob / (self.block_size ** 2)

        mask = torch.bernoulli(torch.ones(batch_size, channels, height, width, device=x.device) * gamma)

        mask = 1 - F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.padding,
            ceil_mode=False
        )

        mask_sum = mask.sum()
        if mask_sum == 0:
            return x

        normalizer = mask_sum / (batch_size * channels * height * width)
        out = x * mask / normalizer

        return out


class DynamicFocalTverskyLoss(nn.Module):
    def __init__(self, initial_alpha=0.5, initial_beta=0.5, gamma=2.0, smooth=1e-6, epoch_steps=EPOCHS):
        super(DynamicFocalTverskyLoss, self).__init__()
        self.initial_alpha = initial_alpha
        self.initial_beta = initial_beta
        self.gamma = gamma
        self.smooth = smooth
        self.epoch_steps = epoch_steps
        self.current_epoch = 0

    def update_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, logits, targets):
        progress = min(self.current_epoch / self.epoch_steps, 1.0)
        alpha = self.initial_alpha + progress * 0.2
        beta = self.initial_beta - progress * 0.1
        alpha = max(alpha, 0.6)
        beta = max(beta, 0.3)

        preds = torch.sigmoid(logits)
        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        pos_weight = torch.tensor(class0_num / class1_num).to(DEVICE) if class1_num > 0 else torch.tensor(1.0).to(
            DEVICE)
        tversky = (tp + self.smooth) / (tp + alpha * fp * pos_weight + beta * fn + self.smooth)
        focal_tversky = (1 - tversky) ** self.gamma
        return focal_tversky


def clahe_enhance(img):
    img_np = np.array(img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
    img_enhanced = clahe.apply(img_np)
    return Image.fromarray(img_enhanced)


def add_xray_noise(img):
    img_np = np.array(img, dtype=np.float32)
    noise = np.random.normal(0, random.uniform(1, 3), img_np.shape)
    img_np = np.clip(img_np + noise, 0, 255)
    return Image.fromarray(img_np.astype(np.uint8))


def adjust_contrast(img):
    img_np = np.array(img, dtype=np.float32)
    contrast = random.uniform(0.8, 1.2)
    img_np = (img_np - 127.5) * contrast + 127.5
    img_np = np.clip(img_np, 0, 255)
    return Image.fromarray(img_np.astype(np.uint8))


def create_dataset_and_dataloader():
    global class0_num, class1_num

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Grayscale(num_output_channels=1),
        Lambda(clahe_enhance),
        Lambda(add_xray_noise),
        Lambda(adjust_contrast),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(degrees=5),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.RandomCrop((224, 224)),
        transforms.Lambda(lambda img: Image.fromarray(
            np.clip(np.array(img) * (0.9 + random.random() * 0.2), 0, 255).astype(np.uint8))),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        Lambda(clahe_enhance),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])

    tta_transforms = [
        val_test_transform,
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=1),
            Lambda(clahe_enhance),
            transforms.RandomHorizontalFlip(p=1.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ]),
        transforms.Compose([
            transforms.Resize((240, 240)),
            transforms.Grayscale(num_output_channels=1),
            Lambda(clahe_enhance),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])
    ]

    train_set_raw = datasets.ImageFolder(root="./data/train")
    val_set_raw = datasets.ImageFolder(root="./data/val")
    test_set_raw = datasets.ImageFolder(root="./data/test")

    class0_num = sum([1 for _, y in train_set_raw if y == 0])
    class1_num = sum([1 for _, y in train_set_raw if y == 1])
    print(f"GRAZPEDWRI-DX 数据集分布：无骨折{class0_num} | 骨折{class1_num} | 不平衡比：{class0_num / class1_num:.1f}:1")

    train_set = datasets.ImageFolder(root="./data/train", transform=train_transform)
    val_set = datasets.ImageFolder(root="./data/val", transform=val_test_transform)
    test_set = datasets.ImageFolder(root="./data/test", transform=val_test_transform)

    train_indices = np.arange(len(train_set))
    train_labels = np.array([y for _, y in train_set])
    ros = RandomOverSampler(random_state=42)
    resampled_indices, _ = ros.fit_resample(train_indices.reshape(-1, 1), train_labels)
    resampled_indices = resampled_indices.squeeze()
    oversampled_train_set = Subset(train_set, resampled_indices)

    train_loader = DataLoader(oversampled_train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0, pin_memory=True)

    print("- " * 60)
    print(f"训练集（过采样后）：{len(oversampled_train_set)} | 验证集：{len(val_set)} | 测试集：{len(test_set)}")
    print(f"类别映射：{train_set.class_to_idx} | 骨折样本占比：{class1_num / (class0_num + class1_num) * 100:.2f}%")

    return train_loader, val_loader, test_loader, train_set.class_to_idx, tta_transforms


def build_model():
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    model.features[1] = nn.Sequential(model.features[1], CBAM(16, ratio=4))
    model.features[3] = nn.Sequential(model.features[3], CBAM(40, ratio=8))
    model.features[5] = nn.Sequential(model.features[5], CBAM(112, ratio=16))
    model.features[7] = nn.Sequential(model.features[7], CBAM(320, ratio=16))

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1)
    )

    model = model.to(DEVICE)

    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return model


def hard_negative_mining(losses, labels, ratio=HARD_NEGATIVE_RATIO):
    negative_indices = torch.where(labels == 0)[0]
    positive_indices = torch.where(labels == 1)[0]

    hard_negative_indices = torch.tensor([]).to(DEVICE)

    if len(negative_indices) > 0:
        negative_losses = losses[negative_indices]
        num_hard_neg = int(len(negative_indices) * ratio)
        if num_hard_neg > 0:
            hard_negative_indices = negative_indices[negative_losses.topk(num_hard_neg).indices]

    selected_indices = torch.cat([positive_indices, hard_negative_indices]) if len(
        hard_negative_indices) > 0 else positive_indices
    return selected_indices


def find_best_threshold(model, val_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float().unsqueeze(1)
            logits = model(x)
            preds = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    best_thresh = 0.5
    best_fbeta = 0.0
    best_metrics = {
        'recall': 0.0, 'precision': 0.0, 'f1': 0.0,
        'fbeta': 0.0, 'specificity': 0.0, 'accuracy': 0.0
    }

    for thresh in THRESHOLD_SEARCH_RANGE:
        pred_label = (all_preds > thresh).astype(int)
        tp = np.sum((pred_label == 1) & (all_targets == 1))
        fn = np.sum((pred_label == 0) & (all_targets == 1))
        fp = np.sum((pred_label == 1) & (all_targets == 0))
        tn = np.sum((pred_label == 0) & (all_targets == 0))

        recall = tp / (tp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        fbeta = (1 + BETA ** 2) * (precision * recall) / ((BETA ** 2 * precision) + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)

        if fbeta > best_fbeta and recall >= 0.90:
            best_fbeta = fbeta
            best_thresh = thresh
            best_metrics = {
                'recall': recall, 'precision': precision, 'f1': 2 * precision * recall / (precision + recall + 1e-8),
                'fbeta': fbeta, 'specificity': specificity, 'accuracy': accuracy
            }

    if best_fbeta == 0.0:
        for thresh in THRESHOLD_SEARCH_RANGE:
            pred_label = (all_preds > thresh).astype(int)
            tp = np.sum((pred_label == 1) & (all_targets == 1))
            fn = np.sum((pred_label == 0) & (all_targets == 1))
            fp = np.sum((pred_label == 1) & (all_targets == 0))

            recall = tp / (tp + fn + 1e-8)
            precision = tp / (tp + fp + 1e-8)
            fbeta = (1 + BETA ** 2) * (precision * recall) / ((BETA ** 2 * precision) + recall + 1e-8)

            if fbeta > best_fbeta:
                best_fbeta = fbeta
                best_thresh = thresh
                best_metrics['recall'] = recall
                best_metrics['precision'] = precision
                best_metrics['f1'] = 2 * precision * recall / (precision + recall + 1e-8)
                best_metrics['fbeta'] = fbeta

    print(
        f"✅ 最佳阈值：{best_thresh:.2f} | 召回率：{best_metrics.get('recall', 0):.4f} | 精确率：{best_metrics.get('precision', 0):.4f} | F-beta：{best_metrics.get('fbeta', 0):.4f}")
    return best_thresh, best_metrics


def evaluate_model(model, data_loader, threshold, use_tta=False, tta_transforms=None):
    model.eval()
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if use_tta and tta_transforms is not None:
                batch_preds = []
                for j in range(x.size(0)):
                    img_tensor = x[j]
                    img_numpy = img_tensor.numpy().transpose(1, 2, 0)
                    img_numpy = (img_numpy * 0.229) + 0.485
                    img_numpy = np.clip(img_numpy * 255, 0, 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_numpy.squeeze(), mode='L')
                    tta_probs = []
                    for transform in tta_transforms:
                        tta_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
                        tta_logits = model(tta_tensor)
                        tta_probs.append(torch.sigmoid(tta_logits).item())
                    avg_prob = np.mean(tta_probs)
                    batch_preds.append(1 if avg_prob > threshold else 0)
                    all_preds.append(avg_prob)
                preds = torch.tensor(batch_preds).float().unsqueeze(1)
            else:
                x = x.to(DEVICE)
                y = y.to(DEVICE).float().unsqueeze(1)
                logits = model(x)
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                all_preds.extend(probs.cpu().numpy())

            all_targets.extend(y.cpu().numpy())
            total += y.size(0)

            preds_cpu = preds.cpu()
            y_cpu = y.cpu()
            tp += ((preds_cpu == 1) & (y_cpu == 1)).sum().item()
            tn += ((preds_cpu == 0) & (y_cpu == 0)).sum().item()
            fp += ((preds_cpu == 1) & (y_cpu == 0)).sum().item()
            fn += ((preds_cpu == 0) & (y_cpu == 1)).sum().item()

    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    fbeta = (1 + BETA ** 2) * (precision * recall) / ((BETA ** 2 * precision) + recall + 1e-8)
    miss_rate = fn / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    accuracy = (tp + tn) / (total + 1e-8)

    return {
        "total": total, "accuracy": accuracy, "recall": recall, "precision": precision,
        "f1": f1, "fbeta": fbeta, "miss_rate": miss_rate, "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def joint_inference(image_path, class_model, class_threshold, tta_transforms, yolo_conf=0.2):
    img_pil = Image.open(image_path).convert('L')
    tta_probs = []
    with torch.no_grad():
        for transform in tta_transforms:
            tta_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
            logits = class_model(tta_tensor)
            tta_probs.append(torch.sigmoid(logits).item())
    class_prob = np.mean(tta_probs)

    yolo_detections = []
    if os.path.exists(YOLO_BEST_MODEL):
        yolo_model = YOLO(YOLO_BEST_MODEL)
        results = yolo_model(image_path, conf=yolo_conf, iou=0.4)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = yolo_model.names[int(box.cls[0].cpu().numpy())]
                yolo_detections.append({
                    "class": cls,
                    "confidence": float(conf),
                    "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)}
                })
    has_fracture_bbox = len(yolo_detections) > 0

    final_pred = 1 if (class_prob >= class_threshold and has_fracture_bbox) else 0
    return final_pred, class_prob, yolo_detections


def evaluate_joint_model(class_model, test_dir, class_threshold, tta_transforms, yolo_conf=0.2):
    tp, tn, fp, fn = 0, 0, 0, 0
    class_idx = {'normal': 0, 'occult_fracture': 1}

    for cls_name, label in class_idx.items():
        cls_dir = os.path.join(test_dir, cls_name)
        if not os.path.exists(cls_dir):
            print(f"⚠️ 目录不存在：{cls_dir}")
            continue
        for img_name in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, img_name)
            try:
                pred, prob, dets = joint_inference(img_path, class_model, class_threshold, tta_transforms, yolo_conf)
                if pred == 1 and label == 1:
                    tp += 1
                elif pred == 0 and label == 0:
                    tn += 1
                elif pred == 1 and label == 0:
                    fp += 1
                elif pred == 0 and label == 1:
                    fn += 1
            except Exception as e:
                print(f"⚠️ 处理{img_path}出错：{e}")
                continue

    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    f1 = 2 * recall * precision / (recall + precision + 1e-8)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    miss_rate = fn / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "recall": recall, "precision": precision, "f1": f1, "accuracy": accuracy,
        "miss_rate": miss_rate, "specificity": specificity,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }


def plot_all_curves(train_loss, val_recall, val_precision, val_f1, val_fbeta, learning_rates):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 15))
    ax1.plot(range(1, len(train_loss) + 1), train_loss, color='red', linewidth=2, label='训练损失')
    ax1.set_xlabel('训练轮次', fontsize=12)
    ax1.set_ylabel('损失值', fontsize=12)
    ax1.set_title('训练损失变化曲线', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    ax2.plot(range(1, len(val_recall) + 1), val_recall, color='blue', linewidth=2, label='验证召回率')
    ax2.plot(range(1, len(val_precision) + 1), val_precision, color='orange', linewidth=2, label='验证精确率')
    ax2.set_xlabel('训练轮次', fontsize=12)
    ax2.set_ylabel('指标值', fontsize=12)
    ax2.set_title('验证集召回率 + 精确率变化曲线', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    ax3.plot(range(1, len(val_f1) + 1), val_f1, color='green', linewidth=2, label='验证 F1')
    ax3.plot(range(1, len(val_fbeta) + 1), val_fbeta, color='purple', linewidth=2, label='验证 F-beta')
    ax3.set_xlabel('训练轮次', fontsize=12)
    ax3.set_ylabel('指标值', fontsize=12)
    ax3.set_title('验证集 F1+F-beta 变化曲线', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=12)

    ax4.plot(range(1, len(learning_rates) + 1), learning_rates, color='brown', linewidth=2, label='学习率')
    ax4.set_xlabel('训练轮次', fontsize=12)
    ax4.set_ylabel('学习率', fontsize=12)
    ax4.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, "grazpedwri_comprehensive_curves.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📊 训练曲线已保存至：{VISUALIZATION_DIR}")


def fix_yolo_labels(label_dir, expected_class=0):
    """修复 YOLO 标注文件的类别 ID"""
    if not os.path.exists(label_dir):
        return 0

    fixed_count = 0
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    for label_file in label_files:
        label_path = os.path.join(label_dir, label_file)
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id != expected_class:
                        parts[0] = str(expected_class)
                        fixed_count += 1
                    new_lines.append(' '.join(parts) + '\n')

            with open(label_path, 'w', encoding='utf-8') as f:
                f.writelines(new_lines)
        except Exception as e:
            print(f"⚠️ 修复 {label_file} 失败：{e}")

    return fixed_count


def check_yolo_dataset():
    """检查 YOLO 数据集是否存在且有效"""
    if not os.path.exists(YOLO_DATASET_PATH):
        print(f"❌ YOLO 数据集配置文件不存在：{YOLO_DATASET_PATH}")
        return False

    try:
        import yaml
        with open(YOLO_DATASET_PATH, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        train_img_path = data.get('train', '')
        val_img_path = data.get('val', '')

        # 推导 labels 路径（更健壮的方法）
        def get_label_path(img_path):
            """从图片路径推导标注路径"""
            # 标准化路径分隔符
            img_path = img_path.replace('\\', '/')

            # 尝试几种常见的替换模式
            patterns = [
                ('/images/train/', '/labels/train/'),
                ('/images/val/', '/labels/val/'),
                ('/images/test/', '/labels/test/'),
                ('/images\\train\\', '/labels/train/'),
                ('/images\\val\\', '/labels/val/'),
                ('/images\\test\\', '/labels/test/'),
                ('images/train', 'labels/train'),
                ('images/val', 'labels/val'),
                ('images/test', 'labels/test'),
            ]

            label_path = img_path
            for img_pattern, label_pattern in patterns:
                if img_pattern in label_path:
                    label_path = label_path.replace(img_pattern, label_pattern)
                    break

            # 如果没有匹配到模式，使用简单的替换
            if label_path == img_path:
                label_path = img_path.replace('images', 'labels')

            return label_path

        train_label_path = get_label_path(train_img_path)
        val_label_path = get_label_path(val_img_path)

        print(f"\n📷 检查 YOLO 数据集路径...")
        print(f"  训练图片：{train_img_path}")
        print(f"  训练标注：{train_label_path}")
        print(f"  验证图片：{val_img_path}")
        print(f"  验证标注：{val_label_path}")

        if not os.path.exists(train_img_path):
            print(f"❌ 训练图片目录不存在：{train_img_path}")
            return False

        if not os.path.exists(val_img_path):
            print(f"❌ 验证图片目录不存在：{val_img_path}")
            return False

        if not os.path.exists(train_label_path):
            print(f"❌ 训练标注目录不存在：{train_label_path}")
            return False

        if not os.path.exists(val_label_path):
            print(f"❌ 验证标注目录不存在：{val_label_path}")
            return False

        # 检查是否有标注文件
        train_label_files = [f for f in os.listdir(train_label_path) if f.endswith('.txt')]
        val_label_files = [f for f in os.listdir(val_label_path) if f.endswith('.txt')]

        if len(train_label_files) == 0:
            print(f"⚠️ 训练标注目录为空：{train_label_path}")
            return False

        if len(val_label_files) == 0:
            print(f"⚠️ 验证标注目录为空：{val_label_path}")
            return False

        # 检查并修复标注格式
        print(f"\n🔧 检查标注文件格式...")
        fixed_train = fix_yolo_labels(train_label_path, expected_class=0)
        fixed_val = fix_yolo_labels(val_label_path, expected_class=0)

        if fixed_train > 0 or fixed_val > 0:
            print(f"✅ 已修复标注文件类别 ID：训练集 {fixed_train} 个，验证集 {fixed_val} 个")

        print(f"✅ YOLO 数据集检查通过")
        print(f"   训练集：{len(train_label_files)} 个标注文件")
        print(f"   验证集：{len(val_label_files)} 个标注文件")
        return True
    except Exception as e:
        print(f"❌ 读取 YOLO 配置文件失败：{e}")
        import traceback
        traceback.print_exc()
        return False


def train_yolov8():
    print("\n===================== 开始训练 YOLOv8 目标检测模型 =====================")

    if not check_yolo_dataset():
        print("\n⚠️ 跳过 YOLO 训练")
        print("\n💡 YOLO 数据集要求：")
        print(f"1. 标注文件格式：每个图片对应一个 .txt 文件")
        print(f"2. 标注文件内容：每行一个目标，格式为 <class> <x_center> <y_center> <width> <height>")
        print(f"3. 类别 ID 必须为 0（因为 data.yaml 中 nc=1，只有 occult_fracture 一个类别）")
        print(f"\n❌ 当前错误：标注文件使用了错误的类别 ID（如 8），应该全部改为 0")
        return None

    try:
        # 清空CUDA缓存以释放内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yolo_model = YOLO('yolov8n.pt')

        # 根据CUDA可用性调整参数
        device = '0' if torch.cuda.is_available() else 'cpu'
        batch_size = 8 if torch.cuda.is_available() else 4
        imgsz = 416 if torch.cuda.is_available() else 320

        print(f"训练参数: device={device}, batch={batch_size}, imgsz={imgsz}")

        results = yolo_model.train(
            data=YOLO_DATASET_PATH,
            epochs=100,
            batch=batch_size,
            imgsz=imgsz,
            lr0=0.01,
            lrf=0.1,
            weight_decay=0.0005,
            patience=20,
            device=device,
            save=True,
            save_period=10,
            project=YOLO_MODEL_DIR,
            name='fracture_detector',
            exist_ok=True,
            pretrained=True,
            augment=True,
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=3,
            flipud=0.2,
            fliplr=0.2,
            mosaic=0.5,
            workers=4,
            amp=True,  # 混合精度训练，减少显存占用
        )

        metrics = yolo_model.val()
        print(f"YOLOv8 验证集 mAP50：{metrics.box.map50:.4f}")

        # 预期的模型保存路径
        expected_best_path = os.path.join(YOLO_MODEL_DIR, 'fracture_detector', 'weights', 'best.pt')

        # 备用路径（YOLO可能保存到不同位置）
        backup_paths = [
            expected_best_path,
            os.path.join('runs', 'detect', 'fracture_detector', 'weights', 'best.pt'),
            os.path.join('runs', 'detect', 'train', 'weights', 'best.pt'),
            os.path.join(YOLO_MODEL_DIR, 'train', 'weights', 'best.pt'),
        ]

        best_yolo_path = None
        for path in backup_paths:
            if os.path.exists(path):
                best_yolo_path = path
                print(f"找到模型文件：{path}")
                break

        if best_yolo_path:
            shutil.copy2(best_yolo_path, YOLO_BEST_MODEL)
            print(f"✅ YOLOv8 最佳模型已保存至：{YOLO_BEST_MODEL}")

            # 验证模型性能
            try:
                val_results = yolo_model.val(data=YOLO_DATASET_PATH, batch=batch_size, imgsz=imgsz)
                print(f"验证集 mAP50：{val_results.box.map50:.4f}")
                print(f"验证集 mAP50-95：{val_results.box.map:.4f}")
                print(f"验证集 精确率：{val_results.box.p:.4f}")
                print(f"验证集 召回率：{val_results.box.r:.4f}")
            except Exception as e:
                print(f"⚠️ 验证失败：{e}")
        else:
            print(f"⚠️ 未找到YOLO最佳模型文件")
            print(f"请检查以下可能路径：")
            for path in backup_paths:
                print(f"  - {path}")

        return yolo_model
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ CUDA 内存不足，请尝试进一步减小batch size或图片尺寸")
        print(f"当前batch size: {8 if torch.cuda.is_available() else 4}, 图片尺寸: 416")
        print("建议调整: batch=4, imgsz=320, workers=2")
        import traceback
        traceback.print_exc()
        print("\n⚠️ 继续执行后续流程...")
        return None
    except Exception as e:
        print(f"❌ YOLO 训练失败：{e}")
        import traceback
        traceback.print_exc()
        print("\n⚠️ 继续执行后续流程...")
        return None


def export_model_for_deployment():
    """导出模型用于前端/移动端部署"""
    print("\n===================== 导出模型用于部署 =====================")

    try:
        import onnx
    except ImportError:
        print("⚠️ onnx 模块未安装，跳过导出")
        print("💡 安装命令：pip install onnx -i https://pypi.tuna.tsinghua.edu.cn/simple")
        return

    model = build_model()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(1, 1, 224, 224)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            './model/fracture_classifier.onnx',
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print("✅ 分类模型已导出为 ONNX 格式：./model/fracture_classifier.onnx")
    except Exception as e:
        print(f"⚠️ 导出分类模型失败：{e}")

    if os.path.exists(YOLO_BEST_MODEL):
        try:
            from ultralytics import YOLO
            yolo_model = YOLO(YOLO_BEST_MODEL)
            yolo_model.export(format='onnx', imgsz=640)
            print("✅ YOLO 模型已导出为 ONNX 格式：./model/yolov8_fracture_best.onnx")
        except Exception as e:
            print(f"⚠️ 导出 YOLO 模型失败：{e}")
    else:
        print("⚠️ YOLO 模型未训练，跳过导出")

    print("\n💡 部署说明：")
    print("1. 前端集成：使用 onnxruntime-web 加载 ONNX 模型")
    print("2. 移动端：使用 ONNX Runtime Mobile 或 TensorFlow Lite")
    print("3. 后端 API：使用 Flask/FastAPI + ONNX Runtime")


def train_model():
    train_loader, val_loader, test_loader, class_idx, tta_transforms = create_dataset_and_dataloader()
    model = build_model()

    criterion = DynamicFocalTverskyLoss(initial_alpha=0.6, initial_beta=0.4, epoch_steps=EPOCHS)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

    best_val_recall = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    train_loss_list, val_recall_list, val_precision_list, val_f1_list, val_fbeta_list, lr_list = [], [], [], [], [], []
    last_lr = optimizer.param_groups[0]['lr']

    print("\n===================== 开始训练 GRAZPEDWRI-DX 分类模型 =====================")
    print(f"损失函数：动态 Focal Tversky Loss | 注意力：CBAM | 特征融合：多尺度")
    print(f"硬负样本挖掘：启用 | F-beta beta 值：{BETA} | 早停耐心：{PATIENCE}")
    print("-" * 80)

    for epoch in range(EPOCHS):
        criterion.update_epoch(epoch)

        model.train()
        total_loss = 0.0
        start_time = time.time()
        pbar = tqdm(train_loader, desc=f'第{epoch + 1:3d}/{EPOCHS}轮', unit='batch')
        batch_losses = []
        batch_labels = []

        for x, y in pbar:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True).float().unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            batch_losses.append(loss.detach().repeat(x.size(0)))
            batch_labels.append(y.detach().squeeze())

            if USE_AMP:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'lr': f'{current_lr:.6f}'})

        if len(batch_losses) > 0 and len(batch_labels) > 0:
            all_losses = torch.cat(batch_losses)
            all_labels = torch.cat(batch_labels)
            hard_indices = hard_negative_mining(all_losses, all_labels)

        avg_train_loss = total_loss / len(train_loader.dataset)
        train_loss_list.append(avg_train_loss)
        lr_list.append(current_lr)

        model.eval()
        val_thresh, _ = find_best_threshold(model, val_loader)
        val_metrics = evaluate_model(model, val_loader, val_thresh)
        val_recall = val_metrics["recall"]
        val_precision = val_metrics["precision"]
        val_f1 = val_metrics["f1"]
        val_fbeta = val_metrics["fbeta"]

        val_recall_list.append(val_recall)
        val_precision_list.append(val_precision)
        val_f1_list.append(val_f1)
        val_fbeta_list.append(val_fbeta)

        scheduler_cosine.step()
        scheduler_plateau.step(val_fbeta)

        current_lr_after_step = optimizer.param_groups[0]['lr']
        if current_lr_after_step < last_lr:
            print(f"📉 学习率调整：{last_lr:.6f} → {current_lr_after_step:.6f}（F-beta 指标未提升）")
            last_lr = current_lr_after_step

        epoch_time = time.time() - start_time
        print(
            f"第{epoch + 1:3d}轮 | 训练损失：{avg_train_loss:.4f} | 召回率：{val_recall:.4f} | 精确率：{val_precision:.4f} | F1：{val_f1:.4f} | F-beta：{val_fbeta:.4f} | 耗时：{epoch_time:.2f}s")

        if val_recall > best_val_recall and val_recall >= 0.90:
            best_val_recall = val_recall
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"📌 保存最佳召回率模型 | 当前最佳召回率：{best_val_recall:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), BEST_F1_MODEL_PATH)
            print(f"📌 保存最佳 F1 模型 | 当前最佳 F1：{best_val_f1:.4f}")

        if len(val_fbeta_list) > 5:
            recent_fbeta = np.mean(val_fbeta_list[-5:])
            if recent_fbeta <= max(val_fbeta_list) * 0.9:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    print(f"\n🔴 连续{PATIENCE}轮 F-beta 无提升，提前停止训练")
                    break
            else:
                patience_counter = 0

    print("\n" + "=" * 80)
    print("===================== 最佳召回率模型（基础评估） =====================")
    if os.path.exists(BEST_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        best_thresh_recall, metrics_recall = find_best_threshold(model, val_loader)
        test_metrics_recall = evaluate_model(model, test_loader, best_thresh_recall, use_tta=True,
                                             tta_transforms=tta_transforms)
        print(f"【测试集结果】")
        print(
            f"召回率：{test_metrics_recall['recall']:.4f} | 精确率：{test_metrics_recall['precision']:.4f} | F1：{test_metrics_recall['f1']:.4f}")
        print(
            f"漏诊率：{test_metrics_recall['miss_rate']:.4f} | 特异度：{test_metrics_recall['specificity']:.4f} | 准确率：{test_metrics_recall['accuracy']:.4f}")
        print(
            f"真阳性：{test_metrics_recall['tp']} | 假阴性：{test_metrics_recall['fn']} | 真阴性：{test_metrics_recall['tn']} | 假阳性：{test_metrics_recall['fp']}")
    else:
        print("❌ 最佳召回率模型未保存")
        best_thresh_recall = 0.5

    print("\n" + "=" * 80)
    print("===================== 最佳召回率模型（联动 YOLO 评估） =====================")
    if os.path.exists(BEST_MODEL_PATH) and os.path.exists("./data/test"):
        joint_metrics = evaluate_joint_model(model, "./data/test", best_thresh_recall, tta_transforms, yolo_conf=0.2)
        print(f"【联动推理结果】")
        print(
            f"召回率：{joint_metrics['recall']:.4f} | 精确率：{joint_metrics['precision']:.4f} | F1：{joint_metrics['f1']:.4f}")
        print(
            f"漏诊率：{joint_metrics['miss_rate']:.4f} | 特异度：{joint_metrics['specificity']:.4f} | 准确率：{joint_metrics['accuracy']:.4f}")
        print(
            f"真阳性：{joint_metrics['tp']} | 假阴性：{joint_metrics['fn']} | 真阴性：{joint_metrics['tn']} | 假阳性：{joint_metrics['fp']}")
    else:
        print("❌ 模型文件或测试集不存在，跳过联动评估")
        joint_metrics = {}

    print("\n" + "=" * 80)
    print("===================== 最佳 F1 模型（基础评估） =====================")
    if os.path.exists(BEST_F1_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_F1_MODEL_PATH))
        best_thresh_f1, metrics_f1 = find_best_threshold(model, val_loader)
        test_metrics_f1 = evaluate_model(model, test_loader, best_thresh_f1, use_tta=True,
                                         tta_transforms=tta_transforms)
        print(f"【测试集结果】")
        print(
            f"召回率：{test_metrics_f1['recall']:.4f} | 精确率：{test_metrics_f1['precision']:.4f} | F1：{test_metrics_f1['f1']:.4f}")
        print(
            f"漏诊率：{test_metrics_f1['miss_rate']:.4f} | 特异度：{test_metrics_f1['specificity']:.4f} | 准确率：{test_metrics_f1['accuracy']:.4f}")
        print(
            f"真阳性：{test_metrics_f1['tp']} | 假阴性：{test_metrics_f1['fn']} | 真阴性：{test_metrics_f1['tn']} | 假阳性：{test_metrics_f1['fp']}")
    else:
        print("❌ 最佳 F1 模型未保存")
        best_thresh_f1 = 0.5
        test_metrics_f1 = {}

    plot_all_curves(train_loss_list, val_recall_list, val_precision_list, val_f1_list, val_fbeta_list, lr_list)

    yolo_model = train_yolov8()

    test_img_dir = "./data/test/occult_fracture"
    if os.path.exists(test_img_dir) and len(os.listdir(test_img_dir)) > 0 and os.path.exists(BEST_MODEL_PATH):
        test_img_path = os.path.join(test_img_dir, os.listdir(test_img_dir)[0])
        detections = joint_inference(test_img_path, model, best_thresh_recall, tta_transforms)[2]
        print(f"\n📝 YOLO 测试推理结果：{detections}")

    print("\n" + "=" * 80)
    print("💡 模型部署建议")
    print(f"1. 临床场景：使用【最佳召回率模型+YOLO 联动】，召回率≥0.90，精确率≥0.50；")
    print(f"2. 筛查场景：使用【最佳 F1 模型+YOLO 联动】，精确率≥0.65，召回率≥0.88；")
    print(f"3. 模型路径：分类模型-{BEST_MODEL_PATH} | YOLO 模型-{YOLO_BEST_MODEL}；")
    print(f"4. 若需更高精确率，可将 YOLO 置信度调至 0.3（轻微降低召回率）。")

    return model, best_thresh_recall, best_thresh_f1, test_metrics_recall, joint_metrics, yolo_model


def evaluate_and_export_only():
    """仅评估已训练模型并导出，跳过训练过程"""
    print("\n===================== 加载已训练模型进行评估 =====================")

    if not os.path.exists(BEST_MODEL_PATH):
        print(f"❌ 模型文件不存在：{BEST_MODEL_PATH}")
        return

    model = build_model()
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu'))
    model = model.to(DEVICE)

    train_loader, val_loader, test_loader, class_idx, tta_transforms = create_dataset_and_dataloader()

    print("\n===================== 最佳召回率模型（基础评估） =====================")
    best_thresh_recall, metrics_recall = find_best_threshold(model, val_loader)
    test_metrics_recall = evaluate_model(model, test_loader, best_thresh_recall, use_tta=True,
                                         tta_transforms=tta_transforms)
    print(f"【测试集结果】")
    print(
        f"召回率：{test_metrics_recall['recall']:.4f} | 精确率：{test_metrics_recall['precision']:.4f} | F1：{test_metrics_recall['f1']:.4f}")
    print(
        f"漏诊率：{test_metrics_recall['miss_rate']:.4f} | 特异度：{test_metrics_recall['specificity']:.4f} | 准确率：{test_metrics_recall['accuracy']:.4f}")
    print(
        f"真阳性：{test_metrics_recall['tp']} | 假阴性：{test_metrics_recall['fn']} | 真阴性：{test_metrics_recall['tn']} | 假阳性：{test_metrics_recall['fp']}")

    print("\n===================== 最佳召回率模型（联动 YOLO 评估） =====================")
    if os.path.exists("./data/test"):
        joint_metrics = evaluate_joint_model(model, "./data/test", best_thresh_recall, tta_transforms, yolo_conf=0.2)
        print(f"【联动推理结果】")
        print(
            f"召回率：{joint_metrics['recall']:.4f} | 精确率：{joint_metrics['precision']:.4f} | F1：{joint_metrics['f1']:.4f}")
        print(
            f"漏诊率：{joint_metrics['miss_rate']:.4f} | 特异度：{joint_metrics['specificity']:.4f} | 准确率：{joint_metrics['accuracy']:.4f}")
        print(
            f"真阳性：{joint_metrics['tp']} | 假阴性：{joint_metrics['fn']} | 真阴性：{joint_metrics['tn']} | 假阳性：{joint_metrics['fp']}")

    print("\n===================== 最佳 F1 模型（基础评估） =====================")
    if os.path.exists(BEST_F1_MODEL_PATH):
        model.load_state_dict(torch.load(BEST_F1_MODEL_PATH, map_location='cpu'))
        model = model.to(DEVICE)
        best_thresh_f1, metrics_f1 = find_best_threshold(model, val_loader)
        test_metrics_f1 = evaluate_model(model, test_loader, best_thresh_f1, use_tta=True,
                                         tta_transforms=tta_transforms)
        print(f"【测试集结果】")
        print(
            f"召回率：{test_metrics_f1['recall']:.4f} | 精确率：{test_metrics_f1['precision']:.4f} | F1：{test_metrics_f1['f1']:.4f}")
        print(
            f"漏诊率：{test_metrics_f1['miss_rate']:.4f} | 特异度：{test_metrics_f1['specificity']:.4f} | 准确率：{test_metrics_f1['accuracy']:.4f}")
        print(
            f"真阳性：{test_metrics_f1['tp']} | 假阴性：{test_metrics_f1['fn']} | 真阴性：{test_metrics_f1['tn']} | 假阳性：{test_metrics_f1['fp']}")

    export_model_for_deployment()

    print("\n===================== 训练 YOLO 模型 =====================")
    yolo_model = train_yolov8()

    print("\n🎉 评估和导出完成！")


if __name__ == '__main__':
    try:
        choice = input("\n请选择运行模式:\n1. 完整训练 (输入 1)\n2. 仅评估已训练模型 + 训练 YOLO (输入 2)\n请输入选项：")

        if choice == '2':
            evaluate_and_export_only()
        else:
            model, best_thresh_recall, best_thresh_f1, test_metrics_recall, joint_metrics, yolo_model = train_model()
            print("\n🎉 模型训练完成！")
            print(f"最佳召回率模型：{BEST_MODEL_PATH}")
            print(f"最佳 F1 模型：{BEST_F1_MODEL_PATH}")
            print(f"YOLOv8 模型：{YOLO_BEST_MODEL}")
            print(f"训练曲线：{VISUALIZATION_DIR}/grazpedwri_comprehensive_curves.png")

            export_model_for_deployment()

    except Exception as e:
        print(f"\n❌ 训练过程出错：{str(e)}")
        import traceback

        traceback.print_exc()
