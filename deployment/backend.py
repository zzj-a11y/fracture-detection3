"""
儿童手腕隐匿性骨折检测系统后端
基于 FastAPI 实现模型推理 API
"""

import os
import sys
import logging
from pathlib import Path
import tempfile
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import warnings
import psutil
import gc
import asyncio
import concurrent.futures
warnings.filterwarnings('ignore')

# 检查 multipart 依赖
try:
    import multipart
except ImportError:
    print("错误: python-multipart 未安装。请运行: pip install python-multipart")
    sys.exit(1)

# FastAPI 相关
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
import uvicorn
from typing import Optional, List, Dict, Any
import json
import shutil
from datetime import datetime
import time

# 配置 PyTorch 安全全局变量（用于 PyTorch 2.6+）
try:
    import torch.serialization
    import torch.nn.modules.container
    import torch.nn

    # 添加必要的 PyTorch 类到安全全局列表
    torch.serialization.add_safe_globals([
        torch.nn.modules.container.Sequential,
        torch.nn.modules.container.ModuleList,
        torch.nn.modules.container.ModuleDict,
        torch.nn.Sequential,
        torch.nn.Conv2d,
        torch.nn.ConvTranspose2d,
        torch.nn.ReLU,
        torch.nn.MaxPool2d,
        torch.nn.AvgPool2d,
        torch.nn.AdaptiveAvgPool2d,
        torch.nn.AdaptiveMaxPool2d,
        torch.nn.Linear,
        torch.nn.BatchNorm2d,
        torch.nn.GroupNorm,
        torch.nn.LayerNorm,
        torch.nn.InstanceNorm2d,
        torch.nn.Dropout,
        torch.nn.Dropout2d,
        torch.nn.Sigmoid,
        torch.nn.modules.activation.SiLU,
        torch.nn.SiLU,
        torch.nn.LeakyReLU,
        torch.nn.modules.activation.LeakyReLU,
        torch.nn.Hardswish,
        torch.nn.modules.activation.Hardswish,
        torch.nn.GELU,
        torch.nn.modules.activation.GELU,
        torch.nn.Upsample,
        torch.nn.UpsamplingNearest2d,
        torch.nn.UpsamplingBilinear2d,
        torch.nn.Identity,
        torch.nn.ZeroPad2d,
        torch.nn.ReflectionPad2d,
        torch.nn.ReplicationPad2d,
        torch.nn.ConstantPad2d,
        torch.nn.Module,
        torch.nn.Parameter,
        torch.Tensor,
        torch._utils._rebuild_tensor_v2,
        torch._utils._rebuild_parameter,
        object  # 基础对象
    ])

    # 尝试导入 ultralytics 并添加具体类
    try:
        from ultralytics.nn.modules.conv import Conv, Concat
        torch.serialization.add_safe_globals([Conv, Concat])
        print("✅ 已添加 ultralytics.nn.modules.conv.Conv 和 Concat 到安全全局变量")
    except ImportError:
        pass  # ultralytics 未安装，跳过

    try:
        from ultralytics.nn.modules.block import Bottleneck, C2f, SPPF
        torch.serialization.add_safe_globals([Bottleneck, C2f, SPPF])
    except ImportError:
        pass

    try:
        from ultralytics.nn.modules.head import Detect
        torch.serialization.add_safe_globals([Detect])
    except ImportError:
        pass

    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
    except ImportError:
        pass  # ultralytics 未安装，跳过

    # 尝试导入 ultralytics.nn.modules.Conv (如果存在)
    try:
        from ultralytics.nn.modules import Conv as ConvModule
        torch.serialization.add_safe_globals([ConvModule])
        print("✅ 已添加 ultralytics.nn.modules.Conv 到安全全局变量")
    except ImportError:
        pass

    print("✅ PyTorch 安全全局变量配置完成")
except Exception as e:
    print(f"配置安全全局变量失败: {e}")

# YOLO 检测
try:
    from ultralytics import YOLO
except ImportError:
    print("警告：未安装 ultralytics，YOLO 检测将不可用")
    YOLO = None

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常量定义
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {DEVICE}")

# 获取当前文件所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 模型路径（使用绝对路径）
CLASSIFICATION_MODEL_PATH = os.path.join(BASE_DIR, "model", "model_3.pth")
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "runs", "yolov8_fracture_best.pt")
# 上传目录：使用临时目录以确保可写性
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "fracture_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"上传目录设置为: {UPLOAD_DIR}")

# 检查上传目录是否可写
try:
    test_file = os.path.join(UPLOAD_DIR, "test_write.tmp")
    with open(test_file, "w") as f:
        f.write("test")
    os.remove(test_file)
    logger.info(f"上传目录可写性检查通过: {UPLOAD_DIR}")
except Exception as e:
    logger.error(f"上传目录不可写: {UPLOAD_DIR}, 错误: {e}")
    # 尝试使用当前目录下的上传目录作为备选
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    logger.info(f"切换到备选上传目录: {UPLOAD_DIR}")

# 分类阈值（可根据验证集调整）
CLASSIFICATION_THRESHOLD = 0.7  # 提高阈值，减少误报
YOLO_CONFIDENCE = 0.4          # 提高YOLO置信度阈值，减少误报
TEMPERATURE = 1.5              # 温度缩放参数，>1降低概率，<1提高概率

# 图像预处理
def clahe_enhance(img):
    """CLAHE 对比度增强"""
    img_np = np.array(img, dtype=np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
    img_enhanced = clahe.apply(img_np)
    return Image.fromarray(img_enhanced)

# 定义注意力模块（从原脚本复制）
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

def build_model():
    """构建分类模型（与原脚本一致）"""
    model = None

    # 尝试加载预训练权重，如果失败则创建无预训练权重的模型
    weight_options = [
        ('IMAGENET1K_V1', EfficientNet_B0_Weights.IMAGENET1K_V1),
        ('DEFAULT', EfficientNet_B0_Weights.DEFAULT)
    ]

    for weight_name, weight in weight_options:
        try:
            logger.info(f"尝试加载 EfficientNet-B0 预训练权重: {weight_name}...")
            model = efficientnet_b0(weights=weight)
            logger.info(f"✅ EfficientNet-B0 预训练权重 {weight_name} 加载成功")
            break
        except Exception as e:
            logger.warning(f"预训练权重 {weight_name} 加载失败: {e}")
            continue

    # 如果所有预训练权重都失败，创建无预训练权重的模型
    if model is None:
        logger.info("所有预训练权重加载失败，创建无预训练权重的模型...")
        model = efficientnet_b0(weights=None)
        logger.info("✅ 创建无预训练权重的模型成功")

    # 修改第一层为单通道输入
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

    # 添加注意力模块
    model.features[1] = nn.Sequential(model.features[1], CBAM(16, ratio=4))
    model.features[3] = nn.Sequential(model.features[3], CBAM(40, ratio=8))
    model.features[5] = nn.Sequential(model.features[5], CBAM(112, ratio=16))
    model.features[7] = nn.Sequential(model.features[7], CBAM(320, ratio=16))

    # 修改分类头
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

    # 初始化权重
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    return model

# 图像变换（与原脚本一致）
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.Lambda(clahe_enhance),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

# 简化版变换（CPU环境下使用，跳过CLAHE增强以提高速度）
val_test_transform_simple = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229])
])

tta_transforms = [
    val_test_transform,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(clahe_enhance),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.Grayscale(num_output_channels=1),
        transforms.Lambda(clahe_enhance),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
]

# CPU环境下使用简化版TTA
tta_transforms_simple = [
    val_test_transform_simple,
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])
    ])
]

class FractureDetectionSystem:
    """骨折检测系统"""

    def __init__(self):
        self.device = DEVICE
        self.classification_model = None
        self.yolo_model = None
        self.classification_threshold = CLASSIFICATION_THRESHOLD
        self.yolo_confidence = YOLO_CONFIDENCE
        self.temperature = TEMPERATURE

        # 在CPU环境下使用简化版转换以节省内存和时间
        if str(self.device) == 'cpu':
            logger.info("CPU环境检测到，使用简化版图像变换（跳过CLAHE增强）")
            self.tta_transforms = tta_transforms_simple
            self.val_test_transform = val_test_transform_simple
        else:
            self.tta_transforms = tta_transforms
            self.val_test_transform = val_test_transform

        self.load_models()

    def load_models(self):
        """加载分类和 YOLO 模型"""
        try:
            # 记录初始内存使用
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(f"模型加载前内存: {mem_before:.2f} MB")

            logger.info("加载分类模型...")
            self.classification_model = build_model()

            # 使用更省内存的方式加载模型权重
            try:
                # 尝试使用 weights_only=True 以减少内存使用
                state_dict = torch.load(CLASSIFICATION_MODEL_PATH, map_location=self.device, weights_only=True)
            except RuntimeError:
                # 如果失败，回退到 weights_only=False
                logger.warning("无法使用weights_only=True加载模型，使用weights_only=False")
                state_dict = torch.load(CLASSIFICATION_MODEL_PATH, map_location=self.device, weights_only=False)

            self.classification_model.load_state_dict(state_dict)
            self.classification_model.eval()

            # 立即删除state_dict以释放内存
            del state_dict

            # 强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 记录加载后内存使用
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(f"分类模型加载成功: {CLASSIFICATION_MODEL_PATH}, 内存使用: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB)")
        except Exception as e:
            logger.error(f"加载分类模型失败: {e}")
            raise

        if YOLO is not None:
            try:
                logger.info("加载 YOLO 模型...")

                # 简化：直接使用 YOLO 类加载模型文件
                # 设置简单参数以减少内存使用
                try:
                    self.yolo_model = YOLO(YOLO_MODEL_PATH)
                    logger.info("YOLO 类成功加载模型文件")
                except Exception as yolo_error:
                    logger.warning(f"YOLO 类直接加载失败: {yolo_error}, 尝试备用方法...")

                    # 备用方法：使用基础模型
                    try:
                        # 创建基础模型（使用 yolov8n）
                        base_model = YOLO('yolov8n.pt')

                        # 尝试加载自定义权重
                        try:
                            # 先尝试 weights_only=True 以减少内存
                            model_data = torch.load(YOLO_MODEL_PATH, map_location=self.device, weights_only=True)
                        except RuntimeError:
                            # 回退到 weights_only=False
                            logger.warning("无法使用weights_only=True加载，使用weights_only=False")
                            model_data = torch.load(YOLO_MODEL_PATH, map_location=self.device, weights_only=False)

                        if isinstance(model_data, dict):
                            if 'model' in model_data:
                                model_weights = model_data['model']
                            else:
                                model_weights = model_data
                        else:
                            model_weights = model_data

                        # 加载权重到基础模型
                        if isinstance(model_weights, dict):
                            base_model.model.load_state_dict(model_weights, strict=False)
                            logger.info("使用strict=False加载自定义权重成功")

                        self.yolo_model = base_model
                        logger.info("备用方法加载 YOLO 模型成功")

                        # 清理临时变量
                        del model_data, model_weights

                    except Exception as backup_error:
                        logger.error(f"备用方法也失败: {backup_error}")
                        self.yolo_model = None

                if self.yolo_model is not None:
                    # 强制垃圾回收
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # 记录YOLO加载后内存使用
                    process = psutil.Process()
                    mem_yolo = process.memory_info().rss / (1024 * 1024)  # MB
                    logger.info(f"YOLO 模型加载成功: {YOLO_MODEL_PATH}, 总内存: {mem_yolo:.2f} MB")
                else:
                    logger.warning("YOLO 模型加载失败，将禁用 YOLO 检测功能")

            except Exception as e:
                logger.error(f"加载 YOLO 模型失败: {e}")
                self.yolo_model = None
        else:
            logger.warning("YOLO 模块未安装，跳过 YOLO 模型加载")
            self.yolo_model = None

    def classify_image(self, image_path: str, use_tta: bool = True):
        """对单张图像进行分类"""
        try:
            # 记录分类前内存
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB

            img_pil = Image.open(image_path).convert('L')

            if use_tta and self.tta_transforms:
                tta_probs = []
                with torch.no_grad():
                    for transform in self.tta_transforms:
                        tta_tensor = transform(img_pil).unsqueeze(0).to(self.device)
                        logits = self.classification_model(tta_tensor)
                        tta_probs.append(torch.sigmoid(logits / self.temperature).item())
                prob = np.mean(tta_probs)
            else:
                with torch.no_grad():
                    tensor = self.val_test_transform(img_pil).unsqueeze(0).to(self.device)
                    logits = self.classification_model(tensor)
                    prob = torch.sigmoid(logits / self.temperature).item()

            prediction = 1 if prob >= self.classification_threshold else 0

            # 记录分类后内存
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(f"分类完成，内存使用: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB)")

            # 分类后清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "probability": float(prob),
                "prediction": int(prediction),
                "threshold": float(self.classification_threshold)
            }
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            # 清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    def detect_fracture_yolo(self, image_path: str):
        """使用 YOLO 检测骨折区域"""
        if self.yolo_model is None:
            return []

        try:
            # 记录检测前内存
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB

            # 使用更小的图像尺寸以节省内存 (224x224)
            logger.info(f"开始YOLO检测，图像路径: {image_path}, 置信度阈值: {self.yolo_confidence}, 内存: {mem_before:.2f} MB")

            try:
                results = self.yolo_model(image_path, conf=self.yolo_confidence, iou=0.5, imgsz=320, max_det=10, verbose=False)
            except Exception as inference_error:
                logger.error(f"YOLO推理失败: {inference_error}")
                # 尝试更小的尺寸
                logger.info("尝试更小的图像尺寸: 224x224")
                try:
                    results = self.yolo_model(image_path, conf=self.yolo_confidence, iou=0.5, imgsz=224, max_det=10, verbose=False)
                except Exception as retry_error:
                    logger.error(f"重试也失败: {retry_error}")
                    # 尝试更小的尺寸
                    logger.info("尝试更小的图像尺寸: 160x160")
                    try:
                        results = self.yolo_model(image_path, conf=self.yolo_confidence, iou=0.5, imgsz=160, max_det=10, verbose=False)
                    except Exception as retry2_error:
                        logger.error(f"第三次尝试也失败: {retry2_error}")
                        return []

            detections = []

            # 记录检测后内存
            mem_after = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(f"YOLO检测完成，内存使用: {mem_after:.2f} MB (+{mem_after - mem_before:.2f} MB)")

            if results[0].boxes is not None:
                # 获取原始图像尺寸
                if hasattr(results[0], 'orig_shape'):
                    orig_height, orig_width = results[0].orig_shape
                else:
                    # 如果没有orig_shape，使用PIL读取图像尺寸
                    try:
                        from PIL import Image
                        with Image.open(image_path) as img:
                            orig_width, orig_height = img.size
                    except:
                        # 备用默认尺寸
                        orig_width, orig_height = 1000, 1000

                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = self.yolo_model.names[int(box.cls[0].cpu().numpy())]

                    # 转换为整数
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # 应用过滤规则
                    # 1. 置信度过滤 (提高到0.45)
                    if conf < 0.45:
                        continue

                    # 2. 边缘过滤：如果框的任意一边距离图像边缘小于5%，则过滤
                    edge_threshold = 0.05  # 5%
                    if (x1 < edge_threshold * orig_width or
                        y1 < edge_threshold * orig_height or
                        x2 > (1 - edge_threshold) * orig_width or
                        y2 > (1 - edge_threshold) * orig_height):
                        continue

                    # 3. 面积过滤：面积小于图像总面积的0.5%
                    box_area = (x2 - x1) * (y2 - y1)
                    image_area = orig_width * orig_height
                    if box_area < 0.005 * image_area:  # 0.5%
                        continue

                    # 4. 可选：将框向图像中心调整（假设手腕在中心区域）
                    center_adjustment = 0.2  # 调整幅度20%
                    if center_adjustment > 0:
                        # 计算框中心点
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        # 计算图像中心点
                        img_center_x = orig_width / 2
                        img_center_y = orig_height / 2
                        # 将框中心点向图像中心移动一定比例
                        new_center_x = center_x + (img_center_x - center_x) * center_adjustment
                        new_center_y = center_y + (img_center_y - center_y) * center_adjustment
                        # 计算调整后的框坐标（保持框尺寸不变）
                        width = x2 - x1
                        height = y2 - y1
                        x1 = int(max(0, new_center_x - width / 2))
                        y1 = int(max(0, new_center_y - height / 2))
                        x2 = int(min(orig_width, new_center_x + width / 2))
                        y2 = int(min(orig_height, new_center_y + height / 2))

                    detections.append({
                        "class": cls,
                        "confidence": float(conf),
                        "bbox": {
                            "x1": x1, "y1": y1,
                            "x2": x2, "y2": y2
                        }
                    })
                logger.info(f"YOLO检测完成，原始检测到 {len(results[0].boxes)} 个目标，过滤后剩余 {len(detections)} 个")
            else:
                logger.info("YOLO检测完成，未检测到任何目标")

            # 检测完成后强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return detections
        except Exception as e:
            logger.error(f"YOLO 检测失败: {e}")
            # 异常时也要清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return []

    def analyze_image(self, image_path: str, patient_id: Optional[str] = None):
        """综合分析图像：分类 + YOLO 检测"""
        try:
            # 监控内存使用
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(f"开始综合分析图像: {image_path}, 内存: {mem_before:.2f} MB")

            # 设置内存限制 (2.5GB，留一些空间给系统)
            MEMORY_LIMIT_MB = 2500

            # 检查内存是否接近限制
            if mem_before > MEMORY_LIMIT_MB:
                logger.warning(f"内存使用较高: {mem_before:.2f} MB > {MEMORY_LIMIT_MB} MB, 尝试垃圾回收")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                mem_after_gc = process.memory_info().rss / (1024 * 1024)  # MB
                logger.info(f"垃圾回收后内存: {mem_after_gc:.2f} MB")

            # 分类 - 禁用TTA以加快速度，使用更小的图像尺寸
            classify_start = time.time()
            try:
                classification_result = self.classify_image(image_path, use_tta=False)
                classify_time = time.time() - classify_start
                logger.info(f"分类成功: 原始概率={classification_result['probability']:.3f}, 预测={classification_result['prediction']}, 耗时: {classify_time:.2f}秒")
            except Exception as e:
                logger.error(f"分类失败: {e}")
                classification_result = {
                    "probability": 0.0,
                    "prediction": 0,
                    "threshold": self.classification_threshold
                }

            # YOLO 检测 - 增加异常处理和超时
            yolo_detections = []
            yolo_time = 0

            # 如果分类结果很低概率，跳过YOLO检测以节省时间和内存
            if classification_result["probability"] < 0.2:  # 如果概率低于20%，跳过检测
                logger.info(f"分类概率较低 ({classification_result['probability']:.3f} < 0.2)，跳过YOLO检测")
            else:
                try:
                    if self.yolo_model is not None:
                        # 检查内存是否足够
                        mem_before_yolo = process.memory_info().rss / (1024 * 1024)  # MB
                        if mem_before_yolo > MEMORY_LIMIT_MB * 0.7:  # 如果内存使用超过70%
                            logger.warning(f"内存使用较高 ({mem_before_yolo:.2f} MB)，跳过YOLO检测以节省内存")
                        else:
                            # 设置YOLO检测超时（20秒）
                            yolo_start = time.time()
                            try:
                                # 使用线程池执行YOLO检测，设置超时
                                import concurrent.futures
                                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                                    yolo_future = executor.submit(self.detect_fracture_yolo, image_path)
                                    yolo_detections = yolo_future.result(timeout=20.0)
                            except concurrent.futures.TimeoutError:
                                logger.warning("YOLO检测超时（20秒），返回空检测结果")
                                yolo_detections = []
                            except Exception as yolo_error:
                                logger.error(f"YOLO检测失败: {yolo_error}")
                                yolo_detections = []

                            yolo_time = time.time() - yolo_start
                            if yolo_detections:
                                logger.info(f"YOLO检测完成: 检测数={len(yolo_detections)}, 耗时: {yolo_time:.2f}秒")
                            else:
                                logger.info(f"YOLO检测完成: 未检测到目标, 耗时: {yolo_time:.2f}秒")
                    else:
                        logger.warning("YOLO模型未加载，跳过检测")
                except Exception as e:
                    logger.error(f"YOLO检测失败: {e}")
                    yolo_detections = []  # 确保返回空列表而不是None

            has_fracture_bbox = len(yolo_detections) > 0

            # 联合判断（分类概率 >= 阈值 且 YOLO 检测到骨折区域）
            final_prediction = 1 if (
                classification_result["prediction"] == 1 and
                has_fracture_bbox
            ) else 0

            # 调整分类概率用于前端显示（减少过度自信）
            adjusted_classification = classification_result.copy()
            original_prob = adjusted_classification["probability"]
            # 将极端概率向0.5压缩
            calibrated_prob = (original_prob - 0.5) * 0.7 + 0.5
            calibrated_prob = max(0.0, min(1.0, calibrated_prob))
            adjusted_classification["probability"] = float(calibrated_prob)
            adjusted_classification["original_probability"] = float(original_prob)  # 保留原始值用于调试

            # 生成报告
            report = self.generate_report(
                classification_result,  # 报告仍然使用原始概率进行医学判断
                yolo_detections,
                final_prediction,
                patient_id
            )

            # 记录最终内存使用
            process = psutil.Process()
            mem_final = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(f"综合分析完成: 最终预测={final_prediction}, 校准后概率={calibrated_prob:.3f}, 最终内存: {mem_final:.2f} MB")

            # 分析完成后强制垃圾回收
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {
                "classification": adjusted_classification,  # 使用校准后的概率
                "detections": yolo_detections,
                "final_prediction": final_prediction,
                "report": report,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"图像分析失败: {e}", exc_info=True)
            # 异常时清理内存
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # 返回一个安全的默认结果，而不是让进程崩溃
            return {
                "classification": {
                    "probability": 0.0,
                    "prediction": 0,
                    "threshold": self.classification_threshold
                },
                "detections": [],
                "final_prediction": 0,
                "report": {
                    "patient_id": patient_id,
                    "fracture_detected": False,
                    "confidence_score": 0.0,
                    "detection_count": 0,
                    "recommendation": "分析过程中出现错误，请重试",
                    "detailed_findings": []
                },
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }

    def generate_report(self, classification_result, detections, final_prediction, patient_id):
        """生成分析报告"""
        fracture_prob = classification_result["probability"]
        has_fracture = final_prediction == 1

        # 置信度校准：将极端概率向0.5压缩，减少过度自信
        calibrated_prob = (fracture_prob - 0.5) * 0.7 + 0.5
        calibrated_prob = max(0.0, min(1.0, calibrated_prob))  # 确保在[0,1]范围内

        report = {
            "patient_id": patient_id,
            "fracture_detected": bool(has_fracture),
            "confidence_score": float(calibrated_prob),
            "detection_count": len(detections),
            "recommendation": "",
            "detailed_findings": []
        }

        if has_fracture:
            report["recommendation"] = "发现疑似骨折病灶，建议进一步临床检查确认。"
            for i, det in enumerate(detections):
                report["detailed_findings"].append(
                    f"病灶 {i+1}: 置信度 {det['confidence']:.2%}, 位置: [{det['bbox']['x1']}, {det['bbox']['y1']}] - [{det['bbox']['x2']}, {det['bbox']['y2']}]"
                )
        else:
            if fracture_prob > 0.3:
                report["recommendation"] = "未发现明确骨折病灶，但存在可疑区域，建议定期复查。"
            else:
                report["recommendation"] = "未发现骨折病灶，影像表现正常。"

        return report

# 初始化检测系统
detection_system = FractureDetectionSystem()

# 创建 FastAPI 应用
app = FastAPI(
    title="儿童手腕隐匿性骨折检测系统",
    description="基于深度学习的骨折检测后端 API",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载上传目录为静态文件（可选）
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# 提供前端页面
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """返回前端HTML页面"""
    try:
        # 前端HTML文件路径
        frontend_path = os.path.join(os.path.dirname(__file__), "fracture_frontend.html")
        with open(frontend_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content, status_code=200)
    except Exception as e:
        logger.error(f"加载前端页面失败: {e}")
        # 如果前端加载失败，返回API信息
        return {
            "service": "儿童手腕隐匿性骨折检测系统",
            "version": "1.0.0",
            "status": "running",
            "models_loaded": {
                "classification": detection_system.classification_model is not None,
                "yolo": detection_system.yolo_model is not None
            },
            "error": "前端页面加载失败"
        }

# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点，用于云平台监控"""
    return {
        "status": "healthy",
        "service": "儿童手腕隐匿性骨折检测系统",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "classification": detection_system.classification_model is not None,
            "yolo": detection_system.yolo_model is not None
        }
    }

# 模拟数据存储（生产环境应使用数据库）
mock_patients = [
    {
        "id": "P001",
        "name": "张三",
        "age": 8,
        "gender": "男",
        "last_visit": "2024-03-15",
        "result": "positive",
        "risk_level": "high"
    },
    {
        "id": "P002",
        "name": "李四",
        "age": 6,
        "gender": "女",
        "last_visit": "2024-03-14",
        "result": "negative",
        "risk_level": "low"
    },
    {
        "id": "P003",
        "name": "王五",
        "age": 10,
        "gender": "男",
        "last_visit": "2024-03-13",
        "result": "positive",
        "risk_level": "medium"
    },
    {
        "id": "P004",
        "name": "赵六",
        "age": 7,
        "gender": "女",
        "last_visit": "2024-03-12",
        "result": "negative",
        "risk_level": "low"
    }
]

mock_analysis_history = [
    {
        "id": "A001",
        "patient_id": "P001",
        "patient_name": "张三",
        "date": "2024-03-15 10:30",
        "result": "positive",
        "confidence": 0.89,
        "image_url": "/uploads/sample1.jpg"
    },
    {
        "id": "A002",
        "patient_id": "P002",
        "patient_name": "李四",
        "date": "2024-03-14 14:20",
        "result": "negative",
        "confidence": 0.12,
        "image_url": "/uploads/sample2.jpg"
    },
    {
        "id": "A003",
        "patient_id": "P003",
        "patient_name": "王五",
        "date": "2024-03-13 09:15",
        "result": "positive",
        "confidence": 0.76,
        "image_url": "/uploads/sample3.jpg"
    }
]

chat_messages = []


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    use_yolo: bool = True
):
    """
    分析上传的X光图像
    - file: 图像文件（支持 JPG, PNG 等）
    - patient_id: 患者ID（可选）
    - use_yolo: 是否使用 YOLO 进行病灶检测
    """
    file_path = None
    try:
        # 验证文件类型
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型。仅支持: {', '.join(allowed_extensions)}"
            )

        # 检查文件大小（限制为10MB）
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        file_size = 0
        # 获取文件大小
        file.file.seek(0, 2)  # 移动到文件末尾
        file_size = file.file.tell()
        file.file.seek(0)  # 重置到文件开头

        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"文件过大（{file_size / (1024*1024):.1f}MB）。最大支持10MB。"
            )

        # 检查是否为PNG格式（用户上传的是PNG）
        if file_ext == '.png':
            logger.info(f"PNG图像上传: {file.filename}, 大小: {file_size / 1024:.1f}KB")

        # 保存上传的文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(UPLOAD_DIR, filename)

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"文件已保存: {file_path}")
        except Exception as e:
            logger.error(f"保存文件失败: {e}")
            raise HTTPException(status_code=500, detail="文件保存失败")

        try:
            # 分析图像（使用线程池执行，避免阻塞事件循环）
            ANALYSIS_TIMEOUT = 55.0  # 55秒超时，略小于前端60秒
            start_time = time.time()
            logger.info(f"开始分析图像: {filename}")

            try:
                # 将同步分析函数放到线程池中执行，并设置超时
                analysis_result = await asyncio.wait_for(
                    asyncio.to_thread(detection_system.analyze_image, file_path, patient_id),
                    timeout=ANALYSIS_TIMEOUT
                )
            except asyncio.TimeoutError:
                logger.error(f"分析超时: {ANALYSIS_TIMEOUT}秒")
                raise HTTPException(status_code=504, detail=f"分析超时（{ANALYSIS_TIMEOUT}秒），请尝试使用更小的图像或稍后重试")
            except Exception as e:
                logger.error(f"分析过程出错: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

            end_time = time.time()
            elapsed_time = end_time - start_time
            logger.info(f"图像分析完成: {filename}, 耗时: {elapsed_time:.2f}秒")

            # 生成可访问的图片 URL，对文件名进行URL编码
            import urllib.parse
            encoded_filename = urllib.parse.quote(filename)
            image_url = f"/uploads/{encoded_filename}"

            # 记录分析结果详情，用于调试
            logger.info(f"分析结果详情 - 分类概率: {analysis_result['classification']['probability']:.3f}, "
                       f"分类预测: {analysis_result['classification']['prediction']}, "
                       f"YOLO检测数: {len(analysis_result['detections'])}, "
                       f"最终预测: {analysis_result['final_prediction']}, "
                       f"图像URL: {image_url}")

            # 返回结果
            return JSONResponse({
                "success": True,
                "message": "分析完成",
                "data": {
                    **analysis_result,
                    "image_url": image_url
                }
            })
        except HTTPException:
            # 重新抛出HTTP异常
            raise
        except Exception as e:
            logger.error(f"分析过程出错: {e}", exc_info=True)
            # 清理上传的文件
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"清理文件: {file_path}")
            raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

    except HTTPException:
        # 重新抛出HTTP异常
        raise
    except Exception as e:
        logger.error(f"未预期的错误: {e}", exc_info=True)
        # 清理上传的文件
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"清理文件: {file_path}")
        raise HTTPException(status_code=500, detail="服务器内部错误")

@app.get("/api/analysis/{filename}")
async def get_analysis_result(filename: str):
    """获取之前的分析结果（示例）"""
    # 这里可以添加结果缓存或数据库查询
    return {"message": "此功能待实现", "filename": filename}

# 患者管理端点
@app.get("/api/patients")
async def get_patients(
    search: Optional[str] = None,
    result_filter: Optional[str] = None,
    risk_filter: Optional[str] = None
):
    """获取患者列表（支持搜索和过滤）"""
    patients = mock_patients

    # 搜索过滤
    if search:
        search = search.lower()
        patients = [p for p in patients if
                   search in p["name"].lower() or
                   search in p["id"].lower()]

    # 结果过滤
    if result_filter and result_filter != "all":
        patients = [p for p in patients if p["result"] == result_filter]

    # 风险等级过滤
    if risk_filter and risk_filter != "all":
        patients = [p for p in patients if p["risk_level"] == risk_filter]

    return {
        "success": True,
        "count": len(patients),
        "patients": patients
    }

@app.get("/api/patients/{patient_id}")
async def get_patient_detail(patient_id: str):
    """获取患者详情"""
    for patient in mock_patients:
        if patient["id"] == patient_id:
            # 模拟检查记录
            examinations = [
                {
                    "date": "2024-03-15",
                    "type": "X光检查",
                    "result": patient["result"],
                    "doctor": "张医生",
                    "notes": "手腕正侧位片"
                }
            ]
            return {
                "success": True,
                "patient": {**patient, "examinations": examinations}
            }

    raise HTTPException(status_code=404, detail="患者不存在")

@app.post("/api/patients")
async def create_patient(
    name: str,
    age: int,
    gender: str,
    notes: Optional[str] = None
):
    """创建新患者"""
    new_id = f"P{len(mock_patients) + 1:03d}"
    new_patient = {
        "id": new_id,
        "name": name,
        "age": age,
        "gender": gender,
        "last_visit": datetime.now().strftime("%Y-%m-%d"),
        "result": "pending",
        "risk_level": "unknown",
        "notes": notes or ""
    }
    mock_patients.append(new_patient)
    return {
        "success": True,
        "message": "患者创建成功",
        "patient": new_patient
    }

# 分析历史端点
@app.get("/api/analyses")
async def get_analysis_history(
    patient_id: Optional[str] = None,
    limit: int = 10,
    offset: int = 0
):
    """获取分析历史"""
    history = mock_analysis_history

    if patient_id:
        history = [h for h in history if h["patient_id"] == patient_id]

    # 分页
    paginated_history = history[offset:offset + limit]

    return {
        "success": True,
        "total": len(history),
        "count": len(paginated_history),
        "analyses": paginated_history
    }

# 对话端点
@app.get("/api/chat/messages")
async def get_chat_messages(limit: int = 20):
    """获取对话消息"""
    return {
        "success": True,
        "messages": chat_messages[-limit:] if chat_messages else []
    }

@app.post("/api/chat/send")
async def send_chat_message(
    message: str,
    role: str = "user"  # user 或 assistant
):
    """发送聊天消息"""
    if role not in ["user", "assistant"]:
        raise HTTPException(status_code=400, detail="角色必须是 user 或 assistant")

    new_message = {
        "id": f"MSG{len(chat_messages) + 1:06d}",
        "role": role,
        "content": message,
        "timestamp": datetime.now().isoformat()
    }
    chat_messages.append(new_message)

    # 简单的自动回复逻辑
    if role == "user":
        auto_reply = {
            "id": f"MSG{len(chat_messages) + 1:06d}",
            "role": "assistant",
            "content": "已收到您的消息，关于骨折诊断的问题我可以帮您解答。请上传X光图像进行详细分析。",
            "timestamp": datetime.now().isoformat()
        }
        chat_messages.append(auto_reply)

    return {
        "success": True,
        "message": new_message,
        "auto_reply": auto_reply if role == "user" else None
    }

# 统计端点
@app.get("/api/statistics")
async def get_statistics():
    """获取系统统计信息"""
    total_patients = len(mock_patients)
    positive_cases = sum(1 for p in mock_patients if p["result"] == "positive")
    negative_cases = sum(1 for p in mock_patients if p["result"] == "negative")

    return {
        "success": True,
        "statistics": {
            "total_patients": total_patients,
            "positive_cases": positive_cases,
            "negative_cases": negative_cases,
            "positive_rate": positive_cases / total_patients if total_patients > 0 else 0,
            "today_analyses": 24,
            "today_completed": 24,
            "avg_confidence": 0.78,
            "avg_processing_time": "2.5秒"
        }
    }

@app.get("/api/models/status")
async def get_model_status():
    """获取模型状态"""
    return {
        "classification_model_loaded": detection_system.classification_model is not None,
        "yolo_model_loaded": detection_system.yolo_model is not None,
        "device": str(DEVICE),
        "classification_threshold": detection_system.classification_threshold,
        "yolo_confidence": detection_system.yolo_confidence
    }

@app.post("/api/models/threshold")
async def update_threshold(
    classification_threshold: Optional[float] = None,
    yolo_confidence: Optional[float] = None
):
    """更新模型阈值"""
    if classification_threshold is not None:
        if 0 <= classification_threshold <= 1:
            detection_system.classification_threshold = classification_threshold
        else:
            raise HTTPException(status_code=400, detail="分类阈值必须在 0 到 1 之间")

    if yolo_confidence is not None:
        if 0 <= yolo_confidence <= 1:
            detection_system.yolo_confidence = yolo_confidence
        else:
            raise HTTPException(status_code=400, detail="YOLO 置信度必须在 0 到 1 之间")

    return {
        "success": True,
        "message": "阈值已更新",
        "new_thresholds": {
            "classification": detection_system.classification_threshold,
            "yolo_confidence": detection_system.yolo_confidence
        }
    }

if __name__ == "__main__":
    print("=" * 60)
    print("儿童手腕隐匿性骨折检测系统后端")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"分类模型: {CLASSIFICATION_MODEL_PATH}")
    print(f"YOLO模型: {YOLO_MODEL_PATH}")
    print(f"上传目录: {UPLOAD_DIR}")
    print("-" * 60)

    # 检查是否在交互式环境中运行（如Jupyter、IPython）
    is_interactive = False

    # 首先检查是否有运行中的事件循环（更可靠的方法）
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            is_interactive = True
            print("⚠️  检测到运行中的事件循环")
    except (RuntimeError, ImportError):
        pass

    # 然后检查IPython环境（可能无法导入IPython模块）
    if not is_interactive:
        # 检查IPython环境
        if 'IPython' in sys.modules:
            try:
                from IPython import get_ipython
                if get_ipython() is not None:
                    is_interactive = True
                    print("⚠️  检测到IPython/Jupyter环境")
            except ImportError:
                # IPython未安装，忽略这个检查
                pass

    if is_interactive:
        print("\n❌ 错误：检测到交互式环境或已有事件循环运行。")
        print("   无法在此环境中启动FastAPI服务器。")
        print("\n✅ 请使用以下方法启动后端：")
        print("   1. 在命令行中运行: python backend.py")
        print("   2. 或双击运行: run.bat")
        print("   3. 或使用终端/cmd/PowerShell执行")
        print("\n⚠️  不要在Jupyter notebook或IPython中运行此脚本。")
        print("=" * 60)
        sys.exit(1)

    # 从环境变量获取端口（Render等云平台会提供）
    port = int(os.environ.get("PORT", "8000"))
    print("正在启动服务器...")
    print("\n请访问以下地址:")
    print(f"1. http://localhost:{port}")
    print(f"2. http://127.0.0.1:{port}")
    print("\nAPI文档:")
    print(f"- Swagger UI: http://localhost:{port}/docs")
    print(f"- ReDoc文档: http://localhost:{port}/redoc")
    print("=" * 60)

    logger.info("启动骨折检测系统后端服务...")
    try:
        uvicorn.run(
            app,
            host="0.0.0.0",  # 绑定到所有接口
            port=port,
            log_level="info"
        )
    except RuntimeError as e:
        if "asyncio.run() cannot be called from a running event loop" in str(e):
            print("\n❌ 错误：检测到运行中的事件循环。")
            print("   您可能正在Jupyter/IPython或已有异步环境运行。")
            print("\n✅ 请使用以下方法启动后端：")
            print("   1. 在命令行中运行: python backend.py")
            print("   2. 或双击运行: run.bat")
            print("   3. 或使用终端/cmd/PowerShell执行")
            print("\n⚠️  不要在Jupyter notebook或IPython中运行此脚本。")
            sys.exit(1)
        else:
            raise
