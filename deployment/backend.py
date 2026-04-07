"""
儿童手腕隐匿性骨折检测系统后端
基于 FastAPI 实现模型推理 API
"""

import os
import sys
import logging
from pathlib import Path
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
UPLOAD_DIR = os.path.join(BASE_DIR, "exit", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# 分类阈值（可根据验证集调整）
CLASSIFICATION_THRESHOLD = 0.7  # 提高阈值，减少误报
YOLO_CONFIDENCE = 0.7          # 进一步提高YOLO置信度阈值，减少误检
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
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    model = efficientnet_b0(weights=weights)

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

class FractureDetectionSystem:
    """骨折检测系统"""

    def __init__(self):
        self.device = DEVICE
        self.classification_model = None
        self.yolo_model = None
        self.classification_threshold = CLASSIFICATION_THRESHOLD
        self.yolo_confidence = YOLO_CONFIDENCE
        self.temperature = TEMPERATURE
        self.tta_transforms = tta_transforms
        self.load_models()

    def load_models(self):
        """加载分类和 YOLO 模型"""
        try:
            logger.info("加载分类模型...")
            self.classification_model = build_model()
            self.classification_model.load_state_dict(
                torch.load(CLASSIFICATION_MODEL_PATH, map_location=self.device, weights_only=False)
            )
            self.classification_model.eval()
            logger.info(f"分类模型加载成功: {CLASSIFICATION_MODEL_PATH}")
        except Exception as e:
            logger.error(f"加载分类模型失败: {e}")
            raise

        if YOLO is not None:
            try:
                logger.info("加载 YOLO 模型...")

                # 简单直接的方法：使用 torch.load 加载模型文件，设置 weights_only=False
                # 这样可以绕过 PyTorch 2.6+ 的安全限制
                logger.info(f"使用 torch.load(weights_only=False) 加载: {YOLO_MODEL_PATH}")

                # 直接使用 torch.load 加载模型文件
                # 根据错误信息，我们需要信任这个模型文件（来自用户自己的训练）
                model_data = torch.load(YOLO_MODEL_PATH, map_location=self.device, weights_only=False)
                logger.info(f"torch.load 成功，加载的数据类型: {type(model_data)}")

                # 根据加载的数据类型处理
                if isinstance(model_data, dict):
                    # 如果是状态字典，创建 YOLO 模型并加载状态
                    logger.info("加载的是状态字典，创建基础 YOLO 模型并加载状态...")

                    # 创建基础 YOLO 模型（使用 yolov8n 作为基础架构）
                    base_model = YOLO('yolov8n.pt')

                    # 加载状态字典到模型
                    base_model.model.load_state_dict(model_data)

                    self.yolo_model = base_model
                    logger.info("YOLO 模型通过状态字典加载成功")
                else:
                    # 如果是完整的模型对象，直接使用
                    logger.info("加载的是完整模型对象，直接使用...")
                    self.yolo_model = model_data
                    logger.info("YOLO 模型作为完整对象加载成功")

                logger.info(f"YOLO 模型加载成功: {YOLO_MODEL_PATH}")
            except Exception as e:
                logger.error(f"加载 YOLO 模型失败: {e}")
                self.yolo_model = None
        else:
            logger.warning("YOLO 模块未安装，跳过 YOLO 模型加载")
            self.yolo_model = None

    def classify_image(self, image_path: str, use_tta: bool = True):
        """对单张图像进行分类"""
        try:
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
                    tensor = val_test_transform(img_pil).unsqueeze(0).to(self.device)
                    logits = self.classification_model(tensor)
                    prob = torch.sigmoid(logits / self.temperature).item()

            prediction = 1 if prob >= self.classification_threshold else 0
            return {
                "probability": float(prob),
                "prediction": int(prediction),
                "threshold": float(self.classification_threshold)
            }
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            raise

    def detect_fracture_yolo(self, image_path: str):
        """使用 YOLO 检测骨折区域"""
        if self.yolo_model is None:
            return []

        try:
            results = self.yolo_model(image_path, conf=self.yolo_confidence, iou=0.4)
            detections = []

            if results[0].boxes is not None:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = self.yolo_model.names[int(box.cls[0].cpu().numpy())]
                    detections.append({
                        "class": cls,
                        "confidence": float(conf),
                        "bbox": {
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2)
                        }
                    })
            return detections
        except Exception as e:
            logger.error(f"YOLO 检测失败: {e}")
            return []

    def analyze_image(self, image_path: str, patient_id: Optional[str] = None):
        """综合分析图像：分类 + YOLO 检测"""
        try:
            # 分类
            classification_result = self.classify_image(image_path, use_tta=True)
            logger.info(f"分类结果: 原始概率={classification_result['probability']:.3f}, 预测={classification_result['prediction']}, 阈值={classification_result['threshold']}")

            # YOLO 检测
            yolo_detections = self.detect_fracture_yolo(image_path)
            has_fracture_bbox = len(yolo_detections) > 0
            if yolo_detections:
                max_conf = max([d['confidence'] for d in yolo_detections])
                logger.info(f"YOLO检测: 检测数={len(yolo_detections)}, 最高置信度={max_conf:.3f}, 阈值={self.yolo_confidence}")
            else:
                logger.info(f"YOLO检测: 无检测结果, 阈值={self.yolo_confidence}")

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

            logger.info(f"最终预测: {final_prediction}, 校准后概率={adjusted_classification['probability']:.3f}, 原始概率={original_prob:.3f}")

            return {
                "classification": adjusted_classification,  # 使用校准后的概率
                "detections": yolo_detections,
                "final_prediction": final_prediction,
                "report": report,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            raise

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
    # 验证文件类型
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。仅支持: {', '.join(allowed_extensions)}"
        )

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
        # 分析图像
        analysis_result = detection_system.analyze_image(file_path, patient_id)

        # 生成可访问的图片 URL
        image_url = f"/uploads/{filename}"

        # 返回结果
        return JSONResponse({
            "success": True,
            "message": "分析完成",
            "data": {
                **analysis_result,
                "image_url": image_url
            }
        })
    except Exception as e:
        logger.error(f"分析过程出错: {e}")
        # 清理上传的文件
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"分析失败: {str(e)}")

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
