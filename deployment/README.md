# 儿童手腕隐匿性骨折检测系统

基于深度学习的儿童手腕隐匿性骨折检测系统，结合分类模型和目标检测模型，提供准确的骨折诊断辅助。

## 功能特点

- 🩻 **X光图像分析**：自动检测手腕X光图像中的骨折病灶
- 🎯 **双模型协同**：EfficientNet分类 + YOLOv8目标检测
- 📊 **可视化结果**：病灶定位、置信度评分、详细报告
- 🏥 **患者管理**：患者信息记录、检查历史追踪
- 💬 **智能对话**：简单的问答交互功能
- 🌐 **Web界面**：友好的中文用户界面

## 快速开始

### 本地运行
1. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
2. 启动应用：
   ```bash
   python backend.py
   ```
3. 访问 http://localhost:8000

### 云部署
支持免费部署到 Render.com 或 Hugging Face Spaces，详见 [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)。

## 项目结构

```
deployment/
├── backend.py              # FastAPI 后端应用
├── fracture_frontend.html  # 前端HTML界面
├── requirements.txt        # Python依赖
├── runtime.txt            # Python版本配置
├── Procfile              # 云平台启动配置
├── model/
│   └── model_3.pth      # 分类模型
├── runs/
│   └── yolov8_fracture_best.pt  # YOLO检测模型
├── exit/
│   └── uploads/         # 上传文件目录
└── docs/               # 文档
```

## 技术栈

- **后端**: FastAPI + Python 3.9
- **前端**: HTML5 + CSS3 + JavaScript
- **深度学习**: PyTorch + EfficientNet + YOLOv8
- **图像处理**: OpenCV + Pillow
- **部署**: Render.com / Hugging Face Spaces

## 模型说明

### 1. 分类模型 (EfficientNet-B0 + CBAM)
- 输入：224×224 灰度图像
- 输出：骨折概率 (0-1)
- 特点：注意力机制增强特征提取

### 2. 目标检测模型 (YOLOv8-nano)
- 检测骨折病灶位置
- 实时边界框标注
- 轻量级，适合部署

## API 接口

| 端点 | 方法 | 描述 |
|------|------|------|
| `/` | GET | 前端页面 |
| `/api/analyze` | POST | 分析X光图像 |
| `/api/patients` | GET | 患者列表 |
| `/api/statistics` | GET | 系统统计 |
| `/docs` | GET | API文档 |

## 使用示例

1. 上传手腕X光图像
2. 系统自动分析并显示结果
3. 查看骨折概率、病灶位置
4. 获取详细的诊断报告

## 性能指标

- 召回率：>90%
- 精确率：>65%
- 推理时间：2-3秒（CPU）
- 模型大小：25MB（合计）

## 注意事项

1. 本系统为辅助诊断工具，不能替代专业医生诊断
2. 建议使用清晰的手腕正位X光图像
3. 模型针对儿童手腕骨折优化

## 许可证

本项目仅供学习和研究使用。

## 联系

如有问题或建议，请查看部署指南或检查日志文件。