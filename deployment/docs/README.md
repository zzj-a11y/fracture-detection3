# 儿童手腕隐匿性骨折检测系统

基于深度学习的手腕X光图像骨折检测系统，包含前端界面和后端API。

## 文件说明

- `backend.py` - 后端FastAPI服务器，提供模型推理API
- `完整功能版_6.html` - 前端界面（HTML/CSS/JavaScript）
- `CNN_隐匿性骨折案例（优化版）.py` - 原始模型训练和推理脚本
- `model/model_3.pth` - 训练好的分类模型权重
- `runs/yolov8_fracture_best.pt` - YOLOv8骨折检测专用模型（用于骨折区域检测）
- `test_load.py` - 模型加载测试脚本
- `test_backend.py` - 后端API测试脚本

## 系统要求

- Python 3.8+
- PyTorch (GPU版本建议)
- CUDA (可选，用于GPU加速)

## 安装依赖

```bash
pip install torch torchvision ultralytics fastapi uvicorn python-multipart requests opencv-python pillow numpy matplotlib
```

如果已安装部分依赖，可跳过。

## 启动后端服务器

### 方式一：直接运行
```bash
python backend.py
```
服务器将在 `http://localhost:8000` 启动。

### 方式二：使用uvicorn
```bash
uvicorn backend:app --host 0.0.0.0 --port 8000 --reload
```

## 启动前端

1. 打开 `完整功能版_6.html` 在浏览器中
2. 或使用Python启动静态文件服务器：
```bash
python -m http.server 8080
```
然后访问 `http://localhost:8080/完整功能版_6.html`

## API接口

### 核心接口
- `POST /api/analyze` - 上传X光图像进行分析
  - 参数：`file` (图像文件), `patient_id` (可选), `use_yolo` (可选)
  - 返回：分类结果、YOLO检测框、综合报告

### 患者管理
- `GET /api/patients` - 获取患者列表
- `GET /api/patients/{id}` - 获取患者详情
- `POST /api/patients` - 创建新患者

### 分析历史
- `GET /api/analyses` - 获取分析历史记录

### 对话系统
- `GET /api/chat/messages` - 获取聊天记录
- `POST /api/chat/send` - 发送消息

### 系统状态
- `GET /` - 服务状态
- `GET /api/models/status` - 模型加载状态
- `POST /api/models/threshold` - 更新模型阈值

## 配置说明

可在 `backend.py` 中修改以下配置：

```python
# 模型路径
CLASSIFICATION_MODEL_PATH = "./model/model_3.pth"
YOLO_MODEL_PATH = "./runs/yolov8_fracture_best.pt"

# 上传目录
UPLOAD_DIR = "./exit/uploads"

# 阈值配置
CLASSIFICATION_THRESHOLD = 0.5  # 分类阈值
YOLO_CONFIDENCE = 0.3          # YOLO检测置信度
```

## 使用示例

### 1. 使用curl测试图像分析
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "file=@/path/to/xray.jpg" \
  -F "patient_id=P001"
```

### 2. 获取患者列表
```bash
curl "http://localhost:8000/api/patients"
```

### 3. 更新模型阈值
```bash
curl -X POST "http://localhost:8000/api/models/threshold" \
  -H "Content-Type: application/json" \
  -d '{"classification_threshold": 0.6}'
```

## 前端与后端对接

前端需要调用后端API，主要对接点：

1. **图像分析页面**：上传图像到 `/api/analyze`
2. **患者档案管理**：调用 `/api/patients` 系列接口
3. **数据统计页面**：调用 `/api/statistics`
4. **对话页面**：调用 `/api/chat/` 系列接口

前端JavaScript示例：
```javascript
// 上传图像分析
async function analyzeImage(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('patient_id', 'P001');
    
    const response = await fetch('http://localhost:8000/api/analyze', {
        method: 'POST',
        body: formData
    });
    return await response.json();
}
```

## 故障排除

### 模型加载失败
1. 检查模型文件路径是否正确
2. 确保PyTorch版本兼容
3. 检查CUDA是否可用（GPU推理）

### 文件上传失败
1. 确保 `python-multipart` 已安装
2. 检查上传目录权限
3. 验证文件格式（支持 JPG、PNG、BMP、TIFF）

### CORS错误
1. 前端与后端端口不同时，确保CORS已启用
2. 后端已配置允许所有来源 (`allow_origins=["*"]`)

### 内存不足
1. 减少批量大小
2. 使用CPU推理
3. 关闭不需要的服务

## 性能优化建议

1. **GPU加速**：确保CUDA可用，模型自动使用GPU
2. **批处理**：支持批量图像分析（需扩展API）
3. **缓存**：对相同图像结果进行缓存
4. **异步处理**：长时间任务可改为异步队列

## 免责声明

本系统为辅助诊断工具，不能替代专业医生诊断。所有检测结果仅供参考，临床诊断请以医生判断为准。

## 许可证

仅供学习和研究使用。