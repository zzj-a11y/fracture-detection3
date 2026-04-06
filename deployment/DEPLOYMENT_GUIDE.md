# 儿童手腕隐匿性骨折检测系统 - 部署指南

## 部署方案选择

本项目提供两种免费云部署方案：

1. **Render.com**（推荐）- 最简单，支持 Python + 深度学习模型
2. **Hugging Face Spaces** - 需要 Docker 配置，但完全免费

## 方案一：Render.com 部署（最简单）

Render 提供免费的 Web 服务，非常适合学生作业。

### 准备工作
1. 注册 [Render.com](https://render.com) 账号（支持 GitHub 登录）
2. 将本项目上传到 GitHub 仓库

### 部署步骤

#### 1. 准备部署文件
确保你的 `deployment` 目录包含以下文件：
```
deployment/
├── backend.py              # 后端 FastAPI 应用
├── fracture_frontend.html  # 前端界面
├── requirements.txt        # Python 依赖
├── runtime.txt            # Python 版本
├── Procfile              # 启动命令
├── .gitignore           # Git 忽略文件
├── model/
│   └── model_3.pth      # 分类模型文件
├── runs/
│   └── yolov8_fracture_best.pt  # YOLO 模型文件
└── exit/
    └── uploads/         # 上传目录
```

#### 2. 上传到 GitHub
```bash
# 在 deployment 目录中执行
git init
git add .
git commit -m "初始化骨折检测系统"
git remote add origin https://github.com/你的用户名/你的仓库名.git
git push -u origin main
```

#### 3. 在 Render 上部署
1. 登录 Render.com，点击 "New +" → "Web Service"
2. 连接你的 GitHub 仓库
3. 选择仓库，配置如下：
   - **Name**: `fracture-detection`（自定义）
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python backend.py`（或使用 Procfile）
4. 点击 "Create Web Service"
5. 等待部署完成（约5-10分钟）

#### 4. 访问应用
部署完成后，Render 会提供一个类似 `https://fracture-detection.onrender.com` 的 URL。

## 方案二：Hugging Face Spaces 部署

### 准备工作
1. 注册 [Hugging Face](https://huggingface.co) 账号
2. 创建新的 Space，选择 "Docker" 类型

### 创建 Dockerfile
在 `deployment` 目录中创建 `Dockerfile`：

```dockerfile
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 7860

# 启动应用
CMD ["python", "backend.py"]
```

### 部署步骤
1. 将整个项目上传到 Hugging Face Space
2. 修改 `backend.py` 中的端口为 `7860`（Hugging Face 默认端口）
3. 在 Space 设置中启用 "Public" 访问权限

## 本地测试

### 1. 安装依赖
```bash
cd deployment
pip install -r requirements.txt
```

### 2. 运行应用
```bash
python backend.py
```

### 3. 访问应用
打开浏览器访问：http://localhost:8000

## 文件说明

### 必需文件
- `backend.py` - FastAPI 后端，提供 API 和前端页面
- `fracture_frontend.html` - 前端界面（已集成到后端）
- `requirements.txt` - Python 依赖包
- `runtime.txt` - Python 版本（3.9.18）
- `Procfile` - 云平台启动命令
- `model/model_3.pth` - 分类模型（19MB）
- `runs/yolov8_fracture_best.pt` - YOLO 检测模型（6MB）

### 可选文件
- `.gitignore` - Git 忽略规则
- `DEPLOYMENT_GUIDE.md` - 本部署指南

## 常见问题

### Q1: 部署时内存不足？
- Render 免费版提供 512MB RAM，可能不够
- 解决方案：使用更小的模型或升级到付费计划

### Q2: 模型下载失败？
- 确保模型文件在 GitHub 仓库中
- 检查文件路径是否正确

### Q3: 应用启动后无法访问？
- 检查 Render 的构建日志
- 确保端口绑定正确（Render 使用 $PORT 环境变量）

### Q4: 如何更新应用？
- 推送代码到 GitHub，Render 会自动重新部署

### Q5: 免费服务有使用限制吗？
- Render：免费实例在 15 分钟无活动后会休眠，下次访问需要冷启动（约30秒）
- Hugging Face：完全免费，但资源有限

## 技术细节

### 环境变量
- `PORT`：服务端口（Render 自动设置）
- `PYTHONUNBUFFERED`：建议设置为 `1`（已在代码中处理）

### 模型加载
- 分类模型：EfficientNet-B0 + CBAM 注意力
- 检测模型：YOLOv8-nano
- 推理时间：CPU 约 2-3 秒，GPU 约 0.5 秒

### API 端点
- `GET /` - 前端界面
- `POST /api/analyze` - 分析 X 光图像
- `GET /api/patients` - 获取患者列表
- `GET /api/statistics` - 系统统计信息

## 联系支持
如有问题，请检查：
1. 控制台错误日志
2. Render/Hugging Face 的构建日志
3. 确保所有文件路径正确

祝您部署顺利！