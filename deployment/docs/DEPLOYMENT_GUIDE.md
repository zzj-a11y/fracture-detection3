# 儿童手腕隐匿性骨折检测系统 - 部署指南

## 最终部署包准备步骤

### 1. 清理项目文件夹
运行以下脚本清理临时文件：
```
双击运行: cleanup.bat
```

### 2. 测试系统功能
使用以下启动脚本测试系统（避免编码问题）：
```
双击运行: start_final.bat     # 英文界面，无编码问题
或
双击运行: start_ascii.bat     # 纯英文版本
```

### 3. 必需文件清单
部署包必须包含以下文件：

```
骨折检测系统/
├── 核心文件/
│   ├── backend.py                    # FastAPI后端
│   ├── 完整功能版_6.html             # 前端界面
│   ├── requirements.txt              # 依赖列表
│   └── model/model_3.pth            # 分类模型（必需）
├── 启动脚本/
│   ├── start_final.bat               # 推荐启动脚本（英文）
│   ├── start_ascii.bat               # 纯英文启动脚本
│   ├── start_english.bat             # 英文启动脚本
│   └── cleanup.bat                   # 清理工具
├── 模型文件/
│   ├── yolo/yolov8_fracture_best.pt  # YOLO模型（您的yolo目录）
│   └── runs/                         # 自动创建（为空）
├── 目录结构/
│   └── exit/uploads/                 # 上传目录（自动创建）
└── 说明文档/
    ├── DEPLOYMENT_GUIDE.md           # 本文件
    ├── README_一键运行.txt           # 用户使用说明
    └── 部署说明.txt                  # 详细打包指南
```

### 4. 压缩打包
1. 选中整个项目文件夹（zzj）
2. 右键 → 发送到 → 压缩(zipped)文件夹
3. 重命名为：`儿童手腕隐匿性骨折检测系统_部署包.zip`

### 5. 给用户的说明
将以下文件包含在压缩包根目录：
- `start_final.bat` - 主启动脚本
- `README_一键运行.txt` - 使用说明

## 启动脚本说明

### start_final.bat（推荐）
- 全英文界面，无编码问题
- 自动处理模型文件路径（yolo ↔ runs）
- 详细的步骤提示
- 自动打开前端界面

### 如果您需要中文界面
1. 用记事本打开 `start.bat`
2. 点击"文件" → "另存为"
3. 在"编码"下拉菜单中选择"ANSI"
4. 保存并覆盖原文件
5. 双击运行 `start.bat`

## 系统要求
- Windows 7/8/10/11
- Python 3.8+（自动检测）
- 至少4GB内存
- 2GB可用磁盘空间

## 首次运行流程
1. 用户解压部署包
2. 双击 `start_final.bat`
3. 脚本自动检查Python
4. 自动安装依赖（首次运行需要联网）
5. 启动后端服务 (http://localhost:8000)
6. 自动打开前端界面

## 故障排除

### 问题1：双击没反应
```
解决方案：从命令行运行查看具体错误
1. 按 Win+R 输入 cmd
2. 输入：cd "路径到项目文件夹"
3. 输入：start_final.bat
```

### 问题2：端口8000被占用
```
解决方案：
1. 关闭其他占用8000端口的程序
2. 或修改 backend.py 中的端口号（第805行）
```

### 问题3：模型文件缺失
```
确保以下文件存在：
- model/model_3.pth
- yolo/yolov8_fracture_best.pt 或 runs/yolov8_fracture_best.pt
```

### 问题4：依赖安装失败
```
手动安装：
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 技术细节
- 后端：FastAPI (http://localhost:8000)
- 前端：HTML/JavaScript (本地文件)
- 通信：CORS允许本地文件访问
- 模型：PyTorch + YOLOv8
- 依赖：requirements.txt 包含所有必需包

## 最终检查清单
- [ ] 运行 cleanup.bat 清理临时文件
- [ ] 测试 start_final.bat 能正常启动
- [ ] 验证前端能上传图片并分析
- [ ] 检查所有必需文件都存在
- [ ] 压缩整个文件夹（不是单个文件）
- [ ] 提供 README_一键运行.txt 给用户

## 联系方式
如有问题，请联系项目开发者。

---

**注意：** 本系统为医疗辅助诊断工具，不能替代专业医生的临床诊断。