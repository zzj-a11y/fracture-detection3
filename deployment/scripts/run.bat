@echo off
echo 儿童手腕隐匿性骨折检测系统后端
echo =================================

echo 检查依赖...
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo 安装依赖...
    pip install fastapi uvicorn python-multipart -i https://pypi.tuna.tsinghua.edu.cn/simple
)

echo 检查模型文件...
if not exist "model\model_3.pth" (
    echo 错误: 分类模型未找到 (model/model_3.pth)
    pause
    exit /b 1
)

if not exist "yolo\yolov8_fracture_best.pt" (
    echo 警告: YOLO模型未找到 (runs/yolov8_fracture_best.pt), YOLO检测将不可用
)

echo 创建必要的目录...
if not exist "exit\uploads" mkdir exit\uploads

echo.
echo 重要提示: 不要使用 http://0.0.0.0:8000 访问
echo 请使用以下地址访问:
echo 1. http://localhost:8000
echo 2. http://127.0.0.1:8000
echo.
echo 是否要在启动后自动打开浏览器? (Y/N)
set /p open_browser=

echo.
echo 启动后端服务器...
echo 按 Ctrl+C 停止服务器
echo.

python backend.py

if /i "%open_browser%"=="Y" (
    start http://localhost:8000
)