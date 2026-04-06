@echo off
echo ========================================================
echo    Children Wrist Fracture Detection System
echo ========================================================

echo 1. Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ and add to PATH.
    pause
    exit /b 1
)
python -c "import sys; print('Python version:', sys.version.split()[0])"

echo.
echo 2. Checking dependencies...
python -c "import fastapi, torch, torchvision, ultralytics, cv2, PIL, numpy, tqdm, imblearn, yaml" >nul 2>&1
if errorlevel 1 (
    echo Installing required packages...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo ERROR: Failed to install dependencies!
        echo Please install manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo Dependencies installed.
) else (
    echo Dependencies already installed.
)

echo.
echo 3. Checking model files...
if not exist "model\model_3.pth" (
    echo ERROR: Classification model not found (model\model_3.pth)
    pause
    exit /b 1
) else (
    echo Classification model: model\model_3.pth
)

echo.
echo 4. Checking YOLO model...
set "YOLO_SOURCE="
if exist "yolo\yolov8_fracture_best.pt" (
    set "YOLO_SOURCE=yolo\yolov8_fracture_best.pt"
    echo YOLO model found in yolo folder.
) else if exist "runs\yolov8_fracture_best.pt" (
    set "YOLO_SOURCE=runs\yolov8_fracture_best.pt"
    echo YOLO model found in runs folder.
) else (
    echo WARNING: YOLO model not found.
    echo YOLO detection will not be available.
)

echo.
echo 5. Preparing directories...
if not exist "exit\uploads" mkdir exit\uploads
if not exist "model" mkdir model
if not exist "yolo" mkdir yolo
if not exist "runs" mkdir runs

echo.
echo 6. Copying YOLO model for backend...
if not "%YOLO_SOURCE%"=="" (
    if not exist "runs\yolov8_fracture_best.pt" (
        echo Copying YOLO model to runs directory...
        copy "%YOLO_SOURCE%" "runs\yolov8_fracture_best.pt" >nul
        if errorlevel 1 (
            echo WARNING: Failed to copy YOLO model.
        ) else (
            echo YOLO model copied to runs directory.
        )
    ) else (
        echo YOLO model already in runs directory.
    )
)

echo.
echo 7. Checking port 8000...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo ERROR: Port 8000 is already in use!
    echo Please close the program using port 8000.
    echo Waiting 10 seconds...
    timeout /t 10 /nobreak >nul
    netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
    if not errorlevel 1 (
        echo ERROR: Port 8000 still in use. Please try again later.
        pause
        exit /b 1
    )
    echo Port 8000 is now available.
)

echo.
echo 8. Starting backend server...
echo Backend API: http://localhost:8000
echo Press Ctrl+C to stop server.
echo.
start "Fracture Detection Backend" python backend.py

echo Waiting 5 seconds for server to start...
timeout /t 5 /nobreak >nul

echo.
echo 9. Opening frontend interface...
REM Try to get short filename for Chinese HTML file
set "FRONTEND_FILE="
for %%I in ("完整功能版_6.html") do set "FRONTEND_FILE=%%~sI"

if exist "完整功能版_6.html" (
    if not "%FRONTEND_FILE%"=="" (
        echo Opening frontend using short name: %FRONTEND_FILE%
        start "" "%FRONTEND_FILE%"
    ) else (
        echo Opening frontend directly...
        start "" "完整功能版_6.html"
    )
) else (
    echo WARNING: Frontend file not found!
    echo Please open manually: http://localhost:8000/docs
)

echo.
echo ========================================================
echo    System Started Successfully!
echo.
echo    Backend API: http://localhost:8000
echo    Frontend: 完整功能版_6.html
echo    API Documentation: http://localhost:8000/docs
echo.
echo    Instructions:
echo    1. Upload X-ray image in frontend
echo    2. View analysis results and medical advice
echo    3. Close all windows when finished
echo ========================================================
echo.
pause
