@echo off
echo ========================================
echo   Fracture Detection System Launcher
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found!
    echo Please install Python 3.8+ from python.org
    echo And make sure to check "Add Python to PATH"
    pause
    exit /b 1
)

echo Python check passed.

REM Check classification model
if not exist "model\model_3.pth" (
    echo ERROR: Classification model not found!
    echo Please ensure model\model_3.pth exists.
    pause
    exit /b 1
)

echo Classification model found.

REM Check and copy YOLO model
if exist "yolo\yolov8_fracture_best.pt" (
    echo YOLO model found in yolo folder.
    if not exist "runs\yolov8_fracture_best.pt" (
        echo Copying YOLO model to runs folder...
        copy "yolo\yolov8_fracture_best.pt" "runs\yolov8_fracture_best.pt" >nul
        if errorlevel 1 (
            echo WARNING: Failed to copy YOLO model.
        ) else (
            echo YOLO model copied successfully.
        )
    ) else (
        echo YOLO model already in runs folder.
    )
) else if exist "runs\yolov8_fracture_best.pt" (
    echo YOLO model found in runs folder.
) else (
    echo WARNING: YOLO model not found.
    echo YOLO detection will not be available.
)

REM Create necessary directories
if not exist "exit\uploads" mkdir exit\uploads
if not exist "runs" mkdir runs

REM Check port 8000
echo.
echo Checking port 8000...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
    echo ERROR: Port 8000 is already in use!
    echo Please close the program using port 8000.
    echo Waiting 10 seconds...
    timeout /t 10 /nobreak >nul
    netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
    if not errorlevel 1 (
        echo ERROR: Port 8000 still in use.
        echo Please try again later.
        pause
        exit /b 1
    )
)

echo Port 8000 is available.

REM Start backend
echo.
echo Starting backend server...
echo Backend API: http://localhost:8000
echo Press Ctrl+C to stop the server.
echo.
start "Backend" python backend.py

echo Waiting 5 seconds for server to start...
timeout /t 5 /nobreak >nul

echo.
echo ========================================
echo   SYSTEM STARTED SUCCESSFULLY!
echo.
echo   Backend server is running.
echo   API: http://localhost:8000
echo   API Docs: http://localhost:8000/docs
echo.
echo   NEXT STEP:
echo   1. Open the frontend by double-clicking
echo      the HTML file: "完整功能版_6.html"
echo   2. Upload X-ray images for analysis
echo   3. View results and medical advice
echo.
echo   To stop the system, close this window
echo   and the backend window.
echo ========================================
echo.
pause
