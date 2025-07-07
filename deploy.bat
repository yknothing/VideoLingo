@echo off
REM ------------
REM VideoLingo Docker Auto Deploy Script for Windows
REM ------------

setlocal enabledelayedexpansion

echo [INFO] Starting VideoLingo Docker deployment...

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed. Please install Docker Desktop first.
    pause
    exit /b 1
)
echo [SUCCESS] Docker is installed

REM Check if docker-compose is available
docker-compose --version >nul 2>&1
if not errorlevel 1 (
    set DOCKER_COMPOSE_CMD=docker-compose
    echo [SUCCESS] Docker Compose is available: docker-compose
) else (
    docker compose version >nul 2>&1
    if not errorlevel 1 (
        set DOCKER_COMPOSE_CMD=docker compose
        echo [SUCCESS] Docker Compose is available: docker compose
    ) else (
        echo [ERROR] Docker Compose is not available. Please install Docker Compose.
        pause
        exit /b 1
    )
)

REM Check NVIDIA GPU support
nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [SUCCESS] NVIDIA GPU detected
    set GPU_SUPPORT=true
) else (
    echo [WARNING] NVIDIA GPU not detected or drivers not installed
    set GPU_SUPPORT=false
)

REM Create necessary directories
echo [INFO] Creating necessary directories...
if not exist "output" mkdir output
if not exist "_model_cache" mkdir _model_cache
echo [SUCCESS] Directories created

REM Stop existing containers
echo [INFO] Stopping existing VideoLingo containers...
for /f "tokens=*" %%i in ('docker ps -a --format "{{.Names}}" 2^>nul ^| findstr "videolingo"') do (
    echo [INFO] Found existing container: %%i
    docker stop %%i >nul 2>&1
    docker rm %%i >nul 2>&1
    echo [SUCCESS] Existing containers stopped and removed
    goto :continue
)
echo [INFO] No existing containers found
:continue

REM Deploy using docker-compose
echo [INFO] Deploying with Docker Compose...

if "%GPU_SUPPORT%"=="false" (
    echo [WARNING] GPU support not available, creating CPU-only compose file...
    (
        echo version: '3.8'
        echo services:
        echo   videolingo:
        echo     deploy:
        echo       resources: {}
    ) > docker-compose.override.yml
)

%DOCKER_COMPOSE_CMD% up -d --build
if errorlevel 1 (
    echo [ERROR] Docker Compose deployment failed
    pause
    exit /b 1
)
echo [SUCCESS] Docker Compose deployment completed

REM Check deployment status
echo [INFO] Checking deployment status...
timeout /t 10 /nobreak >nul

docker ps | findstr "videolingo" >nul
if not errorlevel 1 (
    echo [SUCCESS] VideoLingo container is running
    echo [INFO] Access VideoLingo at: http://localhost:8501
    echo [INFO] Container logs (last 20 lines^):
    docker logs --tail 20 videolingo
) else (
    echo [ERROR] VideoLingo container is not running
    echo [INFO] Container logs:
    docker logs videolingo 2>nul
    pause
    exit /b 1
)

echo [SUCCESS] VideoLingo deployment completed successfully!
echo [INFO] You can now access VideoLingo at http://localhost:8501
pause 