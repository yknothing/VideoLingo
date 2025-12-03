#!/usr/bin/env python3
"""
VideoLingo 一键部署脚本
支持本地和Docker两种部署方式，一条命令完成所有操作
"""

import os
import sys
import subprocess
import platform
import json
import time
import shutil
from pathlib import Path


class VideoLingoDeployment:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.config_file = self.root_dir / "config.yaml"
        self.requirements_file = self.root_dir / "requirements.txt"

    def print_banner(self):
        print(
            """
╔══════════════════════════════════════════════════════════════╗
║                    VideoLingo 一键部署                        ║
║                   One-Click Deployment                        ║
╚══════════════════════════════════════════════════════════════╝
        """
        )

    def print_status(self, message, status="info"):
        icons = {"info": "ℹ️", "success": "✅", "error": "❌", "warning": "⚠️"}
        print(f"{icons.get(status, 'ℹ️')} {message}")

    def check_python(self):
        """检查Python版本"""
        if sys.version_info < (3, 8):
            self.print_status("Python 3.8+ required", "error")
            return False
        self.print_status(f"Python {sys.version.split()[0]} ✓", "success")
        return True

    def check_ffmpeg(self):
        """检查FFmpeg"""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            self.print_status("FFmpeg ✓", "success")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.print_status("FFmpeg not found", "warning")
            self.print_install_ffmpeg_instructions()
            return False

    def print_install_ffmpeg_instructions(self):
        system = platform.system().lower()
        instructions = {
            "darwin": "brew install ffmpeg",
            "linux": "sudo apt update && sudo apt install ffmpeg",
            "windows": "使用 Chocolatey: choco install ffmpeg",
        }
        cmd = instructions.get(system, "请安装 FFmpeg")
        print(f"   安装命令: {cmd}")

    def install_dependencies(self):
        """安装Python依赖"""
        if not self.requirements_file.exists():
            self.print_status("requirements.txt not found", "error")
            return False

        self.print_status("Installing Python dependencies...", "info")
        try:
            # 检测GPU并选择合适的torch版本
            gpu_available = self.check_gpu()
            if gpu_available:
                self.print_status("Installing CUDA-enabled PyTorch...", "info")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cu118",
                    ],
                    check=True,
                )

            # 安装其他依赖
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)],
                check=True,
            )

            self.print_status("Dependencies installed ✓", "success")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Failed to install dependencies: {e}", "error")
            return False

    def check_gpu(self):
        """检查GPU支持"""
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if result.returncode == 0:
                self.print_status("NVIDIA GPU detected ✓", "success")
                return True
            else:
                self.print_status("nvidia-smi failed", "warning")
                return False
        except FileNotFoundError:
            self.print_status("nvidia-smi not found", "warning")
            return False

        self.print_status("No NVIDIA GPU detected, using CPU mode", "warning")
        return False

    def setup_config(self):
        """设置配置文件"""
        if self.config_file.exists():
            response = input("🔧 Configuration file exists. Reconfigure? (y/N): ")
            if response.lower() != "y":
                return True

        print("\n🔑 API Configuration")
        print("推荐使用 OpenRouter (支持多种AI模型)")
        print("获取免费API Key: https://openrouter.ai/")

        # 交互式配置
        api_key = input("API Key (sk-or-v1-xxx): ").strip()
        if not api_key:
            api_key = "your-api-key-here"
            self.print_status("使用默认配置，请稍后在config.yaml中修改API Key", "warning")

        api_base = (
            input("API Base URL [https://openrouter.ai/api/v1]: ").strip()
            or "https://openrouter.ai/api/v1"
        )
        model = (
            input("Model [anthropic/claude-3.5-sonnet]: ").strip() or "anthropic/claude-3.5-sonnet"
        )

        # 创建配置文件
        config_content = f"""# VideoLingo Configuration
api:
  key: '{api_key}'
  base_url: '{api_base}'
  model: '{model}'
  llm_support_json: true

# Display settings
display_language: 'zh-CN'

# Processing settings
target_language: 'Chinese'
resolution: '1080p'

# Video storage
video_storage:
  base_path: './output'

# TTS settings
tts_method: 'edge_tts'
edge_tts:
  voice: 'zh-CN-XiaoxiaoNeural'

# Whisper settings  
whisper:
  language: 'auto'
  runtime: 'local'

# Other settings
burn_subtitles: false
demucs: false
max_workers: 2
"""

        with open(self.config_file, "w", encoding="utf-8") as f:
            f.write(config_content)

        self.print_status("Configuration created ✓", "success")
        return True

    def create_output_dirs(self):
        """创建输出目录"""
        dirs = ["output", "_model_cache"]
        for dir_name in dirs:
            (self.root_dir / dir_name).mkdir(exist_ok=True)
        self.print_status("Output directories created ✓", "success")

    def start_local(self):
        """启动本地服务"""
        self.print_status("Starting VideoLingo locally...", "info")

        # 检查streamlit是否安装
        try:
            import streamlit
        except ImportError:
            self.print_status("Streamlit not installed", "error")
            return False

        # 启动streamlit
        try:
            cmd = [sys.executable, "-m", "streamlit", "run", "st.py", "--server.port", "8501"]
            self.print_status("🌐 Opening VideoLingo at http://localhost:8501", "success")
            self.print_status("Press Ctrl+C to stop", "info")
            subprocess.run(cmd, cwd=self.root_dir)
        except KeyboardInterrupt:
            self.print_status("Service stopped by user", "info")
        except Exception as e:
            self.print_status(f"Failed to start service: {e}", "error")
            return False

        return True

    def deploy_local(self):
        """本地部署"""
        self.print_status("🏠 Local Deployment Selected", "info")

        if not self.check_python():
            return False

        # FFmpeg检查（非必需，但建议安装）
        self.check_ffmpeg()

        # 安装依赖
        if not self.install_dependencies():
            return False

        # 设置配置
        if not self.setup_config():
            return False

        # 创建目录
        self.create_output_dirs()

        # 启动服务
        self.start_local()

        return True

    def check_docker(self):
        """检查Docker"""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            self.print_status("Docker ✓", "success")

            # 检查docker-compose
            try:
                subprocess.run(["docker", "compose", "version"], capture_output=True, check=True)
                return "docker compose"
            except subprocess.CalledProcessError:
                try:
                    subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
                    return "docker-compose"
                except subprocess.CalledProcessError:
                    self.print_status("Docker Compose not found", "error")
                    return False

        except (subprocess.CalledProcessError, FileNotFoundError):
            self.print_status("Docker not found. Please install Docker first.", "error")
            return False

    def deploy_docker(self):
        """Docker部署"""
        self.print_status("🐳 Docker Deployment Selected", "info")

        compose_cmd = self.check_docker()
        if not compose_cmd:
            return False

        # 检查GPU支持
        gpu_available = self.check_gpu()

        # 创建目录和基础配置
        self.create_output_dirs()
        if not self.config_file.exists():
            self.setup_config()

        # 停止现有容器
        self.print_status("Stopping existing containers...", "info")
        subprocess.run(["docker", "stop", "videolingo"], capture_output=True)
        subprocess.run(["docker", "rm", "videolingo"], capture_output=True)

        # 构建并启动
        self.print_status("Building and starting Docker container...", "info")
        try:
            if gpu_available:
                subprocess.run(
                    [*compose_cmd.split(), "up", "-d", "--build"], cwd=self.root_dir, check=True
                )
            else:
                # 创建CPU-only覆盖文件
                cpu_override = """version: '3.8'
services:
  videolingo:
    deploy:
      resources: {}
"""
                with open(self.root_dir / "docker-compose.override.yml", "w") as f:
                    f.write(cpu_override)

                subprocess.run(
                    [*compose_cmd.split(), "up", "-d", "--build"], cwd=self.root_dir, check=True
                )

            # 等待服务启动
            self.print_status("Waiting for service to start...", "info")
            time.sleep(10)

            # 检查状态
            result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
            if "videolingo" in result.stdout:
                self.print_status("🌐 VideoLingo is running at http://localhost:8501", "success")
                self.print_status("Use 'docker logs videolingo' to view logs", "info")
                return True
            else:
                self.print_status("Container failed to start", "error")
                subprocess.run(["docker", "logs", "videolingo"])
                return False

        except subprocess.CalledProcessError as e:
            self.print_status(f"Docker deployment failed: {e}", "error")
            return False

    def deploy(self):
        """主部署函数"""
        self.print_banner()

        print("选择部署方式 | Choose deployment method:")
        print("1. 🏠 本地部署 (Local)")
        print("2. 🐳 Docker部署 (Docker)")
        print("3. ❌ 退出 (Exit)")

        while True:
            choice = input("\n请选择 (1/2/3): ").strip()

            if choice == "1":
                return self.deploy_local()
            elif choice == "2":
                return self.deploy_docker()
            elif choice == "3":
                self.print_status("Deployment cancelled", "info")
                return True
            else:
                print("无效选择，请输入 1、2 或 3")


def main():
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h"]:
            print(
                """
VideoLingo One-Click Deployment

Usage:
    python deploy.py           # Interactive deployment
    python deploy.py local     # Direct local deployment  
    python deploy.py docker    # Direct Docker deployment

Examples:
    python deploy.py           # Show menu and choose
    python deploy.py local     # Deploy locally without menu
    python deploy.py docker    # Deploy with Docker without menu
"""
            )
            return

        deployment = VideoLingoDeployment()
        if sys.argv[1] == "local":
            deployment.deploy_local()
        elif sys.argv[1] == "docker":
            deployment.deploy_docker()
        else:
            print("Invalid argument. Use 'local', 'docker', or '--help'")
    else:
        deployment = VideoLingoDeployment()
        deployment.deploy()


if __name__ == "__main__":
    main()
