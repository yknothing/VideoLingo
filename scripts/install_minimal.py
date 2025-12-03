#!/usr/bin/env python3
"""
VideoLingo 最小化安装脚本
用于快速部署和本地测试
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """运行系统命令并处理错误"""
    print(f"{'=' * 50}")
    print(f"🔧 {description}")
    print(f"Running: {cmd}")
    print(f"{'=' * 50}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def install_core_dependencies():
    """安装核心依赖包"""
    core_packages = [
        "streamlit==1.38.0",
        "openai==1.55.3", 
        "requests==2.32.3",
        "PyYAML==6.0.2",
        "pandas==2.2.3",
        "numpy==1.26.4",
        "python-dotenv",
        "json-repair",
        "ruamel.yaml",
        "yt-dlp",
        "moviepy==1.0.3",
        "pydub==0.25.1",
        "openpyxl==3.1.5",
        "spacy==3.7.4",
        "edge-tts",
    ]
    
    print("📦 Installing core dependencies...")
    for package in core_packages:
        if not run_command(f"pip install {package}", f"Installing {package}"):
            print(f"⚠️  Warning: Failed to install {package}, continuing...")
    
    return True

def create_minimal_config():
    """创建最小化配置"""
    config_content = """# VideoLingo 最小化配置文件
# 生成时间: 2025-01-29

# API配置 - OpenRouter示例 (推荐)
api:
  key: ''  # 请填入您的OpenRouter API Key: sk-or-v1-xxxx
  base_url: 'https://openrouter.ai/api/v1'  # OpenRouter API端点
  model: 'anthropic/claude-3.5-sonnet'  # 推荐模型
  llm_support_json: true

# 基础设置
target_language: 'zh-CN'
resolution: '1080p'
max_workers: 2

# 视频存储配置
video_storage:
  base_path: './output'  # 本地输出目录
  
# 简化的TTS配置
tts:
  method: 'edge'  # 使用免费的Edge TTS
  voice: 'zh-CN-XiaoxiaoNeural'
  
# 简化的ASR配置  
asr:
  method: 'openai_api'  # 使用OpenAI API进行语音识别
  
# 翻译配置
translation:
  target_language: 'zh-CN'
  chunk_size: 800
  
# 输出配置
output:
  subtitle_enabled: true
  audio_enabled: true
"""
    
    config_path = Path("config.yaml")
    if not config_path.exists():
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config_content)
        print(f"✅ Created minimal config: {config_path}")
    else:
        print(f"ℹ️  Config file already exists: {config_path}")

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║               VideoLingo 最小化安装程序                    ║
║                      v1.0.0                             ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("🎯 目标: 快速安装VideoLingo核心功能用于本地测试")
    print("📋 包含: Streamlit界面、基础翻译、TTS功能")
    print()
    
    # 检查Python版本
    python_version = sys.version_info
    if python_version < (3, 8):
        print("❌ Error: Python 3.8+ required")
        sys.exit(1)
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # 安装核心依赖
    if not install_core_dependencies():
        print("❌ Core dependencies installation failed")
        sys.exit(1)
    
    # 创建配置文件
    create_minimal_config()
    
    # 创建输出目录
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    (output_dir / "log").mkdir(exist_ok=True)
    print(f"✅ Created output directory: {output_dir}")
    
    print(f"""
🎉 VideoLingo 最小化安装完成！

📝 下一步:
1. 编辑 config.yaml 文件，填入您的API密钥
2. 运行: streamlit run st.py
3. 在浏览器中访问: http://localhost:8501

🔧 API配置建议:
- OpenRouter (推荐): https://openrouter.ai/
- 支持多种模型: Claude, GPT, Gemini等
- 配置示例已写入 config.yaml

🚀 启动命令: streamlit run st.py
    """)

if __name__ == "__main__":
    main()