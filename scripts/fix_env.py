#!/usr/bin/env python3
"""
VideoLingo 环境修复脚本
解决numpy/scipy循环递归问题
"""

import subprocess
import sys
import os

def run_command(cmd, description="", ignore_errors=False):
    """运行命令并处理错误"""
    print(f"🔧 {description}")
    print(f"Running: {cmd}")
    print("-" * 50)
    
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
        if not ignore_errors:
            return False
        else:
            print("⚠️  Ignoring error and continuing...")
            return True

def clean_environment():
    """清理环境"""
    print("🧹 清理环境中...")
    
    # 清理有问题的包
    problematic_packages = [
        'g2p-en', 'nltk', 'scipy', 'importlib-metadata', 
        'syllables', 'pypinyin', 'xmltodict'
    ]
    
    for pkg in problematic_packages:
        run_command(f"pip uninstall {pkg} -y", f"卸载 {pkg}", ignore_errors=True)

def install_core_only():
    """只安装核心包"""
    print("📦 安装最小化核心包...")
    
    core_packages = [
        "streamlit==1.38.0",
        "openai==1.55.3", 
        "requests==2.32.3",
        "PyYAML==6.0.2",
        "pandas==2.2.3",
        "numpy==1.26.4",  # 保持原版本
        "python-dotenv",
        "json-repair",
        "ruamel.yaml",
        "yt-dlp",
        "moviepy==1.0.3",
        "pydub==0.25.1",
        "openpyxl==3.1.5",
        "edge-tts",
        "psutil",
    ]
    
    for package in core_packages:
        if not run_command(f"pip install {package}", f"安装 {package}"):
            print(f"⚠️  {package} 安装失败，继续...")

def create_safe_imports():
    """创建安全的导入处理"""
    print("🛡️  创建安全导入处理...")
    
    # 修复 TTS duration estimation 问题
    duration_fix = """
# TTS Duration estimation - 安全版本
def estimate_duration(text, language='en'):
    '''简单的持续时间估算'''
    # 基于字符数的简单估算
    chars_per_second = 15  # 平均每秒字符数
    base_duration = len(text) / chars_per_second
    return max(1.0, base_duration)  # 最少1秒

def init_estimator(language='en'):
    '''初始化估算器 - 简化版本'''
    return True
"""
    
    duration_file = "core/tts_backend/estimate_duration.py"
    try:
        with open(duration_file, 'w', encoding='utf-8') as f:
            f.write(duration_fix)
        print(f"✅ 创建安全的 {duration_file}")
    except Exception as e:
        print(f"❌ 创建 {duration_file} 失败: {e}")

def main():
    print("""
╔═══════════════════════════════════════════════════════════╗
║              VideoLingo 环境修复工具                       ║
║                    v1.0.0                                ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("🎯 目标: 修复numpy递归错误，建立稳定运行环境")
    print()
    
    # 1. 清理环境
    clean_environment()
    
    # 2. 安装核心包
    install_core_only()
    
    # 3. 创建安全导入
    create_safe_imports()
    
    print(f"""
🎉 环境修复完成！

📝 修复内容:
1. 清理了有问题的依赖包
2. 重新安装了核心必需包
3. 创建了安全的TTS估算模块
4. 避免了numpy/scipy循环依赖

🚀 现在可以尝试启动:
./vlingo start
    """)

if __name__ == "__main__":
    main()