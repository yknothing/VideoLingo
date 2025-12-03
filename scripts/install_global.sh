#!/bin/bash

# VideoLingo 全局安装脚本
echo "🚀 VideoLingo 全局安装脚本"
echo "=========================="

VLINGO_PATH="/Users/whatsup/workspace/VideoLingo/vlingo"
GLOBAL_PATH="/usr/local/bin/vlingo"

# 检查vlingo文件是否存在
if [ ! -f "$VLINGO_PATH" ]; then
    echo "❌ 错误: vlingo文件不存在: $VLINGO_PATH"
    exit 1
fi

# 检查权限
if [ ! -x "$VLINGO_PATH" ]; then
    echo "🔧 添加执行权限..."
    chmod +x "$VLINGO_PATH"
fi

# 尝试创建全局链接
echo "🔗 创建全局命令链接..."
if [ -w "/usr/local/bin" ]; then
    # 用户有写权限
    ln -sf "$VLINGO_PATH" "$GLOBAL_PATH"
    echo "✅ 全局命令已安装: vlingo"
else
    # 需要sudo权限
    echo "需要管理员权限来安装全局命令..."
    sudo ln -sf "$VLINGO_PATH" "$GLOBAL_PATH"
    if [ $? -eq 0 ]; then
        echo "✅ 全局命令已安装: vlingo"
    else
        echo "⚠️  全局安装失败，使用本地路径..."
        echo "📝 添加到 ~/.zshrc 或 ~/.bashrc:"
        echo "export PATH=\"/Users/whatsup/workspace/VideoLingo:\$PATH\""
        echo ""
        echo "或直接使用: /Users/whatsup/workspace/VideoLingo/vlingo"
    fi
fi

# 测试命令
echo ""
echo "🧪 测试命令..."
if command -v vlingo >/dev/null 2>&1; then
    vlingo --help
    echo ""
    echo "✅ 安装成功! 现在可以使用 'vlingo' 命令"
    echo ""
    echo "📋 常用命令:"
    echo "  vlingo setup    # 配置向导"
    echo "  vlingo start    # 启动应用"
    echo "  vlingo status   # 查看状态"
    echo "  vlingo stop     # 停止应用"
else
    echo "❌ 全局命令安装失败"
    echo "💡 可以直接使用: $VLINGO_PATH"
fi