# VideoLingo 一键部署指南

**更新时间**: 2025-08-10  
**部署方式**: 一键式自动化部署  

---

## 🚀 快速开始 (一键部署)

### 方式一：本地部署 (推荐新手)
```bash
python deploy.py local
```

### 方式二：Docker部署 (推荐服务器)
```bash
python deploy.py docker
```

### 方式三：交互式选择
```bash
python deploy.py
```

**就是这么简单！** 🎉

---

## 📋 系统要求

### 最低要求
- **Python**: 3.8+
- **内存**: 4GB RAM
- **存储**: 5GB 可用空间
- **网络**: 稳定的互联网连接

### 推荐配置
- **Python**: 3.10+
- **内存**: 8GB RAM (支持本地GPU加速)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (可选)
- **存储**: 20GB SSD

### 可选组件
- **FFmpeg**: 用于视频处理 (自动检测和提示安装)
- **Docker**: 用于容器化部署 (Docker模式需要)

---

## 🔧 部署详情

### 本地部署 (`python deploy.py local`)
自动执行以下步骤：
1. ✅ 检查Python环境 (3.8+)
2. ✅ 检查FFmpeg (可选但推荐)
3. ✅ 自动安装Python依赖
4. ✅ 智能检测GPU并安装对应PyTorch版本
5. ✅ 交互式API配置 (支持OpenRouter)
6. ✅ 创建必要目录结构
7. ✅ 启动Web界面 (http://localhost:8501)

### Docker部署 (`python deploy.py docker`)
自动执行以下步骤：
1. ✅ 检查Docker环境
2. ✅ 检查GPU支持并自动配置
3. ✅ 停止现有容器 (如果存在)
4. ✅ 构建并启动新容器
5. ✅ 验证部署状态
6. ✅ 提供访问地址和日志查看方式

---

## ⚙️ 配置说明

### API配置 (必需)
部署时会提示配置以下API信息：

**推荐使用 OpenRouter** (免费额度 + 多模型支持):
- 网站: https://openrouter.ai/
- API Key: `sk-or-v1-xxxxxxxxxx`
- Base URL: `https://openrouter.ai/api/v1`
- 推荐模型: `anthropic/claude-3.5-sonnet`

**其他支持的API**:
- OpenAI Compatible APIs
- 任何支持OpenAI格式的API服务

### 存储配置
- **输出目录**: `./output/` (可在Web界面修改)
- **模型缓存**: `./_model_cache/`
- **配置文件**: `./config.yaml`

---

## 🌐 访问应用

部署完成后：
- **本地访问**: http://localhost:8501
- **界面语言**: 简体中文/English (可切换)
- **主要功能**:
  - YouTube视频下载
  - 视频上传处理  
  - AI转录和翻译
  - 字幕生成和烧录
  - TTS语音合成和配音

---

## 🛠️ 管理命令

### Docker部署管理
```bash
# 查看运行状态
docker ps

# 查看日志
docker logs videolingo

# 停止服务
docker stop videolingo

# 重新部署
python deploy.py docker
```

### 本地部署管理
```bash
# 停止服务: Ctrl+C

# 重新启动
python deploy.py local

# 或直接使用streamlit
streamlit run st.py
```

---

## 🔍 故障排除

### 常见问题

**1. Python版本过低**
```bash
# 检查Python版本
python --version

# 需要Python 3.8+，建议升级到3.10+
```

**2. 依赖安装失败**
```bash
# 更新pip
python -m pip install --upgrade pip

# 重新运行部署
python deploy.py local
```

**3. Docker权限问题**
```bash
# Linux/macOS
sudo usermod -aG docker $USER
# 然后重新登录

# Windows: 确保Docker Desktop正在运行
```

**4. API配置错误**
- 检查API Key格式是否正确
- 确认网络连接正常
- 在Web界面中重新配置API设置

### 获取帮助
- **文档**: 查看项目README
- **问题反馈**: https://github.com/Huanshere/VideoLingo/issues
- **在线答疑**: https://share.fastgpt.in/chat/share?shareId=066w11n3r9aq6879r4z0v9rh

---

## 🎯 使用流程

1. **启动应用**: 使用一键部署命令
2. **配置API**: 按提示输入API密钥
3. **访问界面**: 打开 http://localhost:8501
4. **处理视频**:
   - 下载YouTube视频 或 上传本地视频
   - 选择转录和翻译设置
   - 点击处理按钮
   - 等待完成并下载结果

---

## 📊 与旧版本对比

| 特性 | 旧版本 | 新版本 (一键部署) |
|------|--------|-----------------|
| 安装步骤 | 5-10步手动操作 | 1条命令自动完成 |
| 配置复杂度 | 需要手动编辑多个文件 | 交互式引导配置 |
| 环境检测 | 需要手动检查 | 自动检测并提示 |
| GPU支持 | 手动配置PyTorch | 自动检测并安装 |
| Docker部署 | 复杂的compose设置 | 一键自动化部署 |
| 文档复杂度 | 10+页技术文档 | 简洁明了的使用指南 |

---

**🎉 现在就开始使用VideoLingo吧！**

```bash
# 克隆项目
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo

# 一键部署
python deploy.py

# 打开浏览器访问 http://localhost:8501
```

简单三步，即刻体验AI视频翻译和配音！