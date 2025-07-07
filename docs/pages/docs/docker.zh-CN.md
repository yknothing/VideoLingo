# 🐳 Docker 部署指南

VideoLingo 提供了完整的 Docker 部署方案，支持一键自动部署。

## 📋 前置要求

### 系统要求
- Docker Engine 20.10+
- Docker Compose V2
- NVIDIA Container Toolkit (可选，用于 GPU 加速)

### 安装 Docker
#### Ubuntu/Debian
```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
```

#### CentOS/RHEL
```bash
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install docker-ce docker-ce-cli containerd.io
sudo systemctl start docker
sudo systemctl enable docker
```

#### Windows
下载并安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

#### macOS
```bash
brew install --cask docker
```

### GPU 支持 (可选)
如需 GPU 加速，请安装 NVIDIA Container Toolkit：

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 🚀 一键自动部署

VideoLingo 提供了自动部署脚本，支持跨平台一键部署。

### Linux/macOS
```bash
# 克隆项目
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo

# 配置 API 密钥 (编辑 config.yaml)
# 对于 OpenRouter，请将以下配置取消注释并修改：
# api:
#   key: 'sk-or-v1-your-openrouter-api-key'
#   base_url: 'https://openrouter.ai/api/v1'
#   model: 'anthropic/claude-3.5-sonnet'
#   llm_support_json: true

# 运行自动部署脚本
./deploy.sh
```

### Windows
```cmd
# 克隆项目
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo

# 配置 API 密钥 (编辑 config.yaml)
# 运行自动部署脚本
deploy.bat
```

## 📝 部署脚本功能

自动部署脚本会执行以下操作：

1. **环境检查**
   - 检查 Docker 是否已安装
   - 检查 Docker Compose 是否可用
   - 检测 NVIDIA GPU 支持情况

2. **环境准备**
   - 创建必要的目录 (`output`, `_model_cache`)
   - 停止并清理已存在的容器

3. **部署应用**
   - 构建 Docker 镜像
   - 启动服务容器
   - 配置数据卷和端口映射

4. **验证部署**
   - 检查容器运行状态
   - 显示访问地址和日志信息

## 🔧 手动部署

如果需要手动控制部署过程，可以使用以下命令：

### 使用 Docker Compose (推荐)
```bash
# 构建并启动服务
docker-compose up -d --build

# 查看日志
docker-compose logs -f videolingo

# 停止服务
docker-compose down
```

### 使用 Docker Run
```bash
# 构建镜像
docker build -t videolingo .

# 运行容器 (带 GPU 支持)
docker run -d --name videolingo \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/custom_terms.xlsx:/app/custom_terms.xlsx \
  -v $(pwd)/_model_cache:/app/_model_cache \
  --restart unless-stopped \
  videolingo

# 运行容器 (仅 CPU)
docker run -d --name videolingo \
  -p 8501:8501 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/custom_terms.xlsx:/app/custom_terms.xlsx \
  -v $(pwd)/_model_cache:/app/_model_cache \
  --restart unless-stopped \
  videolingo
```

## 🌐 访问应用

部署成功后，通过以下地址访问 VideoLingo：

- **本地访问**: http://localhost:8501
- **局域网访问**: http://[你的IP地址]:8501

## 📊 监控和管理

### 查看容器状态
```bash
docker ps | grep videolingo
```

### 查看日志
```bash
docker logs -f videolingo
```

### 进入容器
```bash
docker exec -it videolingo bash
```

### 更新应用
```bash
# 停止容器
docker-compose down

# 拉取最新代码
git pull

# 重新构建并启动
docker-compose up -d --build
```

## 🔧 配置说明

### 环境变量
在 `docker-compose.yml` 中可以设置以下环境变量：

```yaml
environment:
  - PYTHONUNBUFFERED=1  # 实时显示 Python 输出
```

### 数据卷映射
以下目录会被映射到主机：

- `./config.yaml` → `/app/config.yaml` (配置文件)
- `./output` → `/app/output` (输出文件)
- `./custom_terms.xlsx` → `/app/custom_terms.xlsx` (自定义术语)
- `./_model_cache` → `/app/_model_cache` (模型缓存)

### 端口配置
- 默认端口：`8501` (Streamlit 应用)
- 如需修改端口，请编辑 `docker-compose.yml` 中的 `ports` 设置

## ❓ 常见问题

### GPU 支持问题
如果 GPU 不被识别，请检查：
1. NVIDIA 驱动是否正确安装
2. NVIDIA Container Toolkit 是否安装
3. Docker 是否重启

### 权限问题
如果遇到权限问题：
```bash
# Linux/macOS
sudo chmod +x deploy.sh

# 或添加用户到 docker 组
sudo usermod -aG docker $USER
```

### 端口占用
如果 8501 端口被占用：
1. 修改 `docker-compose.yml` 中的端口映射
2. 或停止占用端口的服务

### 内存不足
对于大模型处理，建议：
- 至少 8GB 内存
- 如使用 GPU，建议 8GB+ 显存

## 🔄 卸载

完全清理 VideoLingo Docker 部署：

```bash
# 停止并删除容器
docker-compose down

# 删除镜像
docker rmi videolingo

# 清理数据 (可选)
rm -rf output _model_cache
```

