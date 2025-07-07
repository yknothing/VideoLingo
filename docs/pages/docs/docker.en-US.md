# 🐳 Docker Deployment Guide

VideoLingo provides a complete Docker deployment solution with one-click automatic deployment support.

## 📋 Prerequisites

### System Requirements
- Docker Engine 20.10+
- Docker Compose V2
- NVIDIA Container Toolkit (optional, for GPU acceleration)

### Install Docker
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
Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

#### macOS
```bash
brew install --cask docker
```

### GPU Support (Optional)
To enable GPU acceleration, install NVIDIA Container Toolkit:

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## 🚀 One-Click Auto Deployment

VideoLingo provides automatic deployment scripts with cross-platform support.

### Linux/macOS
```bash
# Clone the project
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo

# Configure API keys (edit config.yaml)
# For OpenRouter, uncomment and modify the following configuration:
# api:
#   key: 'sk-or-v1-your-openrouter-api-key'
#   base_url: 'https://openrouter.ai/api/v1'
#   model: 'anthropic/claude-3.5-sonnet'
#   llm_support_json: true

# Run auto deployment script
./deploy.sh
```

### Windows
```cmd
# Clone the project
git clone https://github.com/Huanshere/VideoLingo.git
cd VideoLingo

# Configure API keys (edit config.yaml)
# Run auto deployment script
deploy.bat
```

## 📝 Deployment Script Features

The auto deployment script performs the following operations:

1. **Environment Check**
   - Check if Docker is installed
   - Check if Docker Compose is available
   - Detect NVIDIA GPU support

2. **Environment Preparation**
   - Create necessary directories (`output`, `_model_cache`)
   - Stop and clean existing containers

3. **Deploy Application**
   - Build Docker image
   - Start service container
   - Configure volume mounts and port mapping

4. **Verify Deployment**
   - Check container running status
   - Display access URL and log information

## 🔧 Manual Deployment

If you need manual control over the deployment process, use the following commands:

### Using Docker Compose (Recommended)
```bash
# Build and start services
docker-compose up -d --build

# View logs
docker-compose logs -f videolingo

# Stop services
docker-compose down
```

### Using Docker Run
```bash
# Build image
docker build -t videolingo .

# Run container (with GPU support)
docker run -d --name videolingo \
  --gpus all \
  -p 8501:8501 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/custom_terms.xlsx:/app/custom_terms.xlsx \
  -v $(pwd)/_model_cache:/app/_model_cache \
  --restart unless-stopped \
  videolingo

# Run container (CPU only)
docker run -d --name videolingo \
  -p 8501:8501 \
  -v $(pwd)/config.yaml:/app/config.yaml \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/custom_terms.xlsx:/app/custom_terms.xlsx \
  -v $(pwd)/_model_cache:/app/_model_cache \
  --restart unless-stopped \
  videolingo
```

## 🌐 Access Application

After successful deployment, access VideoLingo at:

- **Local access**: http://localhost:8501
- **Network access**: http://[your-ip-address]:8501

## 📊 Monitoring and Management

### Check container status
```bash
docker ps | grep videolingo
```

### View logs
```bash
docker logs -f videolingo
```

### Enter container
```bash
docker exec -it videolingo bash
```

### Update application
```bash
# Stop container
docker-compose down

# Pull latest code
git pull

# Rebuild and start
docker-compose up -d --build
```

## 🔧 Configuration

### Environment Variables
Set the following environment variables in `docker-compose.yml`:

```yaml
environment:
  - PYTHONUNBUFFERED=1  # Real-time Python output
```

### Volume Mapping
The following directories are mapped to the host:

- `./config.yaml` → `/app/config.yaml` (Configuration file)
- `./output` → `/app/output` (Output files)
- `./custom_terms.xlsx` → `/app/custom_terms.xlsx` (Custom terms)
- `./_model_cache` → `/app/_model_cache` (Model cache)

### Port Configuration
- Default port: `8501` (Streamlit application)
- To modify port, edit the `ports` setting in `docker-compose.yml`

## ❓ Troubleshooting

### GPU Support Issues
If GPU is not recognized, check:
1. NVIDIA drivers are properly installed
2. NVIDIA Container Toolkit is installed
3. Docker has been restarted

### Permission Issues
If you encounter permission issues:
```bash
# Linux/macOS
sudo chmod +x deploy.sh

# Or add user to docker group
sudo usermod -aG docker $USER
```

### Port Conflicts
If port 8501 is occupied:
1. Modify port mapping in `docker-compose.yml`
2. Or stop the service using the port

### Memory Issues
For large model processing, recommend:
- At least 8GB RAM
- If using GPU, recommend 8GB+ VRAM

## 🔄 Uninstall

Complete cleanup of VideoLingo Docker deployment:

```bash
# Stop and remove containers
docker-compose down

# Remove image
docker rmi videolingo

# Clean data (optional)
rm -rf output _model_cache
``` 