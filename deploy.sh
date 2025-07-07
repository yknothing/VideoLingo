#!/bin/bash

# ------------
# VideoLingo Docker Auto Deploy Script
# ------------

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
}

# Check if docker-compose is available
check_docker_compose() {
    if command -v docker-compose &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker-compose"
    elif docker compose version &> /dev/null; then
        DOCKER_COMPOSE_CMD="docker compose"
    else
        print_error "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
    print_success "Docker Compose is available: $DOCKER_COMPOSE_CMD"
}

# Check NVIDIA GPU support
check_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU detected"
        GPU_SUPPORT=true
    else
        print_warning "NVIDIA GPU not detected or drivers not installed"
        GPU_SUPPORT=false
    fi
}

# Create necessary directories
create_directories() {
    print_info "Creating necessary directories..."
    mkdir -p output
    mkdir -p _model_cache
    print_success "Directories created"
}

# Stop existing containers
stop_existing() {
    print_info "Stopping existing VideoLingo containers..."
    if docker ps -a --format '{{.Names}}' | grep -q "videolingo"; then
        docker stop videolingo 2>/dev/null || true
        docker rm videolingo 2>/dev/null || true
        print_success "Existing containers stopped and removed"
    else
        print_info "No existing containers found"
    fi
}

# Deploy using docker-compose
deploy_with_compose() {
    print_info "Deploying with Docker Compose..."
    
    if [ "$GPU_SUPPORT" = false ]; then
        print_warning "GPU support not available, creating CPU-only compose file..."
        # Create a CPU-only docker-compose override
        cat > docker-compose.override.yml << EOF
version: '3.8'
services:
  videolingo:
    deploy:
      resources: {}
EOF
    fi
    
    $DOCKER_COMPOSE_CMD up -d --build
    print_success "Docker Compose deployment completed"
}

# Deploy using docker run
deploy_with_docker() {
    print_info "Deploying with Docker run..."
    
    # Build the image
    docker build -t videolingo .
    
    # Prepare docker run command
    DOCKER_RUN_CMD="docker run -d --name videolingo -p 8501:8501"
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD -v $(pwd)/config.yaml:/app/config.yaml"
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD -v $(pwd)/output:/app/output"
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD -v $(pwd)/custom_terms.xlsx:/app/custom_terms.xlsx"
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD -v $(pwd)/_model_cache:/app/_model_cache"
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD --restart unless-stopped"
    
    if [ "$GPU_SUPPORT" = true ]; then
        DOCKER_RUN_CMD="$DOCKER_RUN_CMD --gpus all"
    fi
    
    DOCKER_RUN_CMD="$DOCKER_RUN_CMD videolingo"
    
    # Run the container
    eval $DOCKER_RUN_CMD
    print_success "Docker run deployment completed"
}

# Check deployment status
check_deployment() {
    print_info "Checking deployment status..."
    sleep 10
    
    if docker ps | grep -q "videolingo"; then
        print_success "VideoLingo container is running"
        print_info "Access VideoLingo at: http://localhost:8501"
        
        # Show logs
        print_info "Container logs (last 20 lines):"
        docker logs --tail 20 videolingo
    else
        print_error "VideoLingo container is not running"
        print_info "Container logs:"
        docker logs videolingo 2>/dev/null || true
        exit 1
    fi
}

# Main deployment function
main() {
    print_info "Starting VideoLingo Docker deployment..."
    
    # Check prerequisites
    check_docker
    check_docker_compose
    check_nvidia_gpu
    
    # Prepare environment
    create_directories
    stop_existing
    
    # Choose deployment method
    if [ "$1" = "--docker-run" ]; then
        deploy_with_docker
    else
        deploy_with_compose
    fi
    
    # Verify deployment
    check_deployment
    
    print_success "VideoLingo deployment completed successfully!"
    print_info "You can now access VideoLingo at http://localhost:8501"
}

# Handle script arguments
case "$1" in
    --help|-h)
        echo "VideoLingo Docker Auto Deploy Script"
        echo ""
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --docker-run    Use 'docker run' instead of docker-compose"
        echo "  --help, -h      Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0                Deploy using docker-compose (recommended)"
        echo "  $0 --docker-run   Deploy using docker run"
        ;;
    *)
        main "$@"
        ;;
esac 