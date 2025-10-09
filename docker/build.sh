#!/bin/bash
# Speaker-Profiles Docker Build Script for Linux/macOS
# This script builds the Docker images for the Speaker-Profiles project

set -e  # Exit on any error

# Default values
VERSION="gpu"
NO_CACHE=false
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_color() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Function to show usage
usage() {
    print_color $GREEN "🐳 Speaker-Profiles Docker Build Script"
    print_color $GREEN "======================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -v, --version VERSION    Build version: gpu, cpu, or all (default: gpu)"
    echo "  -n, --no-cache          Build without using cache"
    echo "  --verbose               Show verbose build output"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Build GPU version"
    echo "  $0 -v cpu               # Build CPU version"
    echo "  $0 -v all --no-cache    # Build all versions without cache"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -n|--no-cache)
            NO_CACHE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            print_color $RED "❌ Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

print_color $GREEN "🐳 Speaker-Profiles Docker Build Script"
print_color $GREEN "======================================"

# Check if Docker is installed and running
if ! command -v docker &> /dev/null; then
    print_color $RED "❌ Docker is not installed"
    print_color $YELLOW "Please install Docker and ensure it's running"
    exit 1
fi

if ! docker info &> /dev/null; then
    print_color $RED "❌ Docker is not running"
    print_color $YELLOW "Please start Docker and try again"
    exit 1
fi

DOCKER_VERSION=$(docker --version)
print_color $CYAN "✅ Docker detected: $DOCKER_VERSION"

# Check if we're in the correct directory
if [[ ! -f "Dockerfile" ]]; then
    print_color $RED "❌ Dockerfile not found in current directory"
    print_color $YELLOW "Please run this script from the docker folder"
    exit 1
fi

# Prepare build arguments
BUILD_ARGS=()
if [[ "$NO_CACHE" == true ]]; then
    BUILD_ARGS+=(--no-cache)
fi
if [[ "$VERBOSE" == true ]]; then
    BUILD_ARGS+=(--progress=plain)
fi

# Function to build a Docker image
build_docker_image() {
    local image_name=$1
    local dockerfile_name=$2
    local description=$3
    
    print_color $YELLOW "🏗️  Building $description..."
    print_color $CYAN "Image: $image_name"
    print_color $CYAN "Dockerfile: $dockerfile_name"
    
    if docker build "${BUILD_ARGS[@]}" -f "$dockerfile_name" -t "$image_name" .; then
        print_color $GREEN "✅ Successfully built $image_name"
        return 0
    else
        print_color $RED "❌ Failed to build $image_name"
        return 1
    fi
}

# Main build logic
SUCCESS=true

case "${VERSION,,}" in  # Convert to lowercase
    "gpu")
        print_color $CYAN "🚀 Building GPU version only..."
        if ! build_docker_image "speaker-profiles:gpu" "Dockerfile" "GPU-enabled version"; then
            SUCCESS=false
        fi
        ;;
    "cpu")
        print_color $CYAN "🖥️  Building CPU version only..."
        if ! build_docker_image "speaker-profiles:cpu" "Dockerfile.cpu" "CPU-only version"; then
            SUCCESS=false
        fi
        ;;
    "all")
        print_color $CYAN "🔄 Building all versions..."
        
        # Build GPU version
        GPU_SUCCESS=true
        if ! build_docker_image "speaker-profiles:gpu" "Dockerfile" "GPU-enabled version"; then
            GPU_SUCCESS=false
        fi
        
        # Build CPU version
        CPU_SUCCESS=true
        if ! build_docker_image "speaker-profiles:cpu" "Dockerfile.cpu" "CPU-only version"; then
            CPU_SUCCESS=false
        fi
        
        if [[ "$GPU_SUCCESS" == false ]] || [[ "$CPU_SUCCESS" == false ]]; then
            SUCCESS=false
        fi
        ;;
    *)
        print_color $RED "❌ Invalid version specified: $VERSION"
        print_color $YELLOW "Valid options: gpu, cpu, all"
        exit 1
        ;;
esac

# Show results
print_color $GREEN "\n🎯 Build Summary"
print_color $GREEN "==============="

if [[ "$SUCCESS" == true ]]; then
    print_color $GREEN "✅ All builds completed successfully!"
    
    # Show built images
    print_color $CYAN "\n📋 Available images:"
    if command -v docker &> /dev/null; then
        docker images speaker-profiles --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}\t{{.CreatedAt}}"
    fi
    
    print_color $GREEN "\n🚀 Next steps:"
    print_color $YELLOW "1. Copy .env.docker and set your HUGGING_FACE_ACCESS_TOKEN"
    print_color $YELLOW "2. Create directories: mkdir -p ../audio ../speakers ../output"
    print_color $YELLOW "3. Run with: docker-compose up speaker-profiles-gpu"
    print_color $YELLOW "   Or CPU version: docker-compose up speaker-profiles-cpu"
    
else
    print_color $RED "❌ One or more builds failed!"
    exit 1
fi

print_color $GREEN "\n🎉 Build script completed!"