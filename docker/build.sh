#!/bin/bash
# Build script for Speaker-Profiles Docker images

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
BUILD_GPU=true
BUILD_CPU=true
BUILD_ARGS=""
PUSH=false
TAG_PREFIX=""

# Function to print colored output
print_status() {
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

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build Docker images for Speaker-Profiles"
    echo ""
    echo "Options:"
    echo "  -g, --gpu-only      Build only GPU version"
    echo "  -c, --cpu-only      Build only CPU version"
    echo "  -t, --tag PREFIX    Tag prefix (default: speaker-profiles)"
    echo "  -p, --push          Push images to registry after building"
    echo "  --build-arg ARG     Pass build argument to docker build"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                  # Build both GPU and CPU versions"
    echo "  $0 --gpu-only       # Build only GPU version"
    echo "  $0 --cpu-only       # Build only CPU version"
    echo "  $0 --tag myrepo/speaker-profiles --push"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu-only)
            BUILD_CPU=false
            shift
            ;;
        -c|--cpu-only)
            BUILD_GPU=false
            shift
            ;;
        -t|--tag)
            TAG_PREFIX="$2"
            shift 2
            ;;
        -p|--push)
            PUSH=true
            shift
            ;;
        --build-arg)
            BUILD_ARGS="$BUILD_ARGS --build-arg $2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Set default tag prefix if not specified
if [ -z "$TAG_PREFIX" ]; then
    TAG_PREFIX="speaker-profiles"
fi

# Check if we're in the docker directory
if [ ! -f "Dockerfile" ] || [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the docker/ directory"
    exit 1
fi

print_status "Building Speaker-Profiles Docker images..."
print_status "GPU version: $BUILD_GPU"
print_status "CPU version: $BUILD_CPU"
print_status "Tag prefix: $TAG_PREFIX"

# Build GPU version
if [ "$BUILD_GPU" = true ]; then
    print_status "Building GPU version..."
    if docker build $BUILD_ARGS -t ${TAG_PREFIX}:latest -f Dockerfile ..; then
        print_success "GPU version built successfully: ${TAG_PREFIX}:latest"
        
        if [ "$PUSH" = true ]; then
            print_status "Pushing GPU version..."
            docker push ${TAG_PREFIX}:latest
            print_success "GPU version pushed successfully"
        fi
    else
        print_error "Failed to build GPU version"
        exit 1
    fi
fi

# Build CPU version
if [ "$BUILD_CPU" = true ]; then
    print_status "Building CPU version..."
    if docker build $BUILD_ARGS -t ${TAG_PREFIX}:latest-cpu -f Dockerfile.cpu ..; then
        print_success "CPU version built successfully: ${TAG_PREFIX}:latest-cpu"
        
        if [ "$PUSH" = true ]; then
            print_status "Pushing CPU version..."
            docker push ${TAG_PREFIX}:latest-cpu
            print_success "CPU version pushed successfully"
        fi
    else
        print_error "Failed to build CPU version"
        exit 1
    fi
fi

print_success "Build process completed!"

# Show next steps
echo ""
echo "Next steps:"
echo "  1. Test the images:"
echo "     docker run --rm ${TAG_PREFIX}:latest python --version"
echo "     docker run --rm ${TAG_PREFIX}:latest-cpu python --version"
echo ""
echo "  2. Run with docker-compose:"
echo "     docker-compose up"
echo ""
echo "  3. Process an audio file:"
echo "     docker-compose exec speaker-profiles python auto_speaker_tagging_system.py /app/audio_files/your_audio.wav"