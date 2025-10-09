#!/bin/bash
# Validation script for Speaker-Profiles Standalone Docker Environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

print_status "🔍 Validating Speaker-Profiles Standalone Docker Environment..."

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ] || [ ! -f "Dockerfile" ]; then
    print_error "Not in the correct directory. Please run this from the docker folder."
    exit 1
fi

print_success "✅ Directory structure looks correct"

# Check required files
required_files=(
    ".env.template"
    "Dockerfile"
    "Dockerfile.cpu"
    "docker-compose.yml"
    "requirements.txt"
    "requirements-docker.txt"
    "auto_speaker_tagging_system.py"
    "simple_auto_tagging_example.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    print_success "✅ All required files present"
else
    print_error "❌ Missing required files:"
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    exit 1
fi

# Check Docker installation
if ! command -v docker &> /dev/null; then
    print_error "❌ Docker not found. Please install Docker."
    exit 1
fi

print_success "✅ Docker is installed: $(docker --version)"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    print_error "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

print_success "✅ Docker Compose is installed: $(docker-compose --version)"

# Check .env file
if [ ! -f ".env" ]; then
    print_warning "⚠️  .env file not found"
    print_status "Creating .env from template..."
    cp .env.template .env
    print_warning "Please edit .env and add your HUGGING_FACE_ACCESS_TOKEN"
    print_status "Get your token from: https://huggingface.co/settings/tokens"
else
    print_success "✅ .env file exists"
    
    # Check if token is set
    if grep -q "your_hugging_face_token_here" .env; then
        print_warning "⚠️  Please update your HUGGING_FACE_ACCESS_TOKEN in .env"
    else
        print_success "✅ HuggingFace token appears to be set"
    fi
fi

# Check required directories
required_dirs=(
    "speakers"
    "output"
    "transcription_output"
    "pretrained_models"
    "audio_files"
)

missing_dirs=()
for dir in "${required_dirs[@]}"; do
    if [ ! -d "$dir" ]; then
        missing_dirs+=("$dir")
    fi
done

if [ ${#missing_dirs[@]} -eq 0 ]; then
    print_success "✅ All required directories exist"
else
    print_warning "⚠️  Creating missing directories..."
    for dir in "${missing_dirs[@]}"; do
        mkdir -p "$dir"
        print_status "Created directory: $dir"
    done
    print_success "✅ All directories created"
fi

# Test Docker build (dry run)
print_status "🧪 Testing Docker build syntax..."
if docker build --dry-run -f Dockerfile . > /dev/null 2>&1; then
    print_success "✅ Dockerfile syntax is valid"
else
    print_warning "⚠️  Dockerfile syntax check failed (this might be normal on some Docker versions)"
fi

if docker build --dry-run -f Dockerfile.cpu . > /dev/null 2>&1; then
    print_success "✅ Dockerfile.cpu syntax is valid"
else
    print_warning "⚠️  Dockerfile.cpu syntax check failed (this might be normal on some Docker versions)"
fi

# Check GPU support (optional)
if command -v nvidia-smi &> /dev/null; then
    print_success "✅ NVIDIA GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
    if docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi > /dev/null 2>&1; then
        print_success "✅ Docker GPU support is working"
    else
        print_warning "⚠️  Docker GPU support not available. Use CPU version: ./run.sh --cpu-only"
    fi
else
    print_status "ℹ️  No NVIDIA GPU detected. CPU version will be used."
fi

print_success "🎉 Validation complete!"

echo ""
echo "Next steps:"
echo "1. If you haven't already, edit .env with your HuggingFace token"
echo "2. Start the environment: ./run.sh"
echo "3. Process an audio file: ./run.sh process /app/audio_files/your_audio.wav"
echo ""
echo "For more information, see README-STANDALONE.md"