#!/bin/bash
# Quick run script for Speaker-Profiles Docker environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODE="compose"
CPU_ONLY=false
DETACH=false
BUILD=false

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

show_usage() {
    echo "Usage: $0 [OPTIONS] [COMMAND]"
    echo ""
    echo "Quick start script for Speaker-Profiles Docker environment"
    echo ""
    echo "Options:"
    echo "  -c, --cpu-only      Use CPU-only version"
    echo "  -d, --detach        Run in background (detached mode)"
    echo "  -b, --build         Build images before running"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Commands (optional):"
    echo "  up                  Start the environment (default)"
    echo "  down                Stop the environment"
    echo "  logs                Show logs"
    echo "  shell               Open shell in container"
    echo "  process FILE        Process an audio file"
    echo "  status              Show container status"
    echo ""
    echo "Examples:"
    echo "  $0                              # Start with GPU support"
    echo "  $0 --cpu-only                  # Start CPU-only version"
    echo "  $0 --detach                    # Start in background"
    echo "  $0 shell                       # Open shell in running container"
    echo "  $0 process /path/to/audio.wav  # Process audio file"
    echo "  $0 down                        # Stop the environment"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--cpu-only)
            CPU_ONLY=true
            shift
            ;;
        -d|--detach)
            DETACH=true
            shift
            ;;
        -b|--build)
            BUILD=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        up|down|logs|shell|process|status)
            MODE="$1"
            shift
            break
            ;;
        -*)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Assume it's a command
            MODE="$1"
            shift
            break
            ;;
    esac
done

# Check if we're in the docker directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the docker/ directory"
    exit 1
fi

# Check if .env exists in current directory
if [ ! -f "./.env" ]; then
    print_warning ".env file not found in current directory"
    if [ -f "./.env.template" ]; then
        print_status "Creating .env from template..."
        cp ./.env.template ./.env
        print_warning "Please edit ./.env with your HUGGING_FACE_ACCESS_TOKEN"
        echo ""
        echo "Required steps:"
        echo "1. Get token from: https://huggingface.co/settings/tokens"
        echo "2. Edit ./.env and replace 'your_hugging_face_token_here' with your actual token"
        echo "3. Run this script again"
        exit 1
    else
        print_error "No .env.template found. Please create ./.env manually"
        exit 1
    fi
fi

# Prepare docker-compose command
COMPOSE_CMD="docker-compose"
if [ "$CPU_ONLY" = true ]; then
    COMPOSE_CMD="$COMPOSE_CMD --profile cpu-only"
fi

# Execute based on mode
case $MODE in
    up|compose)
        print_status "Starting Speaker-Profiles environment..."
        if [ "$CPU_ONLY" = true ]; then
            print_status "Using CPU-only version"
        else
            print_status "Using GPU version (if available)"
        fi
        
        # Build if requested
        if [ "$BUILD" = true ]; then
            print_status "Building images..."
            $COMPOSE_CMD build
        fi
        
        # Start the environment
        if [ "$DETACH" = true ]; then
            $COMPOSE_CMD up -d
            print_success "Environment started in background"
            print_status "View logs with: $0 logs"
            print_status "Open shell with: $0 shell"
        else
            $COMPOSE_CMD up
        fi
        ;;
        
    down)
        print_status "Stopping Speaker-Profiles environment..."
        $COMPOSE_CMD down
        print_success "Environment stopped"
        ;;
        
    logs)
        print_status "Showing logs..."
        $COMPOSE_CMD logs -f
        ;;
        
    shell)
        print_status "Opening shell in container..."
        if [ "$CPU_ONLY" = true ]; then
            $COMPOSE_CMD exec speaker-profiles-cpu bash
        else
            $COMPOSE_CMD exec speaker-profiles bash
        fi
        ;;
        
    process)
        if [ $# -eq 0 ]; then
            print_error "Please specify an audio file to process"
            echo "Usage: $0 process /path/to/audio.wav"
            exit 1
        fi
        
        AUDIO_FILE="$1"
        print_status "Processing audio file: $AUDIO_FILE"
        
        if [ "$CPU_ONLY" = true ]; then
            $COMPOSE_CMD exec speaker-profiles-cpu python auto_speaker_tagging_system.py "$AUDIO_FILE"
        else
            $COMPOSE_CMD exec speaker-profiles python auto_speaker_tagging_system.py "$AUDIO_FILE"
        fi
        ;;
        
    status)
        print_status "Container status:"
        $COMPOSE_CMD ps
        echo ""
        print_status "Docker images:"
        docker images | grep speaker-profiles || echo "No speaker-profiles images found"
        ;;
        
    *)
        print_error "Unknown command: $MODE"
        show_usage
        exit 1
        ;;
esac