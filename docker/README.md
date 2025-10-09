# 🐳 Speaker-Profiles Docker Environment

This folder contains a complete dockerized environment for the Speaker-Profiles project, enabling easy setup and sharing of the speaker recognition system with all its dependencies.

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [What's Included](#-whats-included)
- [Prerequisites](#-prerequisites)
- [Setup Instructions](#-setup-instructions)
- [Usage Examples](#-usage-examples)
- [Configuration](#-configuration)
- [Available Services](#-available-services)
- [Volume Mounts](#-volume-mounts)
- [Troubleshooting](#-troubleshooting)
- [Development](#-development)
- [Performance Notes](#-performance-notes)

## 🚀 Quick Start

### Windows (PowerShell)
```powershell
# 1. Build the Docker image
.\build.ps1

# 2. Setup environment
.\run.ps1 -Setup

# 3. Edit .env.docker and add your HuggingFace token

# 4. Run the container
.\run.ps1 -Interactive
```

### Linux/macOS (Bash)
```bash
# 1. Make scripts executable
chmod +x build.sh

# 2. Build the Docker image
./build.sh

# 3. Setup environment
mkdir -p ../audio ../speakers ../output

# 4. Copy and configure environment
cp .env.docker .env.docker.local
# Edit .env.docker.local and set your HUGGING_FACE_ACCESS_TOKEN

# 5. Run the container
docker-compose up speaker-profiles-gpu
```

## 📦 What's Included

This Docker environment provides:

- **🎯 Complete AI/ML Stack**: PyTorch, SpeechBrain, pyannote.audio, transformers
- **🔧 Audio Processing**: FFmpeg, SoX, librosa, soundfile
- **🐍 Python Environment**: Python 3.9+ with all required dependencies
- **⚡ GPU Support**: CUDA-enabled PyTorch for accelerated inference
- **💻 CPU Fallback**: CPU-only version for systems without GPU
- **📁 Volume Persistence**: Persistent model cache and speaker database
- **🔒 Security**: Non-root user execution
- **🩺 Health Checks**: Built-in container health monitoring

## 📋 Prerequisites

### Required
- **Docker**: Docker Desktop (Windows/macOS) or Docker Engine (Linux)
- **Docker Compose**: Usually included with Docker Desktop
- **HuggingFace Account**: For accessing pyannote.audio models
  - Get your token: https://huggingface.co/settings/tokens

### For GPU Support (Optional)
- **NVIDIA GPU**: Compatible with CUDA 11.8+
- **NVIDIA Container Toolkit**: For Docker GPU access
  - [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### System Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: At least 10GB free space for models and cache
- **CPU**: Multi-core processor recommended

## 🛠 Setup Instructions

### Step 1: Build Docker Images

#### Windows
```powershell
# Build GPU version (default)
.\build.ps1

# Build CPU version
.\build.ps1 -Version cpu

# Build all versions
.\build.ps1 -Version all

# Build without cache (clean build)
.\build.ps1 -NoCache
```

#### Linux/macOS
```bash
# Build GPU version (default)
./build.sh

# Build CPU version
./build.sh -v cpu

# Build all versions
./build.sh -v all

# Build without cache
./build.sh --no-cache
```

### Step 2: Environment Configuration

1. **Copy environment template**:
   ```bash
   cp .env.docker .env.docker.local
   ```

2. **Edit environment file**:
   - Set `HUGGING_FACE_ACCESS_TOKEN=your_actual_token_here`
   - Adjust other settings as needed

3. **Create directory structure**:
   ```bash
   mkdir -p ../audio ../speakers ../output
   ```

### Step 3: Initial Run

```bash
# Start GPU version interactively
docker-compose up speaker-profiles-gpu

# Start CPU version
docker-compose up speaker-profiles-cpu

# Start in background
docker-compose up -d speaker-profiles-gpu
```

## 💡 Usage Examples

### Basic Audio Processing

1. **Place audio files** in the `../audio` directory
2. **Run the container**:
   ```bash
   docker-compose run speaker-profiles-gpu python simple_auto_tagging_example.py /app/audio/your_audio.wav
   ```

### Interactive Development

```bash
# Start development container
docker-compose up speaker-profiles-dev

# Access the container
docker exec -it speaker-profiles-dev bash

# Inside container
cd /app
python simple_auto_tagging_example.py audio/sample.wav
```

### Processing Multiple Files

```bash
# Process all files in audio directory
docker-compose run speaker-profiles-gpu python -c "
import os
from auto_speaker_tagging_system import AutoSpeakerTaggingSystem

system = AutoSpeakerTaggingSystem()
for filename in os.listdir('/app/audio'):
    if filename.endswith(('.wav', '.mp3', '.flac')):
        print(f'Processing {filename}...')
        system.process_audio_file(f'/app/audio/{filename}')
"
```

### Using Windows Scripts

```powershell
# Check container status
.\run.ps1 -Status

# Run setup
.\run.ps1 -Setup

# Start interactively
.\run.ps1 -Interactive

# Start in background
.\run.ps1 -Background

# Process specific file
.\run.ps1 -AudioFile "sample.wav"
```

## ⚙ Configuration

### Environment Variables (.env.docker)

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGING_FACE_ACCESS_TOKEN` | **Required**: HF token for model access | - |
| `TRANSFORMERS_CACHE` | Model cache directory | `/app/pretrained_models/transformers` |
| `HF_HOME` | HuggingFace cache directory | `/app/pretrained_models/huggingface` |
| `SPEECHBRAIN_CACHE` | SpeechBrain cache directory | `/app/pretrained_models/speechbrain` |
| `DEFAULT_SIMILARITY_THRESHOLD` | Speaker matching threshold | `0.75` |
| `MIN_SPEECH_TIME` | Minimum speech duration (seconds) | `2.0` |
| `MIN_QUALITY_THRESHOLD` | Minimum embedding quality | `0.3` |

### Docker Compose Services

| Service | Description | Port | Use Case |
|---------|-------------|------|----------|
| `speaker-profiles-gpu` | GPU-accelerated version | 8000 | Production, fast processing |
| `speaker-profiles-cpu` | CPU-only version | 8001 | Development, testing |
| `speaker-profiles-dev` | Development version | 8002 | Code development, debugging |

## 📁 Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `../audio` | `/app/audio` | Input audio files (read-only) |
| `../speakers` | `/app/speakers` | Speaker database (persistent) |
| `../output` | `/app/output` | Processing results |
| `speaker-models-cache` | `/app/pretrained_models` | Model cache (persistent) |

## 🔧 Available Services

### GPU Version (Default)
- **Image**: `speaker-profiles:gpu`
- **Features**: CUDA acceleration, faster processing
- **Requirements**: NVIDIA GPU with Docker GPU support

### CPU Version (Fallback)
- **Image**: `speaker-profiles:cpu`
- **Features**: CPU-only processing, broader compatibility
- **Requirements**: Standard Docker installation

### Development Version
- **Image**: `speaker-profiles:dev`
- **Features**: Live code mounting, development tools
- **Use**: Code development and testing

## 🐛 Troubleshooting

### Common Issues

#### "HUGGING_FACE_ACCESS_TOKEN not set"
```bash
# Check your .env.docker file
cat .env.docker | grep HUGGING_FACE_ACCESS_TOKEN

# Update the token
echo "HUGGING_FACE_ACCESS_TOKEN=your_token_here" >> .env.docker
```

#### GPU Not Detected
```bash
# Check GPU support
docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi

# If this fails, install NVIDIA Container Toolkit
# https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

#### Model Download Issues
```bash
# Check network connectivity
docker run speaker-profiles:gpu ping -c 3 huggingface.co

# Clear model cache
docker volume rm speaker-profiles_speaker-models-cache
```

#### Permission Issues (Linux)
```bash
# Fix directory permissions
sudo chown -R $(whoami):$(whoami) ../speakers ../output

# Or run with proper user mapping
docker-compose run --user $(id -u):$(id -g) speaker-profiles-gpu bash
```

#### Out of Memory
```bash
# Monitor container memory usage
docker stats speaker-profiles-gpu

# Increase Docker memory limit in Docker Desktop settings
# Or use CPU version for lower memory usage
docker-compose up speaker-profiles-cpu
```

### Debugging Commands

```bash
# Check container logs
docker-compose logs speaker-profiles-gpu

# Access running container
docker exec -it speaker-profiles-gpu bash

# Check Python imports
docker-compose run speaker-profiles-gpu python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Test model loading
docker-compose run speaker-profiles-gpu python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('HF Token set:', bool(os.getenv('HUGGING_FACE_ACCESS_TOKEN')))
"
```

## 👨‍💻 Development

### Code Development Workflow

1. **Start development container**:
   ```bash
   docker-compose up speaker-profiles-dev
   ```

2. **Access container**:
   ```bash
   docker exec -it speaker-profiles-dev bash
   ```

3. **Edit code** on host system (files are mounted)

4. **Test changes** inside container without rebuilding

### Adding New Dependencies

1. **Update requirements file**:
   - Edit `requirements-docker.txt` (GPU) or `requirements-docker-cpu.txt` (CPU)

2. **Rebuild image**:
   ```bash
   ./build.sh --no-cache
   ```

### Custom Configuration

1. **Create custom docker-compose override**:
   ```yaml
   # docker-compose.override.yml
   version: '3.8'
   services:
     speaker-profiles-gpu:
       environment:
         - CUSTOM_ENV_VAR=value
       volumes:
         - ./custom-config:/app/config
   ```

## ⚡ Performance Notes

### GPU Performance
- **First run**: Slower due to model downloads (~5-10 GB)
- **Subsequent runs**: Fast model loading from cache
- **Memory usage**: ~4-6GB GPU memory for typical models

### CPU Performance  
- **Processing time**: 3-5x slower than GPU
- **Memory usage**: ~2-4GB RAM
- **Suitable for**: Testing, small files, development

### Optimization Tips

1. **Pre-warm models**: Run container once to download all models
2. **Use GPU version**: For production workloads
3. **Batch processing**: Process multiple files in single session
4. **Monitor resources**: Use `docker stats` to monitor usage

## 📝 File Structure

```
docker/
├── Dockerfile              # Main GPU-enabled image
├── Dockerfile.cpu          # CPU-only image
├── docker-compose.yml      # Container orchestration
├── requirements-docker.txt # Python dependencies (GPU)
├── requirements-docker-cpu.txt # Python dependencies (CPU)
├── .env.docker            # Environment template
├── build.ps1              # Windows build script
├── build.sh               # Linux build script  
├── run.ps1                # Windows run script
├── README.md              # This file
├── TROUBLESHOOTING.md     # Detailed troubleshooting guide
└── *.py                   # Project Python files
```

## 🤝 Contributing

1. **Test changes** with both GPU and CPU versions
2. **Update documentation** for new features
3. **Test on multiple platforms** (Windows/Linux)
4. **Verify security** (no root execution, proper secrets handling)

## 📄 License

This Docker environment inherits the license from the main Speaker-Profiles project.

---

🎉 **Enjoy using Speaker-Profiles in Docker!** 

For additional help, check the [Troubleshooting Guide](TROUBLESHOOTING.md) or open an issue on the project repository.