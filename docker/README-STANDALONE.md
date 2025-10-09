# 🐳 Speaker-Profiles Standalone Docker Environment

This is a **self-contained Docker environment** for the Speaker-Profiles AI system that can be used independently without needing to clone the entire repository.

## 🚀 Quick Start (Standalone)

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (20.10.0 or later)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0.0 or later)
- For GPU support: [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (nvidia-container-toolkit)

### 1. Download This Docker Folder

You can either:

**Option A: Clone just this folder**
```bash
# Using sparse-checkout to get only the docker folder
git clone --filter=blob:none --sparse https://github.com/JustinTheGreat/Speaker-Profiles.git
cd Speaker-Profiles
git sparse-checkout set docker
cd docker
```

**Option B: Download as ZIP**
1. Go to the [GitHub repository](https://github.com/JustinTheGreat/Speaker-Profiles)
2. Navigate to the `docker` folder
3. Download all files in this folder to a local directory

### 2. Setup Environment
```bash
# Copy the environment template
cp .env.template .env

# Edit the .env file with your HuggingFace token
# Get your token from: https://huggingface.co/settings/tokens
nano .env  # or use your preferred editor
```

Edit `.env` and replace:
```
HUGGING_FACE_ACCESS_TOKEN=your_hugging_face_token_here
```

### 3. Create Required Directories
```bash
# Create directories for data persistence
mkdir -p speakers output transcription_output pretrained_models audio_files
```

### 4. Run the Environment

**For GPU Systems:**
```bash
# Start the container
docker-compose up

# Or run in background
docker-compose up -d

# Or use the convenience script
./run.sh
```

**For CPU-only Systems:**
```bash
# Use the CPU-only profile
docker-compose --profile cpu-only up

# Or use convenience script
./run.sh --cpu-only
```

## 📁 Directory Structure (After Setup)

```
docker/
├── .env                          # Your environment variables
├── .env.template                 # Template for environment setup
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # GPU Docker image
├── Dockerfile.cpu               # CPU Docker image
├── requirements.txt             # Python dependencies
├── requirements-docker.txt      # Minimal Docker dependencies
├── build.sh                     # Build script
├── run.sh                      # Quick run script
├── README-STANDALONE.md        # This file
├── TROUBLESHOOTING.md          # Troubleshooting guide
├── *.py                        # Python source files
├── audio_files/                # Put your audio files here
├── speakers/                   # Speaker profiles (auto-created)
├── output/                     # Processing results
├── transcription_output/       # Transcription results
└── pretrained_models/         # Cached AI models
```

## 🎵 Processing Audio Files

### Basic Usage
```bash
# Process a single audio file
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py "/app/audio_files/your_audio.wav"

# Run interactive example
docker-compose exec speaker-profiles python simple_auto_tagging_example.py

# Transcribe with speaker identification
docker-compose exec speaker-profiles python speaker_transcription_system.py "/app/audio_files/your_audio.wav"
```

### Using the Run Script
```bash
# Process an audio file using the convenience script
./run.sh process /app/audio_files/meeting.wav

# Open shell for manual commands
./run.sh shell

# View logs
./run.sh logs

# Stop the environment
./run.sh down
```

## 🔧 Building Custom Images

### Build Locally
```bash
# Build both GPU and CPU versions
./build.sh

# Build only GPU version
./build.sh --gpu-only

# Build only CPU version
./build.sh --cpu-only

# Build with custom tag
./build.sh --tag my-speaker-profiles --push
```

### Manual Docker Commands
```bash
# Build GPU version
docker build -t speaker-profiles:latest -f Dockerfile .

# Build CPU version
docker build -t speaker-profiles:cpu -f Dockerfile.cpu .
```

## 📋 Available Commands

### Speaker Management
```bash
# List all known speakers
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py --list-speakers

# Verify speaker database integrity
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py --verify-folders

# Migrate speaker data (if needed)
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py --migrate
```

### System Information
```bash
# Check GPU availability
docker-compose exec speaker-profiles python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test all dependencies
docker-compose exec speaker-profiles python -c "import torch, speechbrain, whisper, pyannote.audio; print('All imports successful')"

# Check container status
./run.sh status
```

## 🚨 Troubleshooting

### Common Issues

**1. HuggingFace Token Issues**
```bash
# Verify your token is set correctly
docker-compose exec speaker-profiles python -c "import os; print('HF Token:', os.getenv('HUGGING_FACE_ACCESS_TOKEN', 'NOT SET'))"
```

**2. Permission Errors**
```bash
# Fix ownership of generated files (Linux/Mac)
sudo chown -R $USER:$USER speakers/ output/ transcription_output/

# On Windows, ensure Docker has access to the mounted directories
```

**3. GPU Not Available**
```bash
# Test GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Use CPU version if GPU issues persist
./run.sh --cpu-only
```

**4. Out of Memory**
```bash
# Use CPU version for lower memory usage
docker-compose --profile cpu-only up

# Or increase Docker memory limits in Docker Desktop settings
```

**5. Audio Format Issues**
```bash
# Convert audio to supported format
docker-compose exec speaker-profiles ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
```

## 🔄 Updates and Maintenance

### Updating the Environment
```bash
# Pull latest images (if using pre-built)
docker-compose pull

# Rebuild local images
./build.sh

# Clean up old images
docker system prune -f
```

### Backing Up Data
```bash
# Create backup of speaker profiles
tar -czf speakers_backup_$(date +%Y%m%d).tar.gz speakers/

# Backup entire data directory
tar -czf speaker_data_backup_$(date +%Y%m%d).tar.gz speakers/ output/ transcription_output/
```

## 🌍 Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGING_FACE_ACCESS_TOKEN` | ✅ Yes | - | HuggingFace API token |
| `TRANSFORMERS_CACHE` | No | `/app/pretrained_models/transformers` | Transformers cache directory |
| `HF_HOME` | No | `/app/pretrained_models/huggingface` | HuggingFace cache directory |
| `SPEECHBRAIN_CACHE` | No | `/app/pretrained_models/speechbrain` | SpeechBrain cache directory |
| `CUDA_VISIBLE_DEVICES` | No | (all) | GPU devices to use (CPU: "") |

## 🐋 Using Pre-built Images

Instead of building locally, you can use pre-built images from GitHub Container Registry:

```bash
# Pull pre-built images
docker pull ghcr.io/justinthegreat/speaker-profiles:latest
docker pull ghcr.io/justinthegreat/speaker-profiles:latest-cpu

# Update docker-compose.yml to use pre-built images
# Replace:
#   build:
#     context: .
#     dockerfile: Dockerfile
# With:
#   image: ghcr.io/justinthegreat/speaker-profiles:latest
```

## 📊 Performance Tips

### For Better Performance:
- Use GPU version when available
- Mount SSD storage for `pretrained_models/` cache
- Allocate sufficient memory to Docker (8GB+ recommended)
- Use local audio files (avoid network storage for processing)

### For Lower Resource Usage:
- Use CPU version: `./run.sh --cpu-only`
- Limit Docker memory in Docker Desktop settings
- Process smaller audio files in batches

## 📝 License

This Docker environment and the Speaker-Profiles system are licensed under the MIT License.

## 🆘 Getting Help

1. **Check logs**: `./run.sh logs`
2. **Troubleshooting guide**: See `TROUBLESHOOTING.md`
3. **GitHub Issues**: [Report bugs or request features](https://github.com/JustinTheGreat/Speaker-Profiles/issues)
4. **Documentation**: See the main repository README for detailed information about the AI system

---

**Note**: This is a standalone Docker environment. For development or full repository access, clone the complete [Speaker-Profiles repository](https://github.com/JustinTheGreat/Speaker-Profiles).