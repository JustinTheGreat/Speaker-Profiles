# Speaker-Profiles Docker Setup 🐳

This document provides comprehensive instructions for running the Speaker-Profiles system using Docker and Docker Compose.

## Quick Start

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/) (20.10.0 or later)
- [Docker Compose](https://docs.docker.com/compose/install/) (v2.0.0 or later)
- For GPU support: [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker) (nvidia-container-toolkit)

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/JustinTheGreat/Speaker-Profiles.git
cd Speaker-Profiles

# Create your environment file
cp .env.template .env

# Navigate to docker directory
cd docker
```

### 2. Configure Environment

Edit `.env` file with your credentials:

```bash
# Required: Get this from https://huggingface.co/settings/tokens
HUGGING_FACE_ACCESS_TOKEN=your_actual_token_here
```

### 3. Create Required Directories

```bash
# Create directories for audio files and persistent data (from project root)
cd ..
mkdir -p audio_files speakers output transcription_output pretrained_models
cd docker
```

### 4. Run with Docker Compose

**For GPU Systems:**
```bash
# Start the container (from docker/ directory)
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Or use convenience script
./run.sh
```

**For CPU-only Systems:**
```bash
# Use the CPU-only profile
docker-compose --profile cpu-only up

# Or use convenience script
./run.sh --cpu-only
```

## Using Pre-built Images from GitHub Container Registry

Instead of building locally, you can use our pre-built images:

```bash
# Pull the latest GPU image
docker pull ghcr.io/yourusername/speaker-profiles:latest

# Pull the latest CPU image
docker pull ghcr.io/yourusername/speaker-profiles:latest-cpu

# Update docker-compose.yml to use pre-built images
# Replace 'build: .' with 'image: ghcr.io/yourusername/speaker-profiles:latest'
```

## Running Commands

### Basic Usage

```bash
# Process a single audio file
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py "/app/audio_files/your_audio.wav"

# Run interactive example
docker-compose exec speaker-profiles python simple_auto_tagging_example.py

# Transcribe with speaker identification
docker-compose exec speaker-profiles python speaker_transcription_system.py "/app/audio_files/your_audio.wav"
```

### Using Docker Run Directly

```bash
# Process audio file with GPU
docker run --rm --gpus all \
  -v $(pwd)/audio_files:/app/audio_files:ro \
  -v $(pwd)/speakers:/app/speakers \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/.env:/app/.env:ro \
  ghcr.io/yourusername/speaker-profiles:latest \
  python auto_speaker_tagging_system.py "/app/audio_files/your_audio.wav"

# CPU-only version
docker run --rm \
  -v $(pwd)/audio_files:/app/audio_files:ro \
  -v $(pwd)/speakers:/app/speakers \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/.env:/app/.env:ro \
  ghcr.io/yourusername/speaker-profiles:latest-cpu \
  python auto_speaker_tagging_system.py "/app/audio_files/your_audio.wav"
```

## Volume Mounts Explained

| Host Directory | Container Directory | Purpose |
|---------------|-------------------|---------|
| `./speakers` | `/app/speakers` | Persistent speaker profiles database |
| `./audio_files` | `/app/audio_files` | Input audio files (read-only) |
| `./output` | `/app/output` | Speaker identification results |
| `./transcription_output` | `/app/transcription_output` | Transcription results |
| `./pretrained_models` | `/app/pretrained_models` | Downloaded AI models cache |
| `./.env` | `/app/.env` | Environment variables |

## Directory Structure

After setup, your directory should look like:

```
Speaker-Profiles/
├── .env                          # Your environment variables
├── docker-compose.yml            # Docker Compose configuration
├── Dockerfile                    # GPU Docker image
├── Dockerfile.cpu               # CPU Docker image
├── audio_files/                 # Put your audio files here
│   ├── meeting1.wav
│   └── interview.mp3
├── speakers/                    # Speaker profiles (auto-created)
│   ├── John_Doe/
│   │   ├── John_Doe.pkl
│   │   └── John_Doe_info.json
│   └── Speaker_001/
├── output/                      # Processing results
├── transcription_output/        # Transcription results
└── pretrained_models/          # Cached AI models
```

## Common Commands

### Speaker Database Management

```bash
# List all known speakers
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py --list-speakers

# Verify speaker database integrity
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py --verify-folders

# Migrate old speaker structure (if needed)
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py --migrate-dry-run
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py --migrate
```

### Batch Processing

```bash
# Process multiple files
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py \
  "/app/audio_files/file1.wav" "/app/audio_files/file2.wav" --batch

# Save results to files
docker-compose exec speaker-profiles python auto_speaker_tagging_system.py \
  "/app/audio_files/meeting.wav" --save-results
```

### Interactive Development

```bash
# Enter container for development
docker-compose exec speaker-profiles bash

# Install additional packages (temporary)
docker-compose exec speaker-profiles pip install jupyter

# Run Jupyter notebook (if installed)
docker-compose exec speaker-profiles jupyter lab --ip=0.0.0.0 --port=8000 --no-browser --allow-root
```

## GPU Support

### Setup NVIDIA Docker

```bash
# Install nvidia-container-toolkit (Ubuntu/Debian)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Verify GPU Access

```bash
# Test GPU availability in container
docker-compose exec speaker-profiles python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

## Troubleshooting

### Common Issues

1. **Permission Errors**
   ```bash
   # Fix ownership of generated files
   sudo chown -R $USER:$USER speakers/ output/ transcription_output/
   ```

2. **Out of Memory**
   ```bash
   # Use CPU version or add swap
   docker-compose --profile cpu-only up
   ```

3. **Model Download Fails**
   ```bash
   # Check your .env file and internet connection
   docker-compose exec speaker-profiles python -c "import os; print('HF Token:', os.getenv('HUGGING_FACE_ACCESS_TOKEN', 'NOT SET'))"
   ```

4. **Audio Format Issues**
   ```bash
   # Convert audio format using ffmpeg
   docker-compose exec speaker-profiles ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav
   ```

### Logs and Debugging

```bash
# View container logs
docker-compose logs speaker-profiles

# Follow logs in real-time
docker-compose logs -f speaker-profiles

# Debug container startup
docker-compose up --no-deps speaker-profiles

# Check container health
docker-compose exec speaker-profiles python -c "import torch, speechbrain, whisper, pyannote.audio; print('All imports successful')"
```

## Building Custom Images

### Build Locally

```bash
# Build GPU version
docker build -t speaker-profiles:local .

# Build CPU version
docker build -t speaker-profiles:local-cpu -f Dockerfile.cpu .

# Build with custom base image
docker build --build-arg BASE_IMAGE=python:3.9 -t speaker-profiles:custom .
```

### Development Build

```bash
# Build with development tools
docker build -t speaker-profiles:dev --target development .

# Mount source code for live editing
docker-compose -f docker-compose.dev.yml up
```

## Production Deployment

### Using Docker Swarm

```bash
# Deploy as a service
docker stack deploy -c docker-compose.yml speaker-profiles-stack
```

### Using Kubernetes

```yaml
# Create deployment.yaml for Kubernetes
apiVersion: apps/v1
kind: Deployment
metadata:
  name: speaker-profiles
spec:
  replicas: 1
  selector:
    matchLabels:
      app: speaker-profiles
  template:
    metadata:
      labels:
        app: speaker-profiles
    spec:
      containers:
      - name: speaker-profiles
        image: ghcr.io/yourusername/speaker-profiles:latest
        env:
        - name: HUGGING_FACE_ACCESS_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token
              key: token
        volumeMounts:
        - name: speaker-data
          mountPath: /app/speakers
        - name: audio-files
          mountPath: /app/audio_files
      volumes:
      - name: speaker-data
        persistentVolumeClaim:
          claimName: speaker-data-pvc
      - name: audio-files
        hostPath:
          path: /path/to/audio/files
```

## Environment Variables Reference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGING_FACE_ACCESS_TOKEN` | ✅ Yes | - | HuggingFace API token for model access |
| `TRANSFORMERS_CACHE` | No | `/app/pretrained_models/transformers` | Transformers model cache directory |
| `HF_HOME` | No | `/app/pretrained_models/huggingface` | HuggingFace cache directory |
| `SPEECHBRAIN_CACHE` | No | `/app/pretrained_models/speechbrain` | SpeechBrain model cache directory |
| `CUDA_VISIBLE_DEVICES` | No | (all) | GPU devices to use (CPU: "") |

## Support and Resources

- **GitHub Issues**: Report bugs and request features
- **Documentation**: See `WARP.md` for development guidance
- **Docker Hub**: Pre-built images available
- **GitHub Container Registry**: `ghcr.io/yourusername/speaker-profiles`

## License

This project is licensed under the MIT License - see the LICENSE file for details.