# 🔧 Troubleshooting Guide - Speaker-Profiles Docker

This guide covers common issues and solutions when using the Speaker-Profiles Docker environment.

## 📋 Quick Diagnostics

### System Check Commands

```bash
# Check Docker installation
docker --version
docker-compose --version

# Check Docker daemon
docker info

# Check available images  
docker images speaker-profiles

# Check running containers
docker ps --filter "name=speaker-profiles"

# Check container logs
docker-compose logs speaker-profiles-gpu
```

## 🚨 Common Issues

### 1. Docker Not Running

#### Symptoms:
- `Cannot connect to the Docker daemon`
- `docker: command not found`

#### Solutions:

**Windows:**
```powershell
# Start Docker Desktop
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"

# Wait for Docker to start, then verify
docker --version
```

**Linux:**
```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (requires logout/login)
sudo usermod -aG docker $USER
```

**macOS:**
```bash
# Start Docker Desktop from Applications
open -a Docker

# Or via command line
/Applications/Docker.app/Contents/MacOS/Docker --daemon
```

### 2. GPU Support Issues

#### Symptoms:
- `NVIDIA-SMI has failed`
- `RuntimeError: No CUDA GPUs are available`
- Container uses CPU instead of GPU

#### Diagnostics:

```bash
# Test NVIDIA Docker support
docker run --gpus all nvidia/cuda:11.8-runtime-ubuntu20.04 nvidia-smi

# Check if NVIDIA Container Toolkit is installed
which nvidia-container-runtime
```

#### Solutions:

**Install NVIDIA Container Toolkit:**

Ubuntu/Debian:
```bash
# Add NVIDIA package repository
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install nvidia-docker2
sudo apt-get update
sudo apt-get install -y nvidia-docker2

# Restart Docker
sudo systemctl restart docker
```

**Windows with WSL2:**
```powershell
# Ensure you have:
# 1. NVIDIA GPU drivers for Windows
# 2. WSL2 with NVIDIA support
# 3. Docker Desktop with WSL2 backend

# Test in WSL2
wsl nvidia-smi
```

### 3. HuggingFace Token Issues

#### Symptoms:
- `Please set a valid HUGGING_FACE_ACCESS_TOKEN`
- `401 Unauthorized` when downloading models
- `Repository not found` errors

#### Solutions:

```bash
# Check if token is set
grep HUGGING_FACE_ACCESS_TOKEN .env.docker

# Get token from HuggingFace
echo "Visit: https://huggingface.co/settings/tokens"

# Set token properly (replace YOUR_TOKEN)
sed -i 's/your_hugging_face_token_here/YOUR_ACTUAL_TOKEN/' .env.docker

# Verify token format (should be hf_...)
# Valid format: hf_xxxxxxxxxxxxxxxxxxxxxxxxx
```

### 4. Model Download Failures

#### Symptoms:
- `Failed to download model`
- Connection timeouts
- SSL certificate errors

#### Solutions:

```bash
# Test network connectivity
docker run speaker-profiles:gpu ping -c 3 huggingface.co

# Clear corrupted model cache
docker volume rm speaker-profiles_speaker-models-cache
docker volume rm speaker-profiles_speaker-models-cache-cpu

# Download models manually (if needed)
docker-compose run speaker-profiles-gpu python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('speechbrain/spkrec-ecapa-voxceleb')
print('Model downloaded successfully')
"
```

### 5. Memory Issues

#### Symptoms:
- `CUDA out of memory`
- `Killed` processes
- Container exits unexpectedly

#### Diagnostics:

```bash
# Monitor container resources
docker stats speaker-profiles-gpu

# Check system memory
free -h  # Linux
Get-WmiObject -Class Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory  # Windows
```

#### Solutions:

```bash
# Use CPU version for lower memory usage
docker-compose up speaker-profiles-cpu

# Reduce batch size (edit Python files)
# Process smaller audio files
# Add swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 6. Permission Issues (Linux/macOS)

#### Symptoms:
- `Permission denied` when writing to volumes
- Files owned by root in mounted directories

#### Solutions:

```bash
# Fix directory ownership
sudo chown -R $(whoami):$(whoami) ../speakers ../output ../audio

# Run with proper user mapping
docker-compose run --user $(id -u):$(id -g) speaker-profiles-gpu bash

# Set proper permissions
chmod -R 755 ../speakers ../output ../audio
```

### 7. Audio File Issues

#### Symptoms:
- `FileNotFoundError` for audio files
- `Unsupported audio format`
- Audio processing failures

#### Solutions:

```bash
# Check audio file format
file ../audio/your_file.wav

# Convert audio to supported format
ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav

# Test with sample audio
docker-compose run speaker-profiles-gpu python -c "
import torchaudio
waveform, sample_rate = torchaudio.load('/app/audio/test.wav')
print(f'Audio loaded: {waveform.shape} at {sample_rate}Hz')
"
```

### 8. Build Failures

#### Symptoms:
- Package installation failures
- Docker build errors
- Missing dependencies

#### Solutions:

```bash
# Clean build without cache
./build.sh --no-cache

# Check Docker Hub connectivity
docker pull python:3.9-slim-bullseye

# Free up disk space
docker system prune -a

# Check build logs
docker build --progress=plain -f Dockerfile -t speaker-profiles:debug .
```

## 🔍 Advanced Debugging

### Container Inspection

```bash
# Inspect running container
docker exec -it speaker-profiles-gpu bash

# Inside container - check Python environment
python -c "
import torch
import speechbrain
import pyannote.audio
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

# Check environment variables
env | grep -E "(HF|TRANSFORMERS|CUDA|PYTHON)"

# Check mounted volumes
ls -la /app/audio /app/speakers /app/output
```

### Log Analysis

```bash
# Container logs with timestamps
docker-compose logs -t speaker-profiles-gpu

# Follow logs in real-time
docker-compose logs -f speaker-profiles-gpu

# Python error debugging
docker-compose run speaker-profiles-gpu python -u -c "
import traceback
try:
    from auto_speaker_tagging_system import AutoSpeakerTaggingSystem
    system = AutoSpeakerTaggingSystem()
    print('System initialized successfully')
except Exception as e:
    traceback.print_exc()
"
```

### Performance Monitoring

```bash
# Container resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# GPU utilization (if available)
docker exec speaker-profiles-gpu nvidia-smi

# Disk usage in container
docker exec speaker-profiles-gpu df -h
```

## 🛠 Environment-Specific Issues

### Windows Specific

#### Windows Path Issues
```powershell
# Use proper Windows paths in docker-compose
# Convert paths if needed
$windowsPath = "C:\Users\marcu\Documents\GitHub\Speaker-Profiles"
$dockerPath = $windowsPath -replace '\\', '/' -replace 'C:', '/c'
```

#### WSL2 Issues
```powershell
# Check WSL2 version
wsl --list --verbose

# Update WSL2 if needed
wsl --update

# Set WSL2 as default
wsl --set-default-version 2
```

### Linux Specific

#### SELinux Issues
```bash
# Check SELinux status
getenforce

# Allow Docker volumes (if SELinux is enforcing)
sudo setsebool -P container_manage_cgroup on
```

#### Firewall Issues
```bash
# Check if Docker ports are blocked
sudo ufw status

# Allow Docker ports if needed
sudo ufw allow 8000:8002/tcp
```

### macOS Specific

#### Docker Desktop Resource Limits
```bash
# Check Docker Desktop settings
# Increase CPU/Memory limits in Docker Desktop preferences

# File sharing permissions
# Ensure project directory is shared in Docker Desktop settings
```

## 📞 Getting Help

### Collecting Debug Information

Before seeking help, collect this information:

```bash
# System info
echo "=== System Information ===" > debug_info.txt
uname -a >> debug_info.txt
docker --version >> debug_info.txt
docker-compose --version >> debug_info.txt

# Docker info
echo -e "\n=== Docker Information ===" >> debug_info.txt
docker info >> debug_info.txt

# Container status
echo -e "\n=== Container Status ===" >> debug_info.txt
docker ps -a --filter "name=speaker-profiles" >> debug_info.txt

# Recent logs
echo -e "\n=== Recent Logs ===" >> debug_info.txt
docker-compose logs --tail=50 speaker-profiles-gpu >> debug_info.txt

# Environment
echo -e "\n=== Environment ===" >> debug_info.txt
cat .env.docker | grep -v "HUGGING_FACE_ACCESS_TOKEN" >> debug_info.txt
echo "HUGGING_FACE_ACCESS_TOKEN=***REDACTED***" >> debug_info.txt
```

### Community Resources

- **Project Issues**: GitHub repository issues
- **Docker Forums**: [Docker Community](https://forums.docker.com/)
- **Stack Overflow**: Tag your questions with `docker`, `pytorch`, `speechbrain`

### Emergency Recovery

If everything fails:

```bash
# Complete cleanup and restart
docker-compose down
docker system prune -a -f
docker volume prune -f

# Remove all related images
docker rmi $(docker images speaker-profiles -q)

# Start fresh
./build.sh --no-cache
./run.ps1 -Setup
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [NVIDIA Container Toolkit Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
- [PyTorch Docker Guide](https://pytorch.org/get-started/locally/#docker-image)
- [HuggingFace Hub Documentation](https://huggingface.co/docs/hub/index)

---

💡 **Remember**: Most issues are resolved by ensuring Docker is properly installed, the HuggingFace token is correct, and sufficient system resources are available.