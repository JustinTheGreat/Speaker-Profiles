# Docker Build Troubleshooting Guide

This guide helps resolve common issues when building and running Speaker-Profiles Docker containers.

## 🚫 Disk Space Issues (GitHub Actions)

### Problem
```
System.IO.IOException: No space left on device
```

### Solutions

**Option 1: Use optimized workflow (recommended)**
The repository now includes optimized workflows that:
- Free up ~14GB disk space before building
- Use minimal requirements for faster builds  
- Build only AMD64 architecture by default
- Use efficient caching strategies

**Option 2: Build locally and push**
```bash
# Build locally on your machine
cd docker
./build.sh --tag ghcr.io/yourusername/speaker-profiles --push
```

**Option 3: Manual workflow trigger**
- Go to Actions tab in GitHub
- Select "Build and Push Docker Images" 
- Click "Run workflow" 
- Select only CPU or GPU build

## 🐳 Local Docker Issues

### Problem: Build fails with "no space left on device" locally

**Solution:**
```bash
# Clean up Docker
docker system prune -af
docker volume prune -f

# Check available space
df -h

# Build with smaller image
cd docker
docker build -f Dockerfile.cpu -t speaker-profiles:cpu ..
```

### Problem: Container runs but AI models fail to load

**Check 1: Verify .env file**
```bash
cat ../.env
# Should contain: HUGGING_FACE_ACCESS_TOKEN=your_actual_token
```

**Check 2: Test HuggingFace access**
```bash
docker-compose exec speaker-profiles python -c "
import os
token = os.getenv('HUGGING_FACE_ACCESS_TOKEN')
print('Token set:' if token and token != 'your_hugging_face_token_here' else 'Token missing')
"
```

**Check 3: Download models manually**
```bash
# Enter container
docker-compose exec speaker-profiles bash

# Test model download
python -c "
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization-3.1')
print('Model loaded successfully!')
"
```

## 🔧 Performance Issues

### Problem: Slow builds

**Solution 1: Use minimal requirements**
```bash
# Use the optimized Docker requirements
cd docker
docker build -f Dockerfile.cpu --build-arg REQUIREMENTS_FILE=requirements-docker.txt ..
```

**Solution 2: Multi-stage builds**
```bash
# Build with caching
docker build --cache-from speaker-profiles:latest -t speaker-profiles:latest -f Dockerfile ..
```

### Problem: Out of memory during processing

**Solution 1: Use CPU version**
```bash
docker-compose --profile cpu-only up
```

**Solution 2: Increase Docker memory limit**
- Docker Desktop → Settings → Resources → Memory → Increase to 8GB+

**Solution 3: Process smaller files**
```bash
# Split large audio files
docker-compose exec speaker-profiles ffmpeg -i large_audio.wav -t 300 smaller_chunk.wav
```

## 🌐 Network Issues

### Problem: Can't download models or packages

**Check 1: Internet connectivity**
```bash
docker-compose exec speaker-profiles curl -I https://huggingface.co
```

**Check 2: Proxy settings (if behind corporate firewall)**
```bash
# Add to docker-compose.yml environment:
environment:
  - HTTP_PROXY=http://proxy.company.com:8080
  - HTTPS_PROXY=http://proxy.company.com:8080
```

**Check 3: DNS issues**
```bash
# Add to docker-compose.yml
dns:
  - 8.8.8.8
  - 8.8.4.4
```

## 🔍 Debugging Tips

### Enable verbose logging
```bash
# Set debug environment
export DOCKER_BUILDKIT=1
export BUILDKIT_PROGRESS=plain

# Build with verbose output
docker build --progress=plain -f docker/Dockerfile .
```

### Check container logs
```bash
# View logs
docker-compose logs -f speaker-profiles

# Debug container startup
docker-compose run --rm speaker-profiles bash
```

### Test specific components
```bash
# Test audio processing
docker-compose exec speaker-profiles python -c "
import torchaudio
print('Audio support:', torchaudio.list_audio_backends())
"

# Test GPU support
docker-compose exec speaker-profiles python -c "
import torch
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU count:', torch.cuda.device_count())
    print('GPU name:', torch.cuda.get_device_name(0))
"
```

## 🚀 Optimization Recommendations

### For Development
```bash
# Use CPU version for development
docker-compose --profile cpu-only up

# Mount source code for live editing
# (Already configured in docker-compose.yml)
```

### For Production
```bash
# Use pre-built images
docker pull ghcr.io/yourusername/speaker-profiles:latest
docker pull ghcr.io/yourusername/speaker-profiles:latest-cpu

# Pin to specific version
docker pull ghcr.io/yourusername/speaker-profiles:v1.0.0
```

### Resource Limits
```yaml
# Add to docker-compose.yml services
deploy:
  resources:
    limits:
      memory: 4G
      cpus: '2.0'
    reservations:
      memory: 2G
      cpus: '1.0'
```

## 📊 Monitoring

### Check resource usage
```bash
# Monitor container resources
docker stats

# Check disk usage
docker system df

# Check image sizes
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

### Health checks
```bash
# Manual health check
docker-compose exec speaker-profiles python -c "
import torch, speechbrain, whisper, pyannote.audio
print('✅ All dependencies working!')
"
```

## 🆘 Getting Help

1. **Check GitHub Issues**: Search for similar problems
2. **Enable Debug Mode**: Set environment variables for verbose logging  
3. **Collect Information**:
   ```bash
   # System info
   docker version
   docker-compose version
   docker system info
   
   # Container info
   docker-compose ps
   docker-compose logs speaker-profiles
   ```
4. **Create Issue**: Include system info, logs, and steps to reproduce

## 📝 Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `No space left on device` | Disk full | Clean Docker: `docker system prune -af` |
| `HUGGING_FACE_ACCESS_TOKEN` | Missing token | Set in `.env` file |
| `CUDA out of memory` | GPU memory full | Use CPU version or smaller batch sizes |
| `Connection refused` | Service not ready | Wait for container startup |
| `Permission denied` | File ownership | Fix with `sudo chown -R $USER:$USER ./` |