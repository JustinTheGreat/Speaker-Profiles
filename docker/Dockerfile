# Multi-stage build for smaller final image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies in single layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# Set working directory
WORKDIR /app

# Copy and install minimal requirements from docker subdirectory
COPY docker/requirements-docker.txt ./
RUN pip install --no-cache-dir -r requirements-docker.txt \
    && python -c "import torch; print(f'PyTorch {torch.__version__} installed')" \
    && rm -rf /tmp/* /root/.cache/pip/*

# Copy necessary application files from root
COPY requirements.txt WARP.md ./
COPY *.py ./

# Create directories and user in single layer
RUN mkdir -p /app/{speakers,output,transcription_output,pretrained_models} \
    && groupadd -r appuser && useradd -r -g appuser appuser \
    && echo "HUGGING_FACE_ACCESS_TOKEN=your_hugging_face_token_here" > .env.template \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for potential web interface (future enhancement)
EXPOSE 8000

# Set default command
CMD ["python", "--version"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch, speechbrain, whisper, pyannote.audio; print('All dependencies working')" || exit 1

# Labels for metadata
LABEL maintainer="Speaker-Profiles"
LABEL description="Automatic Speaker Recognition and Profiling System"
LABEL version="1.0"
LABEL org.opencontainers.image.source="https://github.com/yourusername/Speaker-Profiles"
LABEL org.opencontainers.image.description="AI-powered speaker identification and transcription system"
LABEL org.opencontainers.image.licenses="MIT"

# Environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/pretrained_models/transformers
ENV HF_HOME=/app/pretrained_models/huggingface
ENV SPEECHBRAIN_CACHE=/app/pretrained_models/speechbrain