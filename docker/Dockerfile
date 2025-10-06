# Use Python 3.9 with CUDA support as base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Audio processing libraries
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    # Git for cloning models
    git \
    # Build tools
    build-essential \
    cmake \
    # Audio codecs
    libavcodec-extra \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    # Other utilities
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Upgrade pip and install wheel
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better Docker layer caching
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Install additional system-specific packages for audio processing
RUN pip install \
    torch-audio-tagging \
    asteroid-filterbanks \
    pesq \
    pystoi

# Create directories for the application
RUN mkdir -p /app/speakers \
    && mkdir -p /app/output \
    && mkdir -p /app/transcription_output \
    && mkdir -p /app/pretrained_models

# Copy the application code
COPY ../ .

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Create .env template file
RUN echo "# Copy this file to .env and fill in your tokens" > .env.template \
    && echo "HUGGING_FACE_ACCESS_TOKEN=your_hugging_face_token_here" >> .env.template

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