# Speaker-Profiles Docker Image
# Main Dockerfile with GPU support for CUDA-accelerated speaker recognition
FROM pytorch/pytorch:2.1.0-cuda11.8-devel-ubuntu20.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    sox \
    libsox-fmt-all \
    libsox-dev \
    libasound2-dev \
    portaudio19-dev \
    libportaudio2 \
    libportaudiocpp0 \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Install PyTorch with CUDA support (specific versions for stability)
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Copy requirements first for better Docker layer caching
COPY requirements-docker.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for model caching
RUN mkdir -p /app/pretrained_models/transformers && \
    mkdir -p /app/pretrained_models/huggingface && \
    mkdir -p /app/pretrained_models/speechbrain && \
    mkdir -p /app/speakers && \
    mkdir -p /app/audio && \
    mkdir -p /app/output

# Set environment variables for model caching
ENV TRANSFORMERS_CACHE=/app/pretrained_models/transformers
ENV HF_HOME=/app/pretrained_models/huggingface
ENV SPEECHBRAIN_CACHE=/app/pretrained_models/speechbrain

# Copy project files
COPY *.py /app/
COPY .env.docker /app/.env

# Create a non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Expose port for potential web interface (optional)
EXPOSE 8000

# Set default command
CMD ["python", "simple_auto_tagging_example.py"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; import speechbrain; import pyannote.audio; print('OK')" || exit 1

# Labels for better image management
LABEL maintainer="Speaker-Profiles Project"
LABEL description="GPU-enabled Docker image for automatic speaker recognition and profiling"
LABEL version="1.0"
LABEL gpu.required="true"