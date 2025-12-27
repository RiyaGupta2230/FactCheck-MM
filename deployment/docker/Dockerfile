# ============================================
# FactCheck-MM Production Dockerfile
# Inference-Only API Server
# ============================================

# Stage 1: Base image with system dependencies
FROM python:3.10-slim as base

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build tools
    build-essential \
    cmake \
    git \
    # Audio/video processing
    ffmpeg \
    libsndfile1 \
    # Image processing
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    # Networking
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r factcheck && useradd -r -g factcheck -u 1000 factcheck

# Set working directory
WORKDIR /app

# Stage 2: Python dependencies
FROM base as dependencies

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file
COPY deployment/docker/requirements-prod.txt /tmp/requirements-prod.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements-prod.txt

# Download spaCy model (lightweight)
RUN python -m spacy download en_core_web_sm --quiet

# Download NLTK data (minimal for inference)
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# Stage 3: Application code
FROM dependencies as application

# Copy only necessary application code (NOT data, experiments, tests)
COPY --chown=factcheck:factcheck shared/ /app/shared/
COPY --chown=factcheck:factcheck config/ /app/config/
COPY --chown=factcheck:factcheck sarcasm_detection/ /app/sarcasm_detection/
COPY --chown=factcheck:factcheck paraphrasing/ /app/paraphrasing/
COPY --chown=factcheck:factcheck fact_verification/ /app/fact_verification/
COPY --chown=factcheck:factcheck deployment/ /app/deployment/

# Create necessary directories
RUN mkdir -p \
    /app/logs \
    /app/deployment/models \
    /app/cache \
    && chown -R factcheck:factcheck /app

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Switch to non-root user
USER factcheck

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables for production
ENV ENVIRONMENT=production
ENV LOG_LEVEL=info
ENV PORT=8000

# Run FastAPI application with Uvicorn
CMD ["uvicorn", "deployment.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ============================================
# GPU Variant (Optional)
# ============================================
# To use GPU, build with: docker build --target gpu -t factcheck-mm:gpu .
FROM application as gpu

USER root

# Note: CUDA runtime is typically provided by NVIDIA Docker runtime
# If needed, install CUDA-specific packages here

# Install GPU-optimized PyTorch (if not already installed)
RUN pip install --no-cache-dir \
    torch==2.1.0+cu118 \
    torchvision==0.16.0+cu118 \
    torchaudio==2.1.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install GPU-optimized FAISS
RUN pip uninstall -y faiss-cpu && \
    pip install --no-cache-dir faiss-gpu==1.7.4

USER factcheck

# GPU-specific environment variables
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"
