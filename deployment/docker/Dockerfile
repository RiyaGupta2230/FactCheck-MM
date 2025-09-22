# Production Dockerfile for FactCheck-MM API
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN groupadd -r factcheck && useradd -r -g factcheck factcheck

WORKDIR /app

# Copy requirements and install dependencies
COPY deployment/docker/requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application code
COPY . .
RUN pip install -e .

# Create necessary directories
RUN mkdir -p logs data/cache checkpoints && \
    chown -R factcheck:factcheck /app

USER factcheck

EXPOSE 8000

ENV PYTHONPATH="/app:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "deployment.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
