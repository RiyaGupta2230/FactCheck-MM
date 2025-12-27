# FactCheck-MM Docker Deployment

Production-ready Docker setup for inference-only API serving.

## Quick Start

### 1. Build Docker Image

Navigate to docker directory
cd deployment/docker

Build CPU version
docker-compose build

Or build GPU version
docker-compose build --build-arg target=gpu


### 2. Export Models

Before running, export trained models:

From project root
python deployment/scripts/model_export.py --export-all

Models will be saved to `deployment/models/`

### 3. Run API Server
Start API (CPU)
docker-compose up -d

Start API with Redis caching
docker-compose --profile caching up -d

Start GPU version
docker-compose --profile gpu up -d

View logs
docker-compose logs -f

Stop
docker-compose down


### 4. Test API
Health check
curl http://localhost:8000/health

API documentation
open http://localhost:8000/docs

Test sarcasm detection
curl -X POST http://localhost:8000/api/v1/sarcasm/predict
-H "Content-Type: application/json"
-d '{"text": "Oh great, another meeting!"}'

Test fact verification
curl -X POST http://localhost:8000/api/v1/fact/verify
-H "Content-Type: application/json"
-d '{"claim": "The Eiffel Tower is in Paris"}'


## Configuration

### Environment Variables

Create `.env` file in `deployment/docker/`:

Application
ENVIRONMENT=production
LOG_LEVEL=info
WORKERS=2

API
PORT=8000
MAX_REQUEST_SIZE=10485760
ALLOWED_ORIGINS=http://localhost:3000

Caching
TRANSFORMERS_CACHE=/app/cache/transformers


### GPU Support

Requirements:
- NVIDIA Docker runtime
- CUDA 11.7+
- RTX 2050 or compatible GPU

Check GPU availability
nvidia-smi

Install NVIDIA Docker runtime
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list |
sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

Run with GPU
docker-compose --profile gpu up -d


## Troubleshooting

### Models Not Found

Ensure models are exported
ls -la ../../deployment/models/

Export models if missing
python ../../deployment/scripts/model_export.py --export-all


### Out of Memory

Adjust resource limits in `docker-compose.yml`:

deploy:
resources:
limits:
memory: 8G # Increase if needed


### GPU Not Detected

Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

Check container GPU access
docker exec factcheck-mm-api python -c "import torch; print(torch.cuda.is_available())"


## Production Deployment

### With Reverse Proxy (Nginx)
server {
listen 80;
server_name api.factcheck-mm.com;
location / {
    proxy_pass http://localhost:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
}


### With Load Balancer

Scale horizontally:

docker-compose up -d --scale factcheck-mm=3


### Monitoring

Enable monitoring stack:

docker-compose --profile monitoring up -d


Access:
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## License

MIT License - See LICENSE file for details

