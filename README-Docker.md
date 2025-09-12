# Healthcare Federated Learning - Docker Setup

## Quick Start

### 1. Prerequisites
- Docker Desktop installed and running
- At least 4GB RAM available for containers

### 2. Run Quick Demo (3 clients, 3 rounds)
```bash
./run_docker.sh --scenario quick
```

### 3. Run Large Demo (5 clients, 10 rounds)
```bash
./run_docker.sh --scenario large
```

### 4. Custom Configuration
```bash
./run_docker.sh --clients 4 --rounds 5
```

## Manual Docker Commands

### Build and run all services
```bash
docker-compose up --build
```

### Scale to more clients
```bash
docker-compose up --scale hospital-client-1=5
```

### View logs
```bash
docker-compose logs -f coordinator
docker-compose logs -f hospital-client-1
```

### Stop all services
```bash
docker-compose down
```

### Clean up (remove images)
```bash
docker-compose down --rmi all
```

## Architecture

- **coordinator**: Flower server (port 8080)
- **hospital-client-1 to 5**: Simulated hospitals
- **fl-network**: Internal Docker network

## Troubleshooting

- **Port conflicts**: Change port 8080 in docker-compose.yml
- **Memory issues**: Reduce NUM_SAMPLES in environment variables
- **Slow startup**: Pre-build images with `docker-compose build`
