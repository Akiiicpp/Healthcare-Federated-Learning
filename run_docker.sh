#!/bin/bash
set -e

echo "🏥 Healthcare Federated Learning - Docker Demo"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop."
    exit 1
fi

# Parse arguments
SCENARIO="quick"
CLIENTS=3
ROUNDS=3

while [[ $# -gt 0 ]]; do
    case $1 in
        --scenario)
            SCENARIO="$2"
            shift 2
            ;;
        --clients)
            CLIENTS="$2"
            shift 2
            ;;
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--scenario quick|large] [--clients N] [--rounds N]"
            echo "  --scenario: quick (3 clients, 3 rounds) or large (5 clients, 10 rounds)"
            echo "  --clients: number of hospital clients (3-5)"
            echo "  --rounds: number of federated rounds"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "📋 Configuration:"
echo "  Scenario: $SCENARIO"
echo "  Clients: $CLIENTS"
echo "  Rounds: $ROUNDS"
echo ""

# Build images
echo "🔨 Building Docker images..."
docker-compose build

# Start services
echo "🚀 Starting federated learning simulation..."
if [ "$SCENARIO" = "large" ]; then
    docker-compose --profile large-demo up --scale hospital-client-1=$CLIENTS
else
    docker-compose up --scale hospital-client-1=$CLIENTS
fi

echo "✅ Demo completed!"
