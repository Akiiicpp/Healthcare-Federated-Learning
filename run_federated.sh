#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# Defaults
ROUNDS=2
CLIENTS=3
LOCAL_EPOCHS=1
NUM_SAMPLES=600
BATCH_SIZE=32

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds)
      ROUNDS="$2"; shift 2 ;;
    --clients)
      CLIENTS="$2"; shift 2 ;;
    --local-epochs)
      LOCAL_EPOCHS="$2"; shift 2 ;;
    --num-samples)
      NUM_SAMPLES="$2"; shift 2 ;;
    --batch-size)
      BATCH_SIZE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Start server in background with flags
python -m coordinator.server --rounds "$ROUNDS" --min-clients "$CLIENTS" --local-epochs "$LOCAL_EPOCHS" &
SERVER_PID=$!
sleep 2

# Start N clients
CLIENT_PIDS=()
for ((i=0; i<CLIENTS; i++)); do
  CLIENT_ID=$i python -m hospital_client.client --num-samples "$NUM_SAMPLES" --batch-size "$BATCH_SIZE" --seed 0 &
  CLIENT_PIDS+=("$!")
  sleep 1
done

# Wait for server to finish
wait $SERVER_PID || true

# Cleanup clients
for pid in "${CLIENT_PIDS[@]}"; do
  kill "$pid" >/dev/null 2>&1 || true
done
