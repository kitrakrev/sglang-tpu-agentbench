#!/usr/bin/env bash
# Run ON the TPU VM: start SGLang (cache on, then off), sweep concurrency, write reports.
# Usage: scripts/serve_and_bench.sh v6e-1-qwen7b
set -euo pipefail
CFG=${1:-v6e-1-qwen7b}
HERE=$(cd "$(dirname "$0")/.." && pwd); cd "$HERE"
source "configs/$CFG.env"
PORT=${PORT:-30000}
TAG=${TAG:-$CFG-$(date +%Y%m%d-%H%M)}
CONC=${CONC:-1,4,8,16,32}

for MODE in cache nocache; do
  NO_CACHE=$([ $MODE = nocache ] && echo 1 || echo 0) scripts/launch_v6e.sh "$CFG" > "server-$MODE.log" 2>&1 &
  PID=$!
  until curl -sf "http://localhost:$PORT/health" >/dev/null; do
    kill -0 $PID 2>/dev/null || { tail -50 "server-$MODE.log"; exit 1; }; sleep 10; done
  # warmup: first requests pay XLA compile
  python -m agentbench.run --url "http://localhost:$PORT" --model "$MODEL" --sessions 8 --turns 3 --concurrency 8 --max-tokens 16 >/dev/null
  python -m agentbench.sweep --url "http://localhost:$PORT" --model "$MODEL" \
    --concurrency "$CONC" --turns 6 --out-dir "results/$TAG/$MODE"
  kill $PID; wait $PID 2>/dev/null || true; sleep 5
done
echo "done: results/$TAG"
