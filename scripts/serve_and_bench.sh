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
MODES=${MODES:-cache nocache}   # subset, e.g. MODES=cache
VARIANT=${VARIANT:-}             # output subdir prefix, e.g. VARIANT=page256 -> results/TAG/page256-cache
EXTRA=${EXTRA:-}                 # extra server flags, e.g. EXTRA="--page-size=256"
SESS=${SESS:-4}

for MODE in $MODES; do
  SUB=${VARIANT:+$VARIANT-}$MODE
  NO_CACHE=$([ $MODE = nocache ] && echo 1 || echo 0) scripts/launch_v6e.sh "$CFG" $EXTRA > "server-$SUB.log" 2>&1 &
  PID=$!; T0=$(date +%s)
  until curl -sf "http://localhost:$PORT/health" >/dev/null; do
    kill -0 $PID 2>/dev/null || { tail -50 "server-$SUB.log"; exit 1; }; sleep 10; done
  T1=$(date +%s)
  # warmup: first requests pay XLA compile
  python -m agentbench.run --url "http://localhost:$PORT" --model "$MODEL" --sessions 8 --turns 3 --concurrency 8 --max-tokens 16 >/dev/null
  T2=$(date +%s); mkdir -p "results/$TAG/$SUB"
  echo "{\"startup_s\": $((T1-T0)), \"warmup_s\": $((T2-T1))}" > "results/$TAG/$SUB/timing.json"
  (command -v tpu-info >/dev/null && tpu-info || true) > "results/$TAG/$SUB/hbm_idle.txt" 2>&1
  python -m agentbench.sweep --url "http://localhost:$PORT" --model "$MODEL" \
    --concurrency "$CONC" --turns 6 --sessions-per-conc "$SESS" --out-dir "results/$TAG/$SUB" 2>&1 | tee "results/$TAG/$SUB/client.log"
  (command -v tpu-info >/dev/null && tpu-info || true) > "results/$TAG/$SUB/hbm_after.txt" 2>&1
  kill $PID; wait $PID 2>/dev/null || true; sleep 5
  mkdir -p "results/$TAG/logs"; cp "server-$SUB.log" "results/$TAG/logs/"
done
echo "done: results/$TAG"
