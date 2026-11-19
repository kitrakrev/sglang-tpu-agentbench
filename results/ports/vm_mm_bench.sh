#!/usr/bin/env bash
# Run ON a TPU v6e-1 VM: multimodal throughput for the InternVL3 / Qwen2-Audio ports.
set -uo pipefail
source ~/sglang-jax/.venv/bin/activate
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
OUT=~/ports-results; PORT=30000

run() {  # key model kind conc extra...
  local K=$1 M=$2 KIND=$3 CONC=$4; shift 4
  local D="$OUT/$K-mmbench"; mkdir -p "$D"
  echo "=== $K $(date -u +%T)"
  python3 -u -m sgl_jax.launch_server --model-path "$M" --trust-remote-code --tp-size=1 --device=tpu \
    --mem-fraction-static=0.8 --page-size=128 --chunked-prefill-size=4096 --max-running-requests=64 \
    --attention-backend=fa --dtype=bfloat16 --enable-cache-report --host 0.0.0.0 --port $PORT \
    "$@" > "$D/server.log" 2>&1 &
  local PID=$!; local T0=$(date +%s)
  until curl -sf localhost:$PORT/health >/dev/null; do
    kill -0 $PID 2>/dev/null || { echo "SERVER DIED"; tail -30 "$D/server.log"; return; }; sleep 10
  done
  echo "{\"startup_s\": $(( $(date +%s) - T0 ))}" > "$D/timing.json"
  python3 ~/parity/bench_mm.py --kind "$KIND" --model "$M" --conc "$CONC" --out "$D" 2>&1 | tee "$D/client.log"
  kill $PID; wait $PID 2>/dev/null; sleep 5
}

run internvl3 OpenGVLab/InternVL3-2B-hf vlm 1,4,8,16 --grammar-backend none
run qwen2-audio Qwen/Qwen2-Audio-7B-Instruct audio 1,4,8
echo "ALL DONE $(date -u +%T)"
