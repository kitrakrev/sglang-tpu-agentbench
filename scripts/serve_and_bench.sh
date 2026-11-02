#!/usr/bin/env bash
# Run ON the TPU VM: start the SGLang server, wait for it, run the sweep.
# SERVER_CMD is a placeholder until the TorchTPU/SGLang TPU launch command is confirmed.
set -euo pipefail
MODEL=${MODEL:-Qwen/Qwen3-8B}
PORT=${PORT:-30000}
SERVER_CMD=${SERVER_CMD:-"python -m sglang.launch_server --model-path $MODEL --port $PORT --device tpu --enable-cache-report"}
TAG=${TAG:-$(date +%Y%m%d-%H%M)}

$SERVER_CMD > server.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/v1/models" >/dev/null; do sleep 5; done

# warmup: first requests pay compile cost (21s p99 outlier seen on T4 without it)
python -m agentbench.run --url "http://localhost:$PORT" --model "$MODEL" --sessions 4 --turns 2 --concurrency 4 --max-tokens 16 >/dev/null

for C in 1 4 8 16 32; do
  python -m agentbench.run --url "http://localhost:$PORT" --model "$MODEL" \
    --sessions $((C * 4)) --turns 6 --concurrency "$C" \
    --out "results/$TAG/c$C.json"
done
