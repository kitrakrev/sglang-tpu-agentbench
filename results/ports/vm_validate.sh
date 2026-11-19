#!/usr/bin/env bash
# Run ON a TPU v6e-1 VM. Validates the model ports in kitrakrev/sglang-jax@ports-validate:
# real-checkpoint greedy parity vs HF (fp32, host CPU) + short agentbench sweep (text models).
# Usage: bash vm_validate.sh [model-key ...]   keys: gemma3-1b gemma3-4b glm4 internvl3 qwen2-audio
set -uo pipefail
cd ~
OUT=~/ports-results; mkdir -p "$OUT"
source ~/sglang-jax/.venv/bin/activate
export JAX_COMPILATION_CACHE_DIR=/tmp/jit_cache
PORT=30000

declare -A MODEL KIND EXTRA BENCH
MODEL[gemma3-1b]=unsloth/gemma-3-1b-it;     KIND[gemma3-1b]=text;  BENCH[gemma3-1b]=1
EXTRA[gemma3-1b]="--grammar-backend none"   # tokenizer has 1 more token than the vocab
MODEL[gemma3-4b]=unsloth/gemma-3-4b-it;     KIND[gemma3-4b]=text;  BENCH[gemma3-4b]=1
EXTRA[gemma3-4b]='--json-model-override-args {"architectures":["Gemma3ForCausalLM"]}'
MODEL[glm4]=zai-org/GLM-4-9B-0414;         KIND[glm4]=text;       BENCH[glm4]=1; EXTRA[glm4]=""
MODEL[internvl3]=OpenGVLab/InternVL3-2B-hf; KIND[internvl3]=vlm;  BENCH[internvl3]=0; EXTRA[internvl3]="--grammar-backend none"
MODEL[qwen2-audio]=Qwen/Qwen2-Audio-7B-Instruct; KIND[qwen2-audio]=audio; BENCH[qwen2-audio]=0; EXTRA[qwen2-audio]=""

KEYS=("$@"); [ ${#KEYS[@]} -eq 0 ] && KEYS=(gemma3-1b gemma3-4b glm4 internvl3 qwen2-audio)

for K in "${KEYS[@]}"; do
  M=${MODEL[$K]}; D="$OUT/$K"; mkdir -p "$D"
  echo "=== $K ($M) $(date -u +%T)"
  hf download "$M" >/dev/null 2>&1 || huggingface-cli download "$M" >/dev/null
  # shellcheck disable=SC2086
  python3 -u -m sgl_jax.launch_server --model-path "$M" --trust-remote-code --tp-size=1 --device=tpu \
    --mem-fraction-static=0.8 --page-size=128 --chunked-prefill-size=2048 --max-running-requests=64 \
    --attention-backend=fa --dtype=bfloat16 --enable-cache-report --host 0.0.0.0 --port $PORT \
    ${EXTRA[$K]} > "$D/server.log" 2>&1 &
  PID=$!; T0=$(date +%s)
  until curl -sf localhost:$PORT/health >/dev/null; do
    kill -0 $PID 2>/dev/null || { echo "SERVER DIED"; tail -40 "$D/server.log"; continue 2; }; sleep 10
  done
  echo "{\"startup_s\": $(( $(date +%s) - T0 ))}" > "$D/timing.json"
  # warm compile for the parity request shapes
  python3 ~/parity/tpu_parity.py "$M" "${KIND[$K]}" "$D/warm.json" --new 4 >/dev/null 2>&1 || true
  if [ "${BENCH[$K]}" = 1 ]; then
    (cd ~/sglang-tpu-agentbench && python -m agentbench.run --url http://localhost:$PORT --model "$M" \
       --sessions 8 --turns 3 --concurrency 8 --max-tokens 16 >/dev/null 2>&1;
     python -m agentbench.sweep --url http://localhost:$PORT --model "$M" --concurrency 1,4,8,16 \
       --turns 6 --sessions-per-conc 4 --out-dir "$D/bench" > "$D/bench.log" 2>&1) || echo "bench failed"
  fi
  # server answers first (TPU), then HF reference on CPU
  python3 ~/parity/tpu_parity.py "$M" "${KIND[$K]}" "$D/parity.json" > "$D/parity.log" 2>&1 \
    || { echo "parity failed"; tail -20 "$D/parity.log"; }
  tail -12 "$D/parity.log"
  kill $PID; wait $PID 2>/dev/null; sleep 5
done
echo "ALL DONE $(date -u +%T)"
