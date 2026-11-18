#!/usr/bin/env bash
# Capture a JAX/XProf trace of ~20 decode steps at c=32 via sgl_jax /start_profile. Run ON the VM in the venv.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=${1:-v6e-1-qwen7b}; source configs/$CFG.env; PORT=${PORT:-30000}
TAG=${TAG:-v6e-1-qwen7b-tuning}; OUT=$PWD/results/$TAG/profile_c32; mkdir -p "$OUT"
scripts/launch_v6e.sh "$CFG" > server-profile.log 2>&1 & PID=$!
until curl -sf localhost:$PORT/health >/dev/null; do kill -0 $PID || exit 1; sleep 5; done
python -m agentbench.run --url http://localhost:$PORT --model "$MODEL" --sessions 8 --turns 3 --concurrency 8 --max-tokens 16 >/dev/null
python -m agentbench.run --url http://localhost:$PORT --model "$MODEL" --sessions 96 --turns 6 --concurrency 32 \
  --out "$OUT/c32_during_profile.json" > "$OUT/client.log" 2>&1 & B=$!
sleep 25
curl -s -X POST localhost:$PORT/start_profile -H 'Content-Type: application/json' \
  -d "{\"output_dir\": \"$OUT/trace\", \"num_steps\": 20}" | tee "$OUT/start_profile.txt"; echo
sleep 30; curl -s -X POST localhost:$PORT/stop_profile >> "$OUT/start_profile.txt" 2>&1 || true
wait $B; kill $PID; wait $PID 2>/dev/null; cp server-profile.log results/$TAG/logs/
(cd "$OUT" && tar czf trace.tgz trace && du -sh trace trace.tgz)
echo PROFILE_DONE
