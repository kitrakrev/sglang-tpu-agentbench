#!/usr/bin/env bash
# Start SGLang on a TPU v6e VM. Run ON the VM.
# Usage: scripts/launch_v6e.sh <config> [extra flags]   config = v6e-1-llama8b | v6e-8-llama70b
#   BACKEND=jax (default)  -> sgl_jax (sgl-project/sglang-jax), the documented v6e path.
#   BACKEND=torchtpu       -> SGLang-TorchTPU container (lmsysorg/sglang:v0.1.0-tpu, SGLANG_PLATFORM=tpu_oot).
#                             Upstream tests on TPU7x; v6e and Llama are NOT in its support list: experimental.
#   NO_CACHE=1             -> --disable-radix-cache (prefix cache off, for A/B)
#   DRY_RUN=1              -> print the command only
# Sources: docs.sglang.io/docs/hardware-platforms/tpu, lmsys.org/blog/2026-10-20-sglang-torchtpu (checked 2026-11-18)
# Install (jax backend, once per VM):
#   git clone https://github.com/sgl-project/sglang-jax && cd sglang-jax
#   uv venv --python 3.12 && source .venv/bin/activate && uv pip install -e "python[all]"
#   huggingface-cli login   # Llama is gated: needs HF_TOKEN with access granted
set -euo pipefail
CFG=${1:?config name}; shift || true
HERE=$(cd "$(dirname "$0")/.." && pwd)
# shellcheck disable=SC1090
source "$HERE/configs/$CFG.env"
PORT=${PORT:-30000}
BACKEND=${BACKEND:-jax}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-/tmp/jit_cache}
CACHE_FLAG=""; [ "${NO_CACHE:-0}" = 1 ] && CACHE_FLAG="--disable-radix-cache"

if [ "$BACKEND" = jax ]; then
  CMD=(python3 -u -m sgl_jax.launch_server --model-path "$MODEL" --trust-remote-code
       --dist-init-addr=0.0.0.0:10011 --tp-size="$TP" --device=tpu
       --mem-fraction-static="$MEM_FRAC" --page-size="$PAGE_SIZE"
       --chunked-prefill-size="$CHUNKED_PREFILL" --max-running-requests="$MAX_RUNNING"
       --attention-backend=fa --dtype=bfloat16 --enable-cache-report
       --host 0.0.0.0 --port "$PORT" $CACHE_FLAG "$@")
else
  CMD=(sudo docker run --rm --privileged --net=host --shm-size=64g
       -e SGLANG_PLATFORM=tpu_oot -e HF_TOKEN="${HF_TOKEN:-}" lmsysorg/sglang:v0.1.0-tpu
       python -m sglang.launch_server --model-path "$MODEL" --tp-size "$TP"
       --enable-cache-report --host 0.0.0.0 --port "$PORT" $CACHE_FLAG "$@")
fi
echo "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR ${CMD[*]}"
[ "${DRY_RUN:-0}" = 1 ] && exit 0
exec "${CMD[@]}"
