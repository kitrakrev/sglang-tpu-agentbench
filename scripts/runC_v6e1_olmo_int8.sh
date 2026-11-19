#!/usr/bin/env bash
# Run C: v6e-1 add-ons. Run ON the VM inside the sgl_jax venv (sglang-jax from kitrakrev/sglang-jax@add-olmo3-support).
#  1. OLMo-2-7B-Instruct and Olmo-3-7B-Instruct (new Olmo2/Olmo3ForCausalLM ports), cache on vs off, c=1..32
#  2. Qwen2.5-7B bf16 baseline vs int8 weights (built-in int8.yaml, online quantization), cache on, c=1..32
set -uo pipefail
cd "$(dirname "$0")/.."
TAG=v6e-1-olmo2-7b CONC=1,4,8,16,32 scripts/serve_and_bench.sh v6e-1-olmo2-7b
TAG=v6e-1-olmo3-7b CONC=1,4,8,16,32 scripts/serve_and_bench.sh v6e-1-olmo3-7b
TAG=v6e-1-qwen7b-int8 MODES=cache VARIANT=bf16 CONC=1,4,8,16,32 scripts/serve_and_bench.sh v6e-1-qwen7b
TAG=v6e-1-qwen7b-int8 MODES=cache VARIANT=int8 CONC=1,4,8,16,32 \
  EXTRA="--quantization-config-path=int8.yaml" scripts/serve_and_bench.sh v6e-1-qwen7b
echo RUN_C_DONE
