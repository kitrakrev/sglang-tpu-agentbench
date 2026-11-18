#!/usr/bin/env bash
# Run B (2026-11-18): v6e-4 Qwen3.8-27B-pi-adapted (merged), TP=4, cache on vs off. Run ON the VM.
# Hybrid linear-attention model: sgl_jax needs --enable-unified-radix-tree (+ --enable-recurrent-extra-buffer
# to keep page 128) for cache on; mem-fraction 0.85 OOMs at compile, 0.7 + chunked prefill 1024 works.
# Same memory settings for cache off so the A/B is fair.
set -uo pipefail
cd "$(dirname "$0")/.."
export TAG=${TAG:-v6e-4-qwen38-27b-pi} CONC=${CONC:-1,4,8,16,32,64}
M="--mem-fraction-static=0.7 --chunked-prefill-size=1024"
MODES=cache EXTRA="--enable-unified-radix-tree --enable-recurrent-extra-buffer $M" scripts/serve_and_bench.sh v6e-4-qwen38-27b-pi
MODES=nocache EXTRA="$M" scripts/serve_and_bench.sh v6e-4-qwen38-27b-pi
echo RUNB_DONE
