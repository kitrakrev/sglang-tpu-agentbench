#!/usr/bin/env bash
# Run A (2026-11-18): v6e-1 Qwen2.5-7B tuning sweeps. Run ON the VM inside the sgl_jax venv.
set -uo pipefail
cd "$(dirname "$0")/.."
export TAG=${TAG:-v6e-1-qwen7b-tuning}
CFG=v6e-1-qwen7b
# 1. high concurrency, max-running 128, cache on
MODES=cache VARIANT=hiconc CONC=64,96,128 SESS=2 EXTRA="--max-running-requests=128" scripts/serve_and_bench.sh $CFG
# 2. page size 128 vs 256 at c=16/32
for P in 128 256; do MODES=cache VARIANT=page$P CONC=16,32 EXTRA="--page-size=$P" scripts/serve_and_bench.sh $CFG; done
# 3. chunked prefill at c=32 (2048 = baseline config)
for C in 512 1024 2048 4096; do MODES=cache VARIANT=chunk$C CONC=32 EXTRA="--chunked-prefill-size=$C" scripts/serve_and_bench.sh $CFG; done
echo RUN_A_DONE
