. ~/sglang-jax/.venv/bin/activate; cd ~/agentbench
export TAG=v6e-4-qwen38-27b-pi CONC=1,4,8,16,32,64
M="--mem-fraction-static=0.7 --chunked-prefill-size=1024"
MODES=cache EXTRA="--enable-unified-radix-tree --enable-recurrent-extra-buffer $M" scripts/serve_and_bench.sh v6e-4-qwen38-27b-pi
MODES=nocache EXTRA="$M" scripts/serve_and_bench.sh v6e-4-qwen38-27b-pi
echo RUNB_DONE
