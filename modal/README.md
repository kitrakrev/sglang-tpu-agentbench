# GPU baseline on Modal (A100-80GB / H100 SXM)

`gpu_bench.py` runs upstream SGLang (`lmsysorg/sglang:v0.5.2-cu126` image) and the agentbench client in the
**same container**, so the client talks to `localhost` and network latency does not enter TTFT.
It replays the TPU v6e-1 workload exactly: Qwen/Qwen2.5-7B-Instruct bf16, TP=1, 4 sessions per concurrency
level x 6 turns, 600-word shared system prompt, 150-word tool results, 128 max output tokens, seed 42,
temperature 0, concurrency 1,4,8,16,32,64,96, radix cache on then off (`--disable-radix-cache`).
Server flags: `--mem-fraction-static 0.85 --chunked-prefill-size 2048 --max-running-requests 128 --enable-cache-report`.
Page size is the SGLang-GPU default (1); TPU used 128 (its kernels need it).

Weights are downloaded once by a CPU function into the Modal Volume `agentbench-hf-cache` (public HF repo, no token).
Nothing heavy touches your laptop: only result JSON/logs (~10 MB) come back.

## Run
```bash
pip install modal                     # client 1.4.x tested
modal token new                       # browser login, once
# optional: select a non-default profile for one command only: MODAL_PROFILE=<name> modal ...
cd sglang-tpu-agentbench
modal run modal/gpu_bench.py --gpu A100-80GB --tag a100-qwen7b                  # also downloads weights
modal run modal/gpu_bench.py --gpu H100     --tag h100-qwen7b --skip-download   # H100 pinned with "H100!" (no H200 upgrade)
# quick smoke test: add  --conc 1,8 --tag smoke
```
Results land in `results/<tag>/{cache,nocache}/c*.json`, `timing.json`, `hbm_*.txt` (nvidia-smi), `logs/server-*.log`.

## Analyze (local, needs matplotlib)
Prices: Modal on-demand, per second, checked 2026-11-18 at modal.com/pricing: A100-80GB $0.000694/s (= $2.50/h),
H100 $0.001097/s (= $3.95/h). There is no spot tier, so the "spot" column is set to the same price.
```bash
python -m agentbench.analyze results/a100-qwen7b --chips 1 --price-ondemand 2.50 --price-spot 2.50
python -m agentbench.analyze results/h100-qwen7b --chips 1 --price-ondemand 3.95 --price-spot 3.95
python scripts/roofline.py results/a100-qwen7b --mode on --hbm-bw 2.039e12 --peak 312e12 --title "Qwen2.5-7B bf16 on A100-80GB"
python scripts/roofline.py results/h100-qwen7b --mode on --hbm-bw 3.35e12  --peak 989e12 --title "Qwen2.5-7B bf16 on H100 SXM"
```
Bandwidth/peak: NVIDIA A100 datasheet (A100 80GB SXM: 2,039 GB/s, 312 TFLOPS dense bf16) and H100 datasheet
(H100 SXM: 3.35 TB/s, 989 TFLOPS dense bf16). Modal docs state all its H100s are SXM; `hbm_idle.txt` shows
`NVIDIA A100-SXM4-80GB` for the A100.

## Time and cost
Measured 2026-11-18: A100 ~18 min wall (~$0.75), H100 ~9 min (~$0.60), incl. ~1-3 min server start per mode. Functions have `timeout=90 min`; download 30 min.

## Make sure nothing is left running
```bash
modal app list            # state should be "stopped" for agentbench-gpu-baseline
modal app stop agentbench-gpu-baseline   # if anything is still "ephemeral"/"deployed"
modal volume delete agentbench-hf-cache  # optional: drop the ~15 GB weight cache (storage is billed)
```
