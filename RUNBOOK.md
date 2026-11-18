# RUNBOOK: SGLang on TPU v6e, multi-turn agent benchmark

Project `torchtpu-sglang-kr`, zone `us-east5-b` (v6e). **Do not start until the $5 budget-kill is raised**
(it unlinks billing and kills the VM mid-run). The owner must explicitly OK a new budget, e.g. $400.

## Cost (estimates, checked 2026-11-18)
v6e on-demand is **$2.70 / chip-hour** in us-east5 ([Cloud TPU pricing](https://cloud.google.com/tpu/pricing)).
The spot price changes; assume about $0.80 to $1.20 per chip-hour, so check the console before a run.

| Run | Chips | Wall time | On-demand | Spot (est.) |
|---|---|---|---|---|
| Smoke (install + 1 sweep, 8B) | 1 | 1 h | $2.70 | ~$1 |
| v6e-1 Llama-3.1-8B, cache on + off sweeps | 1 | 2 h | $5.40 | ~$2 |
| v6e-8 Llama-3.1-70B, cache on + off sweeps | 8 | 2.5 h | $54 | ~$20 |
| **Total plan, including 1 retry of each** | | | **~$125** | **~$45** |

The boot disk and egress add less than $1. Idle-delete caps a forgotten VM at 2 h, which costs at most $43 on a v6e-8.

## 0. Preconditions (owner)
```bash
gcloud config set project torchtpu-sglang-kr     # or pass --project each time
gcloud billing projects describe torchtpu-sglang-kr   # billingEnabled: true
# Budget raised in Console > Billing > Budgets (owner only). Keep the kill function, just at the new amount.
# HF_TOKEN with granted access to meta-llama/Llama-3.1-8B-Instruct and -70B-Instruct
gcloud compute tpus accelerator-types list --zone=us-east5-b | grep v6e
```

## 1. Create the VM (v6e-1 spot)
```bash
gcloud compute tpus tpu-vm create agentbench-v6e-1 --project torchtpu-sglang-kr \
  --zone us-east5-b --accelerator-type v6e-1 --version v2-alpha-tpuv6e --spot
# 70B:  --accelerator-type v6e-8  (name agentbench-v6e-8)
```
If spot capacity is unavailable, try `us-east1-d` or `europe-west4-a`, or drop `--spot` (on-demand).

## 2. Set up and arm idle-delete (do this first, on the VM)
```bash
gcloud compute tpus tpu-vm ssh agentbench-v6e-1 --zone us-east5-b --project torchtpu-sglang-kr
git clone <repo> agentbench && cd agentbench     # or: gcloud compute tpus tpu-vm scp --recurse
nohup scripts/idle_delete.sh agentbench-v6e-1 us-east5-b > idle.log 2>&1 &
pip install -e . && git clone https://github.com/sgl-project/sglang-jax ~/sglang-jax
(cd ~/sglang-jax && uv venv --python 3.12 && . .venv/bin/activate && uv pip install -e "python[all]")
. ~/sglang-jax/.venv/bin/activate && pip install -e ~/agentbench
export HF_TOKEN=...   # never commit
python -c "from sgl_jax import check_env; check_env.check_env()"
```
The VM's service account needs `tpu.nodes.delete` for idle-delete to work. If it lacks that, the backstop is the manual delete in step 4.

## 3. Run
```bash
DRY_RUN=1 scripts/launch_v6e.sh v6e-1-llama8b     # check the command
scripts/serve_and_bench.sh v6e-1-llama8b          # cache on, then off; sweep c=1,4,8,16,32
# 70B on v6e-8: scripts/serve_and_bench.sh v6e-8-llama70b
```
If `--enable-cache-report` or `--disable-radix-cache` is rejected by sgl_jax, check `python -m sgl_jax.launch_server --help`
and change those flags in `scripts/launch_v6e.sh`. Without cache report, the TTFT-by-turn curve still shows the cache effect.

## 4. Collect, then delete (always)
```bash
# from laptop
gcloud compute tpus tpu-vm scp --recurse agentbench-v6e-1:~/agentbench/results ./results \
  --zone us-east5-b --project torchtpu-sglang-kr
gcloud compute tpus tpu-vm scp agentbench-v6e-1:~/agentbench/server-*.log ./results/ --zone us-east5-b --project torchtpu-sglang-kr
gcloud compute tpus tpu-vm delete agentbench-v6e-1 --zone us-east5-b --project torchtpu-sglang-kr --quiet
gcloud compute tpus tpu-vm list --zone us-east5-b --project torchtpu-sglang-kr   # must be empty
python -m agentbench.sweep --report results/<TAG>/cache
```
