# Reproduce every result in this repo

Step-by-step, from an empty laptop to the numbers in `docs/WRITEUP_DRAFT.md`.
Everything runs on rented hardware (Cloud TPU, Modal GPU); the laptop only drives it.

| Result | Hardware | Wall time | Cost (Nov 2026) | Section |
|---|---|---|---|---|
| Harness self-test | laptop | 2 min | $0 | 1 |
| Baseline: Qwen2.5-7B cache on/off | TPU v6e-1 | ~1 h | ~$1 spot / $2.70 on-demand | 4 |
| Run A: 7B tuning + profile | TPU v6e-1 | ~30 min | ~$1 spot | 5 |
| Run B: Qwen3.8-27B-pi (LoRA merged), TP=4 | TPU v6e-4 | ~1 h | ~$9 on-demand | 6 |
| Qwen2.5-72B (not run yet) | TPU v6e-8 | ~2.5 h | ~$54 on-demand | 7 |
| GPU baseline (A100 / H100) | Modal | ~1 h | ~$5 | 9 |

## 0. Tools on the laptop
```bash
# Python 3.10+ and uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Google Cloud CLI: https://cloud.google.com/sdk/docs/install
gcloud auth login
# Hugging Face CLI (only needed for gated models; Qwen2.5 is ungated)
curl -LsSf https://hf.co/cli/install.sh | bash
hf auth login            # use a READ-only token; never commit it
# Modal CLI (GPU baseline only)
pip install modal && modal token new
```

## 1. Harness self-test (no accelerator)
```bash
git clone https://github.com/kitrakrev/sglang-tpu-agentbench && cd sglang-tpu-agentbench
uv venv && . .venv/bin/activate && uv pip install -e '.[dev]'
pytest -q
python -m agentbench.mock_server --port 30000 &
python -m agentbench.sweep --url http://localhost:30000 --concurrency 1,4,16 --out-dir results/mock-sweep
kill %1
```

## 2. Google Cloud project + spending guard (do once)
1. Create a project (ours: `torchtpu-sglang-kr`) and link billing. TPU credits (e.g. TPU Builders Program) apply to TPU usage.
2. Enable the API: `gcloud services enable tpu.googleapis.com --project <PROJECT>`
3. **Budget kill switch** (recommended): Console > Billing > Budgets, create a budget (ours: $200, net of credits)
   that publishes to a Pub/Sub topic (`budget-kill`). Deploy a Cloud Function (`stop-billing`) on that topic that
   unlinks billing when cost > budget. Google's sample: https://cloud.google.com/billing/docs/how-to/disable-billing-with-notifications
   Warning: unlinking billing kills running VMs mid-run.
4. Check v6e quota/zones: `gcloud compute tpus accelerator-types list --zone=europe-west4-a | grep v6e`
   (we used `europe-west4-a`; also try `us-east5-b`, `us-east1-d`). Spot capacity is often empty; fall back to on-demand.

## 3. Create a TPU VM and set it up
```bash
export PROJECT=<PROJECT> ZONE=europe-west4-a
scripts/tpu_up.sh v6e-1 $ZONE              # spot; on-demand: SPOT= scripts/tpu_up.sh v6e-1 $ZONE
gcloud compute tpus tpu-vm scp --recurse . agentbench-v6e-1:~/agentbench --zone $ZONE --project $PROJECT
gcloud compute tpus tpu-vm ssh agentbench-v6e-1 --zone $ZONE --project $PROJECT
```
On the VM:
```bash
cd ~/agentbench
# safety net: deletes the VM after 120 min idle or 240 min total
nohup scripts/idle_delete.sh agentbench-v6e-1 europe-west4-a <PROJECT> > idle.log 2>&1 &
git clone https://github.com/sgl-project/sglang-jax ~/sglang-jax
(cd ~/sglang-jax && uv venv --python 3.12 && . .venv/bin/activate && uv pip install -e "python[all]")
. ~/sglang-jax/.venv/bin/activate && pip install -e ~/agentbench
python -c "from sgl_jax import check_env; check_env.check_env()"
```
The VM service account needs `tpu.nodes.delete` for idle-delete; otherwise delete by hand (section 8).

## 4. Baseline: Qwen2.5-7B on v6e-1, cache on vs off
```bash
DRY_RUN=1 scripts/launch_v6e.sh v6e-1-qwen7b                 # print the server command
TAG=v6e-1-qwen7b scripts/serve_and_bench.sh v6e-1-qwen7b      # cache on, then off; c=1,4,8,16,32
python -m agentbench.analyze results/v6e-1-qwen7b --chips 1 --price-ondemand 2.70 --price-spot 1.00
python scripts/roofline.py results/v6e-1-qwen7b
```
Server flags live in `configs/v6e-1-qwen7b.env` (TP=1, mem 0.85, page 128, chunked prefill 2048, max running 128).
First requests pay XLA compile; the script warms up before measuring.

## 5. Run A: 7B tuning + XProf profile (same VM)
```bash
scripts/runA_v6e1_tuning.sh          # c=64/96/128, page 128 vs 256, chunked prefill 512..4096
scripts/profile_c32.sh               # 20 decode steps at c=32 -> results/.../profile_c32/trace.tgz (~450 MB, gitignored)
python -m agentbench.analyze results/v6e-1-qwen7b-tuning --chips 1
python scripts/roofline.py results/v6e-1-qwen7b-tuning
```
View the trace: `pip install xprof && xprof --logdir <untarred trace dir>`.

## 6. Run B: Qwen3.8-27B-pi-adapted on v6e-4 (TP=4)
Create the VM: `SPOT= scripts/tpu_up.sh v6e-4 $ZONE` (no v6e-4 spot capacity on 2026-11-18). Set up as in section 3, then:
```bash
pip install torch torchvision transformers safetensors     # needed by merge + the model's processor
hf download Qwen/Qwen3.8-27B --local-dir /dev/shm/base                     # ~52 GB; host has ~700 GB RAM
hf download harness-hub/Qwen3.8-27B-pi-adapted --local-dir /dev/shm/adapter
python scripts/merge_lora.py /dev/shm/base /dev/shm/adapter ~/models/qwen38-27b-pi-merged   # expect "256 LoRA pairs", 0 unmatched
scripts/runB_v6e4_qwen38.sh
python -m agentbench.analyze results/v6e-4-qwen38-27b-pi --chips 4
python scripts/roofline.py results/v6e-4-qwen38-27b-pi --mode off --chips 4 --params 27.8e9 \
  --kv-per-tok 65536 --title "Qwen3.8-27B-pi on v6e-4"
```
Server startup takes ~6 min (XLA compile). Why the special flags: see the comment in `scripts/runB_v6e4_qwen38.sh`
and the failed-start logs in `results/v6e-4-qwen38-27b-pi/logs/*FAILED.log`, `*OOM.log`.

## 7. Qwen2.5-72B on v6e-8 (planned, not run: no v6e-8 capacity on 2026-11-18/19)
`scripts/tpu_up.sh v6e-8 $ZONE` (or `SPOT= scripts/tpu_up.sh v6e-8 $ZONE` for on-demand), set up as in section 3, then
`scripts/serve_and_bench.sh v6e-8-qwen72b` and analyze with `--chips 8`. v6e-8 on-demand is ~$21.6/h; check quota per zone first
(`gcloud compute tpus tpu-vm create` returns "User does not have permission ... v6e-8" where the project has none).

## 7b. Run C: OLMo 2 / Olmo 3 ports + int8 weights on v6e-1
Install sglang-jax from the port branch instead of upstream (section 3):
```bash
git clone -b add-olmo3-support https://github.com/kitrakrev/sglang-jax ~/sglang-jax   # adds Olmo2/Olmo3ForCausalLM
(cd ~/sglang-jax && uv venv --python 3.12 && . .venv/bin/activate && uv pip install -e "python[all]")
. ~/sglang-jax/.venv/bin/activate && pip install -e ~/agentbench
scripts/runC_v6e1_olmo_int8.sh     # OLMo-2-7B + Olmo-3-7B cache on/off; Qwen2.5-7B bf16 vs int8 (int8.yaml), c=1..32
for T in v6e-1-olmo2-7b v6e-1-olmo3-7b v6e-1-qwen7b-int8; do python -m agentbench.analyze results/$T --chips 1 --price-ondemand 2.70 --price-spot 1.00; done
```
int8 = `--quantization-config-path=int8.yaml` (built-in, weight-only, quantized at load). Port parity tests run on CPU:
`OLMO_DISABLE_HYBRID_SWA=1 JAX_PLATFORMS=cpu python -m pytest -s test/srt/models/test_olmo3_hf_parity.py` (and `test_olmo2_hf_parity.py`).
Spot v6e-1 can be preempted mid-run (it happened on 2026-11-19); use on-demand (`SPOT=`) for runs you need to finish.

## 8. Collect results, then ALWAYS delete the VM
From the laptop:
```bash
gcloud compute tpus tpu-vm scp --recurse agentbench-v6e-1:~/agentbench/results/<TAG> ./results/ --zone $ZONE --project $PROJECT
gcloud compute tpus tpu-vm delete agentbench-v6e-1 --zone $ZONE --project $PROJECT --quiet
gcloud compute tpus tpu-vm list --zone=- --project $PROJECT                 # must print "Listed 0 items."
gcloud compute tpus queued-resources list --zone=- --project $PROJECT       # must print "Listed 0 items."
```

## 9. GPU baseline on Modal (A100 80GB / H100)
Same workload, seeds, concurrency (1,4,8,16,32,64,96), cache on/off and SLO as sections 4-5, using upstream SGLang v0.5.2
on GPU. Server and client run in one Modal container (client -> localhost). Weights download into a Modal Volume, never
to your laptop. Details: `modal/README.md`.

```bash
# 1. install + log in (once). Creates ~/.modal.toml with your own token.
pip install "modal>=1.4"
modal token new
# If you have several Modal profiles, prefix each command with MODAL_PROFILE=<your-profile> instead of activating one.

# 2. run (from the repo root). First run also downloads Qwen2.5-7B (~15 GB) into volume agentbench-hf-cache.
modal run modal/gpu_bench.py --gpu A100-80GB --tag a100-qwen7b
modal run modal/gpu_bench.py --gpu H100     --tag h100-qwen7b --skip-download
#    optional smoke test first (~5 min): modal run modal/gpu_bench.py --gpu A100-80GB --tag smoke --conc 1,8
#    Results are written straight to results/<tag>/ by the local entrypoint (no separate pull step).

# 3. analyze (local; pip install matplotlib). Prices = Modal on-demand per second x 3600 (modal.com/pricing, 2026-11-18).
python -m agentbench.analyze results/a100-qwen7b --chips 1 --price-ondemand 2.50 --price-spot 2.50
python -m agentbench.analyze results/h100-qwen7b --chips 1 --price-ondemand 3.95 --price-spot 3.95
python scripts/roofline.py results/a100-qwen7b --mode on  --hbm-bw 2.039e12 --peak 312e12 --title "Qwen2.5-7B bf16 on A100-80GB SXM" > results/a100-qwen7b/roofline.md
python scripts/roofline.py results/a100-qwen7b --mode off --hbm-bw 2.039e12 --peak 312e12 --title "Qwen2.5-7B bf16 on A100-80GB SXM" >> results/a100-qwen7b/roofline.md
python scripts/roofline.py results/h100-qwen7b --mode on  --hbm-bw 3.35e12  --peak 989e12 --title "Qwen2.5-7B bf16 on H100 SXM" > results/h100-qwen7b/roofline.md
python scripts/roofline.py results/h100-qwen7b --mode off --hbm-bw 3.35e12  --peak 989e12 --title "Qwen2.5-7B bf16 on H100 SXM" >> results/h100-qwen7b/roofline.md

# 4. verify nothing is left running
modal app list                                  # agentbench-gpu-baseline rows must say "stopped"
modal app stop agentbench-gpu-baseline          # only if one is still ephemeral/deployed
modal volume delete agentbench-hf-cache         # optional, frees the stored weights
```
Expected (measured 2026-11-18): A100 ~18 min wall (~$0.75), H100 ~9 min (~$0.60); both can run in parallel in two
terminals. Functions time out at 90 min. Bandwidth sources: NVIDIA A100 datasheet (80GB SXM 2,039 GB/s) and H100 datasheet
(SXM 3.35 TB/s); `gpu="H100!"` stops Modal from silently upgrading to H200.

## Metrics definitions
- **Goodput SLO**: TTFT < 200 ms and ITL < 25 ms per request; goodput = output tok/s from requests meeting it.
- **$/1M output tokens** = (chips x $/chip-hour) / (output tok/s x 3600) x 1e6.
- **% roofline** = (weights + KV bytes per decode step / HBM bandwidth) / measured ITL. v6e: ~1.64 TB/s, 32 GB per chip; A100-80GB SXM 2.039 TB/s; H100 SXM 3.35 TB/s.

## Troubleshooting
- `--enable-cache-report` / `--disable-radix-cache` rejected: check `python -m sgl_jax.launch_server --help`, edit `scripts/launch_v6e.sh`.
- Spot VM preempted: re-create and rerun; results already written are kept.
- `SPOT=` (empty) means on-demand (fixed 2026-11-18).
