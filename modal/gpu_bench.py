"""GPU baseline: SGLang (CUDA) + agentbench sweep in one Modal container (client hits localhost).

    MODAL_PROFILE=<profile> modal run modal/gpu_bench.py --gpu A100-80GB --tag a100-qwen7b
    MODAL_PROFILE=<profile> modal run modal/gpu_bench.py --gpu H100 --tag h100-qwen7b

Mirrors scripts/serve_and_bench.sh + configs/v6e-1-qwen7b.env: same model, workload, seeds,
mem-fraction 0.85, chunked prefill 2048, max-running 128, --enable-cache-report; cache-off adds
--disable-radix-cache. Page size is SGLang-GPU default (1). Weights cached in a Modal Volume.
"""
import os
import pathlib

import modal

MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONC = os.environ.get("CONC", "1,4,8,16,32,64,96")
REPO = pathlib.Path(__file__).resolve().parent.parent

hf_vol = modal.Volume.from_name("agentbench-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.2-cu126")
    .entrypoint([])
    .pip_install("httpx>=0.27", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(REPO / "agentbench", "/root/agentbench", ignore=["__pycache__"])
)
app = modal.App("agentbench-gpu-baseline", image=image)


@app.function(volumes={"/hf": hf_vol}, timeout=30 * 60, cpu=4)
def download():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"])
    hf_vol.commit()


def _bench(conc: str) -> dict:
    import json, subprocess, time, urllib.request
    os.chdir("/root")
    out = {}

    def smi():
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        name, used, tot = [x.strip() for x in q.splitlines()[0].split(",")]
        return f"{name}: {float(used)/1024:.2f} GiB / {float(tot)/1024:.2f} GiB\n"

    for mode in ("cache", "nocache"):
        cmd = ["python", "-m", "sglang.launch_server", "--model-path", MODEL, "--port", "30000",
               "--tp", "1", "--dtype", "bfloat16", "--mem-fraction-static", "0.85",
               "--chunked-prefill-size", "2048", "--max-running-requests", "128",
               "--enable-cache-report"] + (["--disable-radix-cache"] if mode == "nocache" else [])
        log = open(f"/tmp/server-{mode}.log", "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            try:
                urllib.request.urlopen("http://localhost:30000/health", timeout=5); break
            except Exception:
                if p.poll() is not None or time.time() - t0 > 900:
                    raise RuntimeError(open(f"/tmp/server-{mode}.log").read()[-4000:])
                time.sleep(5)
        t1 = time.time()
        subprocess.run(["python", "-m", "agentbench.run", "--url", "http://localhost:30000", "--model", MODEL,
                        "--sessions", "8", "--turns", "3", "--concurrency", "8", "--max-tokens", "16"],
                       check=True, capture_output=True)
        t2 = time.time()
        d = f"/tmp/res/{mode}"; os.makedirs(d, exist_ok=True)
        hbm_idle = smi()
        r = subprocess.run(["python", "-m", "agentbench.sweep", "--url", "http://localhost:30000",
                            "--model", MODEL, "--concurrency", conc, "--turns", "6",
                            "--sessions-per-conc", "4", "--out-dir", d], capture_output=True, text=True)
        hbm_after = smi()
        p.terminate(); p.wait(60); log.close(); time.sleep(5)
        files = {f"{mode}/timing.json": json.dumps({"startup_s": round(t1 - t0), "warmup_s": round(t2 - t1)}),
                 f"{mode}/hbm_idle.txt": hbm_idle, f"{mode}/hbm_after.txt": hbm_after,
                 f"{mode}/client.log": r.stdout + r.stderr,
                 f"logs/server-{mode}.log": open(f"/tmp/server-{mode}.log").read()}
        for f in os.listdir(d):
            files[f"{mode}/{f}"] = open(os.path.join(d, f)).read()
        out.update(files)
        if r.returncode:
            break
    return out


@app.function(gpu="A100-80GB", volumes={"/hf": hf_vol}, timeout=90 * 60)
def bench_a100(conc: str = CONC) -> dict:
    return _bench(conc)


@app.function(gpu="H100!", volumes={"/hf": hf_vol}, timeout=90 * 60)
def bench_h100(conc: str = CONC) -> dict:
    return _bench(conc)


@app.local_entrypoint()
def main(gpu: str = "A100-80GB", tag: str = "a100-qwen7b", conc: str = CONC, skip_download: bool = False):
    if not skip_download:
        download.remote()
    fn = {"A100-80GB": bench_a100, "H100": bench_h100}[gpu]
    files = fn.remote(conc)
    root = REPO / "results" / tag
    for k, v in files.items():
        path = root / k; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(v)
    print(f"wrote {len(files)} files to {root}")
