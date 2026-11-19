"""GPU comparison for the sgl_jax ports: upstream SGLang (CUDA) on one Modal A100-80GB.

    MODAL_PROFILE=<profile> modal run modal/gpu_ports.py

Runs, in one container each:
  - GLM-4-9B-0414: agentbench sweep, radix cache on, c=1,4,8,16 (same workload as TPU)
  - InternVL3-2B:  results/ports/bench_mm.py image load test, c=1,4,8,16
Server flags match the TPU runs where they exist: bf16, mem-fraction 0.8,
chunked prefill 2048 (4096 for the VLM), max-running 64, --enable-cache-report.
"""

import os
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
hf_vol = modal.Volume.from_name("agentbench-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.2-cu126")
    .entrypoint([])
    .pip_install("httpx>=0.27", "aiohttp", "soundfile", "huggingface_hub[hf_transfer]")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(REPO / "agentbench", "/root/agentbench", ignore=["__pycache__"])
    .add_local_file(REPO / "results" / "ports" / "bench_mm.py", "/root/bench_mm.py")
)
app = modal.App("agentbench-gpu-ports", image=image)

RUNS = {
    "glm4": dict(model="zai-org/GLM-4-9B-0414", kind="text", conc="1,4,8,16", chunk="2048", extra=[]),
    "internvl3": dict(model="OpenGVLab/InternVL3-2B-hf", kind="vlm", conc="1,4,8,16", chunk="4096",
                      extra=["--disable-cuda-graph"]),  # v0.5.2: CUDA graph capture fails for this model
}


def _serve_and_bench(key: str) -> dict:
    import subprocess
    import time
    import urllib.request

    cfg = RUNS[key]
    os.chdir("/root")
    from huggingface_hub import snapshot_download

    snapshot_download(cfg["model"], allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py"])
    hf_vol.commit()
    log = open(f"/tmp/server-{key}.log", "w")
    cmd = ["python", "-m", "sglang.launch_server", "--model-path", cfg["model"], "--port", "30000",
           "--tp", "1", "--dtype", "bfloat16", "--mem-fraction-static", "0.8", "--trust-remote-code",
           "--chunked-prefill-size", cfg["chunk"], "--max-running-requests", "64", "--enable-cache-report",
           *cfg["extra"]]
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    while True:
        try:
            urllib.request.urlopen("http://localhost:30000/health", timeout=5)
            break
        except Exception:
            if proc.poll() is not None or time.time() - t0 > 1500:
                return {f"logs/server-{key}.log": open(f"/tmp/server-{key}.log").read(), "error": "server failed"}
            time.sleep(5)
    startup = time.time() - t0
    d = f"/tmp/out-{key}"
    if cfg["kind"] == "text":
        subprocess.run(["python", "-m", "agentbench.run", "--url", "http://localhost:30000", "--model", cfg["model"],
                        "--sessions", "8", "--turns", "3", "--concurrency", "8", "--max-tokens", "16"],
                       capture_output=True)
        r = subprocess.run(["python", "-m", "agentbench.sweep", "--url", "http://localhost:30000", "--model",
                            cfg["model"], "--concurrency", cfg["conc"], "--turns", "6", "--sessions-per-conc", "4",
                            "--out-dir", d], capture_output=True, text=True)
    else:
        r = subprocess.run(["python", "/root/bench_mm.py", "--kind", cfg["kind"], "--model", cfg["model"],
                            "--conc", cfg["conc"], "--out", d], capture_output=True, text=True)
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    proc.terminate()
    out = {"timing.json": f'{{"startup_s": {startup:.0f}}}', "gpu.txt": smi, "client.log": r.stdout + r.stderr,
           f"logs/server-{key}.log": open(f"/tmp/server-{key}.log").read()}
    if os.path.isdir(d):
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p):
                out[f] = open(p).read()
    return out


@app.function(gpu="A100-80GB", volumes={"/hf": hf_vol}, timeout=60 * 60)
def bench_a100(key: str) -> dict:
    return _serve_and_bench(key)


@app.local_entrypoint()
def main(keys: str = "glm4,internvl3"):
    for key in keys.split(","):
        files = bench_a100.remote(key)
        root = REPO / "results" / "ports" / f"{key}-a100"
        for k, v in files.items():
            path = root / k
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(v)
        print(f"{key}: wrote {len(files)} files to {root}")
