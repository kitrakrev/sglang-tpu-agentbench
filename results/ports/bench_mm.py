"""Multimodal load test for an SGLang(-JAX) server: image or audio prompts.

Each request carries a unique image/clip (no encoder-cache hits), streams the
response, and records TTFT (includes preprocessing + encoder + prefill), ITL
and output tok/s. Works against sgl_jax and upstream SGLang /generate.

usage: python bench_mm.py --kind vlm|audio --model M --conc 1,4,8,16 --out results/x
"""

import argparse
import asyncio
import base64
import io
import json
import os
import statistics
import time

import aiohttp
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--kind", choices=["vlm", "audio"], required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--url", default="http://localhost:30000")
ap.add_argument("--conc", default="1,4,8,16")
ap.add_argument("--per-conc", type=int, default=4, help="requests per worker")
ap.add_argument("--max-tokens", type=int, default=128)
ap.add_argument("--out", required=True)
args = ap.parse_args()
SR = 16000


def chat_text(proc, kind, n_mm, prompt):
    item = {"type": "image"} if kind == "vlm" else {"type": "audio", "audio_url": "x"}
    content = [item] * n_mm + [{"type": "text", "text": prompt}]
    return proc.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=False)


def make_image(seed: int, size=(896, 672)) -> str:
    from PIL import Image

    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([(xx * 255 // w), (yy * 255 // h), ((xx // 48 + yy // 48) % 2) * 255], -1)
    img = (base * 0.7 + rng.integers(0, 76, size=base.shape)).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def make_clip(seed: int, seconds=10.0) -> str:
    import soundfile as sf

    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 200 + 400 * rng.random()
    wav = 0.3 * np.sin(2 * np.pi * f0 * t) + 0.02 * rng.standard_normal(t.shape)
    buf = io.BytesIO()
    sf.write(buf, wav.astype(np.float32), SR, format="WAV")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


async def one(session, text, mm, results):
    body = {
        "text": text,
        "sampling_params": {"max_new_tokens": args.max_tokens, "temperature": 0.0, "ignore_eos": True},
        "stream": True,
        **mm,
    }
    t0 = time.perf_counter()
    ttft, last, n_tok, err = None, None, 0, None
    try:
        async with session.post(f"{args.url}/generate", json=body) as r:
            if r.status != 200:
                err = f"HTTP {r.status}: {(await r.text())[:200]}"
            else:
                async for raw in r.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:") or line == "data: [DONE]":
                        continue
                    msg = json.loads(line[5:])
                    now = time.perf_counter()
                    if ttft is None and msg.get("text"):
                        ttft = now - t0
                    last = now
                    n_tok = msg.get("meta_info", {}).get("completion_tokens", n_tok)
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    e2e = (last or time.perf_counter()) - t0
    results.append(dict(ttft=ttft, e2e=e2e, tokens=n_tok, err=err,
                        prompt_tokens=None))


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))] if xs else None


async def run_level(c, proc, seed0):
    prompts = ["Describe this image in detail.", "What objects and colors do you see?"] if args.kind == "vlm" \
        else ["Describe this sound.", "What is the pitch of this tone?"]
    n = c * args.per_conc
    reqs = []
    for i in range(n):
        s = seed0 + i
        if args.kind == "vlm":
            mm = {"image_data": [make_image(s)]}
        else:
            mm = {"audio_data": [make_clip(s)]}
        reqs.append((chat_text(proc, args.kind, 1, prompts[i % 2]), mm))
    results = []
    sem = asyncio.Semaphore(c)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async def guarded(t, m):
            async with sem:
                await one(session, t, m, results)
        t0 = time.perf_counter()
        await asyncio.gather(*(guarded(t, m) for t, m in reqs))
        wall = time.perf_counter() - t0
    ok = [r for r in results if r["err"] is None and r["ttft"] is not None]
    itl = [(r["e2e"] - r["ttft"]) / max(1, r["tokens"] - 1) * 1000 for r in ok]
    row = dict(
        concurrency=c, requests=n, errors=n - len(ok),
        output_tok_per_s=round(sum(r["tokens"] for r in ok) / wall, 2),
        requests_per_s=round(len(ok) / wall, 3),
        ttft_p50_ms=round(pct([r["ttft"] * 1000 for r in ok], 50), 1) if ok else None,
        ttft_p99_ms=round(pct([r["ttft"] * 1000 for r in ok], 99), 1) if ok else None,
        itl_p50_ms=round(statistics.median(itl), 2) if itl else None,
        e2e_p50_ms=round(pct([r["e2e"] * 1000 for r in ok], 50), 1) if ok else None,
        first_error=next((r["err"] for r in results if r["err"]), None),
    )
    return row


async def main():
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.model)
    os.makedirs(args.out, exist_ok=True)
    # warmup: one pass per concurrency level so XLA compiles every batch/encoder shape
    # before measuring; unique seeds so measured requests never hit the encoder cache
    per_conc, args.per_conc = args.per_conc, 1
    for c in (int(x) for x in args.conc.split(",")):
        await run_level(c, proc, seed0=10_000 + 1_000 * c)
    args.per_conc = per_conc
    rows = []
    for i, c in enumerate(int(x) for x in args.conc.split(",")):
        row = await run_level(c, proc, seed0=(i + 1) * 100_000)
        print(json.dumps(row), flush=True)
        rows.append(row)
    json.dump(dict(model=args.model, kind=args.kind, max_tokens=args.max_tokens,
                   image_size="896x672" if args.kind == "vlm" else None,
                   clip_seconds=10.0 if args.kind == "audio" else None, rows=rows),
              open(os.path.join(args.out, "metrics.json"), "w"), indent=1)
    cols = ["concurrency", "requests", "errors", "output_tok_per_s", "requests_per_s",
            "ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "e2e_p50_ms"]
    with open(os.path.join(args.out, "metrics.md"), "w") as f:
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in cols) + " |\n")


asyncio.run(main())
