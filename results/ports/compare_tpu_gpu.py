"""Build TPU v6e-1 vs A100 vs H100 tables for the ports from results/ports/.

$/1M output tokens = hourly price / (output tok/s * 3600) * 1e6 (on-demand list prices,
2026-11-18: v6e $2.70/chip-h on GCP, A100-80GB $2.50/h and H100 $3.95/h on Modal).
"""

import csv
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent
PRICE = {"v6e-1": 2.70, "A100": 2.50, "H100": 3.95}
MODELS = [
    ("gemma3-1b", "gemma-3-1b-it (text)", "text"),
    ("gemma3-4b", "gemma-3-4b-it (text)", "text"),
    ("glm4", "GLM-4-9B-0414 (text)", "text"),
    ("internvl3", "InternVL3-2B-hf (896x672 image, 3,328 image tokens)", "vlm"),
    ("qwen2-audio", "Qwen2-Audio-7B-Instruct (10 s clip, 250 audio tokens)", "audio"),
]


def text_rows(d: pathlib.Path):
    f = next((p for p in (d / "report.csv", d / "bench" / "report.csv") if p.exists()), None)
    if f is None:
        return None
    rows = {}
    for r in csv.DictReader(open(f)):
        rows[int(r["concurrency"])] = dict(
            tps=float(r["output_tok_per_s"]), ttft50=float(r["ttft_p50_ms"]), ttft99=float(r["ttft_p99_ms"]),
            itl=float(r["itl_p50_ms"]), err=int(r["errors"]))
    return rows


def mm_rows(d: pathlib.Path):
    f = d / "metrics.json"
    if not f.exists():
        return None
    return {r["concurrency"]: dict(tps=r["output_tok_per_s"], ttft50=r["ttft_p50_ms"], ttft99=r["ttft_p99_ms"],
                                   itl=r["itl_p50_ms"], err=r["errors"]) for r in json.load(open(f))["rows"]}


def load(key, kind, dev):
    if dev == "v6e-1":
        d = ROOT / (key if kind == "text" else f"{key}-mmbench")
    else:
        d = ROOT / f"{key}-{dev.lower()}"
    rows = text_rows(d) if kind == "text" else mm_rows(d)
    note = (d / "note.txt").read_text().strip() if (d / "note.txt").exists() else ""
    gpu = (d / "gpu.txt").read_text().split(",")[0].strip() if (d / "gpu.txt").exists() else ""
    if gpu and "H100" not in gpu and "A100" not in gpu:
        note = f"ran on {gpu} (Modal upgraded the H100 request; priced as H100). " + note
    return rows, note


def fmt(x, nd=0):
    return "-" if x is None else f"{x:,.{nd}f}"


out = []
for key, title, kind in MODELS:
    out.append(f"#### {title}\n")
    out.append("| c | device | out tok/s | TTFT p50 / p99 ms | ITL p50 ms | $/1M out | errors |")
    out.append("|---|---|---|---|---|---|---|")
    data = {dev: load(key, kind, dev) for dev in PRICE}
    concs = sorted({c for rows, _ in data.values() if rows for c in rows})
    for c in concs:
        for dev in PRICE:
            rows, _ = data[dev]
            r = (rows or {}).get(c)
            if r is None:
                out.append(f"| {c} | {dev} | n/a | n/a | n/a | n/a | n/a |")
                continue
            usd = PRICE[dev] / (r["tps"] * 3600) * 1e6 if r["tps"] else None
            out.append(f"| {c} | {dev} | {fmt(r['tps'])} | {fmt(r['ttft50'])} / {fmt(r['ttft99'])} | "
                       f"{fmt(r['itl'], 1)} | {fmt(usd, 2)} | {r['err']} |")
    notes = [f"{dev}: {n}" for dev, (_, n) in data.items() if n]
    if notes:
        out.append("\nNotes: " + "; ".join(notes))
    out.append("")
md = "\n".join(out)
(ROOT / "tpu_vs_gpu.md").write_text(md)
print(md)
