"""Decode roofline for Qwen2.5-7B bf16 on TPU v6e-1 vs measured ITL. Writes charts/roofline.png and prints a table."""
import csv, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HBM_BW = 1640e9      # v6e HBM bandwidth, B/s (Google Cloud v6e docs)
PEAK = 918e12        # v6e bf16 peak FLOP/s
PARAMS = 7.62e9      # Qwen2.5-7B incl. embeddings
W_BYTES = PARAMS * 2 # bf16
KV_PER_TOK = 2 * 28 * 4 * 128 * 2  # K+V * layers * kv_heads * head_dim * bf16 = 57,344 B
CTX = 1400           # ~mean prompt (1236) + half of output

def floor_ms(b):
    mem = (W_BYTES + b * CTX * KV_PER_TOK) / HBM_BW
    comp = 2 * PARAMS * b / PEAK
    return max(mem, comp) * 1e3

root = pathlib.Path(__file__).resolve().parents[1] / "results/v6e-1-qwen7b"
rows = [r for r in csv.DictReader(open(root / "metrics.csv")) if r["mode"] == "on"]
print("| c | floor ITL ms | measured ITL p50 ms | % of roofline | roofline decode tok/s | measured out tok/s |")
print("|---|---|---|---|---|---|")
for r in rows:
    c = int(r["concurrency"]); f = floor_ms(c); m = float(r["itl_p50_ms"])
    print(f"| {c} | {f:.2f} | {m:.2f} | {100*f/m:.0f}% | {c*1e3/f:.0f} | {float(r['output_tok_per_s']):.0f} |")

bs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(bs, [b * 1e3 / floor_ms(b) for b in bs], label="HBM/compute roofline (decode only)", color="#555")
ax.plot([int(r["concurrency"]) for r in rows], [int(r["concurrency"]) * 1e3 / float(r["itl_p50_ms"]) for r in rows],
        "o-", label="measured: c / ITL p50 (cache on)", color="#1f77b4")
ax.plot([int(r["concurrency"]) for r in rows], [float(r["output_tok_per_s"]) for r in rows],
        "s--", label="measured end-to-end output tok/s", color="#ff7f0e")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("concurrent sequences"); ax.set_ylabel("decode tokens/s")
ax.set_title("Qwen2.5-7B bf16 on v6e-1: decode roofline vs measured"); ax.grid(alpha=.3); ax.legend(fontsize=8)
(root / "charts").mkdir(exist_ok=True); fig.tight_layout(); fig.savefig(root / "charts/roofline.png", dpi=130)
