"""Metric-heavy report + charts from a serve_and_bench results dir.

    python -m agentbench.analyze results/<TAG> --chips 1 --price-ondemand 2.70 --price-spot 1.00

Reads results/<TAG>/{cache,nocache}/c*.json (+ timing.json, hbm_*.txt) and writes
results/<TAG>/metrics.md, metrics.csv, metrics.json and charts/*.png.
"""
import argparse
import glob
import json
import math
import os
import re


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = min(len(xs) - 1, max(0, math.ceil(p / 100 * len(xs)) - 1))
    return xs[k]


def ms(x, nd=1):
    return None if x is None else round(x * 1e3, nd)


SLO_TTFT, SLO_TPOT = 0.200, 0.025


def metrics_for(path, chips, p_od, p_spot):
    d = json.load(open(path))
    s, turns = d["summary"], d["turns"]
    ok = [t for t in turns if t["ok"]]
    n = len(turns)
    ttft = [t["ttft_s"] for t in ok]
    itl = [t["itl_s"] for t in ok if t["out_tokens"] > 1]
    e2e = [t["e2e_s"] for t in ok]
    out_tok = sum(t["out_tokens"] for t in ok)
    wall = s.get("wall_s") or (out_tok / s["output_tok_per_s"] if s.get("output_tok_per_s") else None)
    by_turn = {}
    for t in ok:
        by_turn.setdefault(t["turn"], []).append(t["ttft_s"])
    turns_sorted = sorted(by_turn)
    t0 = pct(by_turn[turns_sorted[0]], 50) if turns_sorted else None
    tn = pct(by_turn[turns_sorted[-1]], 50) if turns_sorted else None
    cached = [t.get("cached_tokens") for t in ok if t.get("cached_tokens") is not None]
    ptok = [t.get("prompt_tokens") for t in ok if t.get("prompt_tokens")]
    cached_pct = None
    if cached and ptok and len(ptok) == len(ok):
        cached_pct = round(100 * sum(t.get("cached_tokens") or 0 for t in ok) / sum(ptok), 1)
    tps = s.get("output_tok_per_s")
    # goodput: requests meeting SLO TTFT<200 ms and TPOT(ITL)<25 ms (docs/NEXT_EXPERIMENTS.md #3)
    good = [t for t in ok if t["ttft_s"] < SLO_TTFT and (t["out_tokens"] <= 1 or t["itl_s"] < SLO_TPOT)]
    good_tok = sum(t["out_tokens"] for t in good)
    usd_1m = lambda price: round(price * chips / 3600 / tps * 1e6, 3) if tps else None
    return {
        "concurrency": s["config"]["concurrency"],
        "requests": n,
        "error_rate_pct": round(100 * (n - len(ok)) / n, 2) if n else None,
        "output_tok_per_s": tps,
        "output_tok_per_s_per_chip": round(tps / chips, 1) if tps else None,
        "requests_per_s": s.get("requests_per_s"),
        "ttft_p50_ms": ms(pct(ttft, 50)), "ttft_p90_ms": ms(pct(ttft, 90)), "ttft_p99_ms": ms(pct(ttft, 99)),
        "itl_p50_ms": ms(pct(itl, 50), 2), "itl_p90_ms": ms(pct(itl, 90), 2), "itl_p99_ms": ms(pct(itl, 99), 2),
        "e2e_p50_ms": ms(pct(e2e, 50)), "e2e_p99_ms": ms(pct(e2e, 99)),
        "mean_prompt_tokens": round(sum(ptok) / len(ptok), 1) if ptok else None,
        "mean_cached_tokens": round(sum(cached) / len(cached), 1) if cached else None,
        "cached_tokens_pct": cached_pct,
        "ttft_turn1_ms": ms(t0), "ttft_turnN_ms": ms(tn),
        "ttft_turn1_over_turnN": round(t0 / tn, 2) if t0 and tn else None,
        "slo_attain_pct": round(100 * len(good) / n, 1) if n else None,
        "goodput_tok_per_s": round(tps * good_tok / out_tok, 1) if tps and out_tok else None,
        "usd_per_1M_out_ondemand": usd_1m(p_od),
        "usd_per_1M_out_spot_est": usd_1m(p_spot),
        "ttft_by_turn_ms": {str(k): ms(pct(v, 50)) for k, v in sorted(by_turn.items())},
        "wall_s": wall,
    }


def hbm(text):
    # tpu-info table rows look like "| 0 | 29.10 GiB / 31.25 GiB | 93.1% |"
    m = re.findall(r"([\d.]+)\s*GiB\s*/\s*([\d.]+)\s*GiB", text or "")
    return f"{m[0][0]} / {m[0][1]} GiB" if m else None


COLS = ["concurrency", "requests", "error_rate_pct", "output_tok_per_s", "output_tok_per_s_per_chip",
        "requests_per_s", "ttft_p50_ms", "ttft_p90_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p90_ms",
        "itl_p99_ms", "e2e_p50_ms", "e2e_p99_ms", "mean_prompt_tokens", "mean_cached_tokens",
        "cached_tokens_pct", "ttft_turn1_ms", "ttft_turnN_ms", "ttft_turn1_over_turnN",
        "slo_attain_pct", "goodput_tok_per_s", "usd_per_1M_out_ondemand", "usd_per_1M_out_spot_est"]


def table(rows):
    out = ["| mode | " + " | ".join(COLS) + " |", "|---|" + "---|" * len(COLS)]
    for r in rows:
        out.append(f"| {r['mode']} | " + " | ".join("" if r[c] is None else str(r[c]) for c in COLS) + " |")
    return "\n".join(out)


def charts(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(out, exist_ok=True)
    modes = sorted({r["mode"] for r in rows})
    spec = [("output_tok_per_s", "Output tok/s"), ("ttft_p50_ms", "TTFT p50 (ms)"),
            ("ttft_p99_ms", "TTFT p99 (ms)"), ("itl_p50_ms", "ITL p50 (ms)"),
            ("e2e_p99_ms", "E2E p99 (ms)")]
    for key, label in spec:
        fig, ax = plt.subplots(figsize=(6, 4))
        for m in modes:
            rr = sorted([r for r in rows if r["mode"] == m], key=lambda r: r["concurrency"])
            ax.plot([r["concurrency"] for r in rr], [r[key] for r in rr], marker="o", label=m)
        ax.set_xscale("log", base=2); ax.set_xlabel("concurrent sessions"); ax.set_ylabel(label)
        ax.grid(alpha=.3); ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(out, f"{key}.png"), dpi=120); plt.close(fig)
    # TTFT by turn at highest concurrency
    fig, ax = plt.subplots(figsize=(6, 4))
    for m in modes:
        rr = sorted([r for r in rows if r["mode"] == m], key=lambda r: r["concurrency"])
        if not rr:
            continue
        bt = rr[-1]["ttft_by_turn_ms"]
        ks = sorted(bt, key=int)
        ax.plot([int(k) + 1 for k in ks], [bt[k] for k in ks], marker="o",
                label=f"cache {m}, c={rr[-1]['concurrency']}")
    ax.set_xlabel("agent turn"); ax.set_ylabel("median TTFT (ms)"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(out, "ttft_by_turn.png"), dpi=120); plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("tagdir")
    p.add_argument("--chips", type=int, default=1)
    p.add_argument("--price-ondemand", type=float, default=2.70, help="$/chip-hour")
    p.add_argument("--price-spot", type=float, default=1.00, help="$/chip-hour (estimate)")
    a = p.parse_args(argv)
    rows, extra = [], {}
    subdirs = sorted(x for x in os.listdir(a.tagdir)
                     if glob.glob(os.path.join(a.tagdir, x, "c*.json")))
    for mode in subdirs:
        d = os.path.join(a.tagdir, mode)
        for f in glob.glob(os.path.join(d, "c*.json")):
            r = metrics_for(f, a.chips, a.price_ondemand, a.price_spot)
            r["mode"] = {"cache": "on", "nocache": "off"}.get(mode, mode)
            rows.append(r)
        info = {}
        tf = os.path.join(d, "timing.json")
        if os.path.exists(tf):
            info.update(json.load(open(tf)))
        for k in ("hbm_idle", "hbm_after"):
            f = os.path.join(d, f"{k}.txt")
            if os.path.exists(f):
                info[k] = hbm(open(f).read())
        extra[mode] = info
    rows.sort(key=lambda r: (r["mode"], r["concurrency"]))
    md = table(rows) + "\n\nStartup / warmup / HBM: " + json.dumps(extra) + "\n"
    open(os.path.join(a.tagdir, "metrics.md"), "w").write(md)
    with open(os.path.join(a.tagdir, "metrics.csv"), "w") as f:
        f.write("mode," + ",".join(COLS) + "\n")
        for r in rows:
            f.write(r["mode"] + "," + ",".join("" if r[c] is None else str(r[c]) for c in COLS) + "\n")
    json.dump({"rows": rows, "server": extra}, open(os.path.join(a.tagdir, "metrics.json"), "w"), indent=1)
    if rows:
        charts(rows, os.path.join(a.tagdir, "charts"))
    print(md)


if __name__ == "__main__":
    main()
