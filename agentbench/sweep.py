"""Concurrency sweep + report: throughput / TTFT / ITL / cache reuse vs concurrency.

    python -m agentbench.sweep --url http://localhost:30000 --model M \
        --concurrency 1,4,8,16,32 --turns 6 --out-dir results/TAG
    python -m agentbench.sweep --report results/TAG      # table from existing c*.json
"""

import argparse
import asyncio
import glob
import json
import os

from .run import main_async, parse_args as run_args

COLS = ["concurrency", "requests", "errors", "output_tok_per_s", "requests_per_s",
        "ttft_p50_ms", "ttft_p99_ms", "itl_p50_ms", "itl_p99_ms",
        "mean_cached_prompt_tokens", "ttft_turn0_ms", "ttft_last_turn_ms"]


def row(conc, s):
    bt = s.get("ttft_median_ms_by_turn") or {}
    keys = sorted(bt, key=int)
    r = {k: s.get(k) for k in COLS}
    r["concurrency"] = conc
    r["ttft_turn0_ms"] = bt[keys[0]] if keys else None
    r["ttft_last_turn_ms"] = bt[keys[-1]] if keys else None
    return r


def to_markdown(rows):
    out = ["| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for r in rows:
        out.append("| " + " | ".join("" if r[c] is None else str(r[c]) for c in COLS) + " |")
    return "\n".join(out)


def load_dir(d):
    rows = []
    for f in glob.glob(os.path.join(d, "c*.json")):
        s = json.load(open(f))["summary"]
        rows.append(row(s["config"]["concurrency"], s))
    return sorted(rows, key=lambda r: r["concurrency"])


def write_report(d, rows):
    os.makedirs(d, exist_ok=True)
    md = to_markdown(rows)
    open(os.path.join(d, "report.md"), "w").write(md + "\n")
    with open(os.path.join(d, "report.csv"), "w") as f:
        f.write(",".join(COLS) + "\n")
        for r in rows:
            f.write(",".join("" if r[c] is None else str(r[c]) for c in COLS) + "\n")
    return md


async def sweep(a):
    for c in [int(x) for x in a.concurrency.split(",")]:
        argv = ["--url", a.url, "--model", a.model, "--sessions", str(c * a.sessions_per_conc),
                "--turns", str(a.turns), "--concurrency", str(c), "--max-tokens", str(a.max_tokens),
                "--tool-words", str(a.tool_words), "--sys-words", str(a.sys_words),
                "--out", os.path.join(a.out_dir, f"c{c}.json")]
        await main_async(run_args(argv))
    return write_report(a.out_dir, load_dir(a.out_dir))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:30000")
    p.add_argument("--model", default="default")
    p.add_argument("--concurrency", default="1,4,8,16,32")
    p.add_argument("--sessions-per-conc", type=int, default=4)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--tool-words", type=int, default=150)
    p.add_argument("--sys-words", type=int, default=600)
    p.add_argument("--out-dir", default="results/sweep")
    p.add_argument("--report", default="", help="only build report from an existing dir")
    a = p.parse_args(argv)
    if a.report:
        print(write_report(a.report, load_dir(a.report)))
    else:
        print(asyncio.run(sweep(a)))


if __name__ == "__main__":
    main()
