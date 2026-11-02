"""Run a multi-turn agent serving benchmark against an OpenAI-compatible server.

    python -m agentbench.run --url http://localhost:30000 --model Qwen/Qwen3-8B \
        --sessions 64 --turns 6 --concurrency 16 --out results/run.json
"""

import argparse
import asyncio
import json
import os
import statistics
import time

import httpx

from .client import chat_stream
from .workload import make_sessions


def pct(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, round(p / 100 * (len(xs) - 1))))
    return xs[k]


def summarize(results, wall_s):
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok]
    itl = [r.itl_s for r in ok if r.out_tokens > 1]
    out_tok = sum(r.out_tokens for r in ok)
    by_turn = {}
    for r in ok:
        by_turn.setdefault(r.turn, []).append(r.ttft_s)
    cached = [r.cached_tokens for r in ok if r.cached_tokens is not None]
    return {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": round(wall_s, 3),
        "output_tok_per_s": round(out_tok / wall_s, 2) if wall_s else 0,
        "requests_per_s": round(len(ok) / wall_s, 3) if wall_s else 0,
        "ttft_p50_ms": round(pct(ttft, 50) * 1e3, 1),
        "ttft_p90_ms": round(pct(ttft, 90) * 1e3, 1),
        "ttft_p99_ms": round(pct(ttft, 99) * 1e3, 1),
        "itl_p50_ms": round(pct(itl, 50) * 1e3, 2),
        "itl_p99_ms": round(pct(itl, 99) * 1e3, 2),
        "ttft_median_ms_by_turn": {t: round(statistics.median(v) * 1e3, 1)
                                   for t, v in sorted(by_turn.items())},
        "mean_cached_prompt_tokens": round(statistics.mean(cached), 1) if cached else None,
    }


async def run_session(client, args, s, sem, results):
    n_turns = len(s.tool_results) + 1
    for turn in range(n_turns):
        async with sem:
            res, text = await chat_stream(client, args.url, args.model, s.messages,
                                          args.max_tokens, s.sid, turn)
        results.append(res)
        if not res.ok:
            return
        s.messages.append({"role": "assistant", "content": text or "{}"})
        if turn < n_turns - 1:
            s.messages.append({"role": "user", "content": s.tool_results[turn]})
        if args.think_time:
            await asyncio.sleep(args.think_time)  # simulated tool execution


async def main_async(args):
    sessions = make_sessions(args.sessions, args.turns, args.tool_words,
                             args.seed, args.sys_words)
    sem = asyncio.Semaphore(args.concurrency)
    results = []
    timeout = httpx.Timeout(args.timeout, connect=10)
    async with httpx.AsyncClient(timeout=timeout) as client:
        t0 = time.perf_counter()
        await asyncio.gather(*(run_session(client, args, s, sem, results)
                               for s in sessions))
        wall = time.perf_counter() - t0
    summary = summarize(results, wall)
    summary["config"] = {k: v for k, v in vars(args).items() if k != "out"}
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary,
                       "turns": [r.to_dict() for r in results]}, f, indent=1)
    return summary


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:30000")
    p.add_argument("--model", default="default")
    p.add_argument("--sessions", type=int, default=32)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--tool-words", type=int, default=150)
    p.add_argument("--sys-words", type=int, default=600)
    p.add_argument("--think-time", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="")
    return p.parse_args(argv)


def main(argv=None):
    summary = asyncio.run(main_async(parse_args(argv)))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=1))


if __name__ == "__main__":
    main()
