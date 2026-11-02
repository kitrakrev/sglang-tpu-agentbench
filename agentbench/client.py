"""Streaming OpenAI-compatible client that records per-turn latency."""

import json
import time
from dataclasses import dataclass, asdict

import httpx


@dataclass
class TurnResult:
    sid: int
    turn: int
    prompt_chars: int
    ttft_s: float
    e2e_s: float
    out_tokens: int
    itl_s: float  # mean inter-token latency
    cached_tokens: int | None
    ok: bool
    error: str = ""

    def to_dict(self):
        return asdict(self)


async def chat_stream(client: httpx.AsyncClient, url: str, model: str,
                      messages: list[dict], max_tokens: int,
                      sid: int, turn: int) -> tuple[TurnResult, str]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": 0.0, "stream": True,
            "stream_options": {"include_usage": True}}
    prompt_chars = sum(len(m["content"]) for m in messages)
    t0 = time.perf_counter()
    first = None
    stamps: list[float] = []
    text = []
    cached = None
    try:
        async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage")
                if usage:
                    details = usage.get("prompt_tokens_details") or {}
                    cached = details.get("cached_tokens", cached)
                for ch in chunk.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        now = time.perf_counter()
                        if first is None:
                            first = now
                        stamps.append(now)
                        text.append(piece)
    except Exception as e:  # record failures, keep the run going
        dt = time.perf_counter() - t0
        return TurnResult(sid, turn, prompt_chars, dt, dt, 0, 0.0, None, False, repr(e)), ""
    end = time.perf_counter()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    itl = sum(gaps) / len(gaps) if gaps else 0.0
    ttft = (first or end) - t0
    return TurnResult(sid, turn, prompt_chars, ttft, end - t0, len(stamps), itl,
                      cached, True), "".join(text)
