"""Fake OpenAI-compatible streaming server with a toy prefix cache.

Lets the harness be tested without a TPU. TTFT scales with *uncached* prompt
chars, so multi-turn sessions show the same shape a radix cache produces.

    python -m agentbench.mock_server --port 30000
"""

import argparse
import asyncio
import json

from aiohttp import web

PREFILL_S_PER_KCHAR = 0.004
DECODE_S_PER_TOKEN = 0.002


class PrefixCache:
    def __init__(self):
        self.seen: set[str] = set()

    def hit_chars(self, prompt: str, block: int = 256) -> int:
        hit = 0
        for end in range(block, len(prompt) + 1, block):
            if prompt[:end] in self.seen:
                hit = end
            else:
                break
        for end in range(block, len(prompt) + 1, block):
            self.seen.add(prompt[:end])
        return hit


cache = PrefixCache()


async def chat(request: web.Request):
    body = await request.json()
    prompt = "\n".join(m["content"] for m in body["messages"])
    hit = cache.hit_chars(prompt)
    await asyncio.sleep((len(prompt) - hit) / 1000 * PREFILL_S_PER_KCHAR)
    n = int(body.get("max_tokens", 16))
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for i in range(n):
        chunk = {"choices": [{"index": 0, "delta": {"content": f"t{i} "}}]}
        await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
        await asyncio.sleep(DECODE_S_PER_TOKEN)
    usage = {"choices": [], "usage": {"prompt_tokens": len(prompt) // 4,
                                      "completion_tokens": n,
                                      "prompt_tokens_details": {"cached_tokens": hit // 4}}}
    await resp.write(f"data: {json.dumps(usage)}\n\n".encode())
    await resp.write(b"data: [DONE]\n\n")
    return resp


def make_app():
    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    return app


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30000)
    web.run_app(make_app(), port=p.parse_args().port)
