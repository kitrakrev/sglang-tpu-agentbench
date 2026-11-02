"""Synthetic multi-turn agent workloads.

Each session models a tool-using agent: a long shared system prompt (tool
schemas), then turns that append a tool result and ask for the next action.
Context grows every turn, so prefix caching (RadixAttention) matters.
"""

import random
from dataclasses import dataclass, field

TOOLS = ["search_docs", "read_file", "run_sql", "http_get", "list_dir", "grep_repo"]


def _words(rng: random.Random, n: int) -> str:
    vocab = ["error", "latency", "request", "table", "user", "config", "token",
             "cache", "shard", "batch", "kernel", "queue", "timeout", "retry",
             "schema", "index", "worker", "result", "status", "payload"]
    return " ".join(rng.choice(vocab) for _ in range(n))


def system_prompt(n_tool_words: int = 600) -> str:
    rng = random.Random(0)  # identical across sessions -> shared prefix
    parts = ["You are an autonomous engineering agent. Use tools to finish the task."]
    for t in TOOLS:
        parts.append(f"Tool `{t}`: {_words(rng, n_tool_words // len(TOOLS))}")
    parts.append("Reply with one tool call as JSON, or FINAL: <answer>.")
    return "\n".join(parts)


@dataclass
class Session:
    sid: int
    task: str
    tool_results: list[str]
    messages: list[dict] = field(default_factory=list)


def make_sessions(n_sessions: int, n_turns: int, tool_result_words: int = 150,
                  seed: int = 42, sys_words: int = 600) -> list[Session]:
    rng = random.Random(seed)
    sys = system_prompt(sys_words)
    out = []
    for i in range(n_sessions):
        task = f"Task {i}: investigate {_words(rng, 12)}"
        results = [f"[{rng.choice(TOOLS)} output] {_words(rng, tool_result_words)}"
                   for _ in range(n_turns - 1)]
        s = Session(i, task, results)
        s.messages = [{"role": "system", "content": sys},
                      {"role": "user", "content": task}]
        out.append(s)
    return out
