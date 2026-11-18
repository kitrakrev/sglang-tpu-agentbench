import argparse
import asyncio
import socket

from aiohttp import web

from agentbench import mock_server
from agentbench.run import main_async, parse_args, pct
from agentbench.workload import make_sessions


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_sessions_share_system_prefix():
    ss = make_sessions(4, 3)
    assert len({s.messages[0]["content"] for s in ss}) == 1
    assert all(len(s.tool_results) == 2 for s in ss)


def test_pct():
    assert pct([3, 1, 2], 50) == 2
    assert pct([], 90) == 0.0


def test_end_to_end_against_mock(tmp_path):
    async def go():
        mock_server.cache = mock_server.PrefixCache()
        port = free_port()
        runner = web.AppRunner(mock_server.make_app())
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        try:
            args = parse_args(["--url", f"http://127.0.0.1:{port}", "--sessions", "6",
                               "--turns", "4", "--concurrency", "3", "--max-tokens", "8",
                               "--out", str(tmp_path / "r.json")])
            return await main_async(args)
        finally:
            await runner.cleanup()

    s = asyncio.run(go())
    assert s["requests"] == 24 and s["errors"] == 0
    assert s["output_tok_per_s"] > 0
    assert s["mean_cached_prompt_tokens"] > 0  # shared system prompt gets reused
    assert (tmp_path / "r.json").exists()


def test_server_down_records_errors():
    args = parse_args(["--url", f"http://127.0.0.1:{free_port()}", "--sessions", "2",
                       "--turns", "3", "--timeout", "2"])
    s = asyncio.run(main_async(args))
    assert s["errors"] == 2  # each session stops at its first failed turn


def test_sweep_against_mock(tmp_path):
    from agentbench import sweep

    async def go():
        mock_server.cache = mock_server.PrefixCache()
        port = free_port()
        runner = web.AppRunner(mock_server.make_app())
        await runner.setup()
        await web.TCPSite(runner, "127.0.0.1", port).start()
        try:
            a = argparse.Namespace(url=f"http://127.0.0.1:{port}", model="m",
                                   concurrency="1,4", sessions_per_conc=2, turns=3,
                                   max_tokens=4, tool_words=50, sys_words=200,
                                   out_dir=str(tmp_path))
            return await sweep.sweep(a)
        finally:
            await runner.cleanup()

    md = asyncio.run(go())
    rows = sweep.load_dir(str(tmp_path))
    assert [r["concurrency"] for r in rows] == [1, 4]
    assert all(r["errors"] == 0 and r["output_tok_per_s"] > 0 for r in rows)
    assert rows[0]["mean_cached_prompt_tokens"] > 0
    assert "| concurrency |" in md
    assert (tmp_path / "report.csv").exists()


def test_launch_script_dry_run():
    import subprocess
    for cfg in ("v6e-1-llama8b", "v6e-8-llama70b"):
        out = subprocess.run(["bash", "scripts/launch_v6e.sh", cfg],
                             env={"DRY_RUN": "1", "PATH": "/usr/bin:/bin"},
                             capture_output=True, text=True, check=True).stdout
        assert "sgl_jax.launch_server" in out and "--tp-size" in out
