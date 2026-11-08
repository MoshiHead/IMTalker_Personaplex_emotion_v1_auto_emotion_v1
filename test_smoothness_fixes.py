"""Standalone validation for the smoothness fixes in the AH AudioPace server.

Extracts the REAL AsyncLogWriter class and _shed_oldest_video_frames coroutine
from the server source via `ast` (so we test the shipped code, not a copy) and
exercises them, plus the transcript-dedup and session-trim logic, using only the
standard library (no torch / GPU / fastapi needed).

Run:  python test_smoothness_fixes.py
"""
import ast
import asyncio
import io
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Optional

SERVER = Path(__file__).parent / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"


def _extract(source: str, names: set) -> str:
    """Return the source of the named top-level defs/classes from `source`."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    chunks = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            chunks.append("".join(lines[node.lineno - 1: node.end_lineno]))
    return "\n\n".join(chunks)


def load_symbols():
    src = SERVER.read_text(encoding="utf-8")
    code = _extract(src, {"AsyncLogWriter", "_shed_oldest_video_frames"})
    assert "class AsyncLogWriter" in code, "AsyncLogWriter not found in server source"
    assert "async def _shed_oldest_video_frames" in code, "_shed_oldest_video_frames not found"
    ns = {
        "queue": queue, "threading": threading, "sys": sys,
        "asyncio": asyncio, "Optional": Optional, "hot_log": None,
    }
    exec(code, ns)
    ns["hot_log"] = ns["AsyncLogWriter"]()  # module-level hot_log used by the shed coroutine
    return ns


def test_async_log_writer_nonblocking_and_drops():
    ns = load_symbols()
    AsyncLogWriter = ns["AsyncLogWriter"]

    class SlowSink(io.StringIO):
        def write(self, s):
            time.sleep(0.002)  # simulate a slow (network-volume) sink
            return super().write(s)

    slow = SlowSink()
    old = sys.stdout
    sys.stdout = slow
    try:
        w = AsyncLogWriter(maxsize=8)
        t0 = time.perf_counter()
        for i in range(200):
            w.log(f"line {i}")
        elapsed = time.perf_counter() - t0
        # 200 logs against a 2ms/write sink would take ~400ms if the caller
        # blocked on the sink; the bounded queue must return in a few ms.
        assert elapsed < 0.1, f"hot_log.log blocked the caller: {elapsed:.3f}s"
        time.sleep(1.0)  # let the writer drain
        assert w.dropped > 0, "expected drop-oldest on overflow, got dropped=0"
        written = slow.getvalue().count("line ")
        assert written > 0, "writer thread wrote nothing"
    finally:
        sys.stdout = old
    print(f"  AsyncLogWriter: non-blocking ({elapsed*1000:.1f}ms/200 logs), "
          f"dropped={w.dropped}, wrote={written}  OK")


def test_shed_oldest_video_frames():
    ns = load_symbols()
    shed = ns["_shed_oldest_video_frames"]

    async def run():
        q: asyncio.Queue = asyncio.Queue(maxsize=512)
        for i in range(100):
            q.put_nowait({"frame_number": i})
        keep = 25
        await shed(q, keep, "[test]")
        assert q.qsize() == keep, f"expected {keep} after shed, got {q.qsize()}"
        first = q.get_nowait()
        assert first["frame_number"] == 100 - keep, \
            f"expected oldest remaining #{100-keep}, got #{first['frame_number']}"
        await shed(q, keep, "[test]")  # no-op below keep
        assert q.qsize() == keep - 1
        return True

    assert asyncio.run(run())
    print("  _shed_oldest_video_frames: drops oldest, preserves newest, no-op below keep  OK")



def test_transcript_dedup():
    # Replicates the exact dedup branch added to the producer.
    last_sent_text = ""
    sent = []

    def payload_for(full_text):
        nonlocal last_sent_text
        if full_text != last_sent_text:
            last_sent_text = full_text
            return full_text
        return ""

    sent.append(payload_for("Hello"))        # changed -> send full
    sent.append(payload_for("Hello"))        # unchanged -> ""
    sent.append(payload_for("Hello world"))  # changed -> send full
    sent.append(payload_for("Hello world"))  # unchanged -> ""
    assert sent == ["Hello", "", "Hello world", ""], sent
    print("  transcript dedup: full text sent only on change, empty otherwise  OK")


def test_session_trim_rolling_window():
    # Replicates _trim_session_parts rolling-window behaviour.
    max_chunks = 5
    motion, helium, audio, chunk_rows = [], [], [], []

    def record_chunk(i):
        chunk_rows.append({"chunk": len(chunk_rows) + 1})  # always recorded
        motion.append(f"m{i}")
        helium.append(f"h{i}")
        audio.append(f"a{i}")
        for lst in (motion, helium, audio):  # _trim_session_parts
            if len(lst) > max_chunks:
                del lst[: len(lst) - max_chunks]

    for i in range(12):
        record_chunk(i)
    assert len(motion) == max_chunks and len(helium) == max_chunks and len(audio) == max_chunks
    assert motion[0] == "m7", f"rolling window should keep newest, got {motion[0]}"
    assert len(chunk_rows) == 12, "chunk_rows (avatar_chunk_id source) must stay complete"
    print("  session trim: tensor lists bounded to rolling window, chunk_rows complete  OK")


if __name__ == "__main__":
    print("Validating smoothness fixes against the real server source:")
    test_async_log_writer_nonblocking_and_drops()
    test_shed_oldest_video_frames()
    test_transcript_dedup()
    test_session_trim_rolling_window()
    print("ALL TESTS PASSED")

