"""Standalone validation for the AV_SMOOTH_PATCH_V1 / EMOTION_LIVE_PATCH_V4 fixes.

Follows test_smoothness_fixes.py conventions: the REAL nested batch-enqueue
functions are extracted from the shipped server source via `ast` (so we test the
deployed code, not a copy) and exercised against a real asyncio event loop, and
the static invariants (pacing, reply_codes defer, watchdog, worklet cap,
frameStore purge, launch-script gating) are checked against the shipped sources.

Run:  python test_av_smooth_patch.py
"""
import ast
import asyncio
import textwrap
import threading
import time
from pathlib import Path

BASE = Path(__file__).parent
SERVER = BASE / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
START_SH = BASE / "IMTalker" / "start_winner_live.sh"
WORKLET = BASE / "IMTalker" / "static" / "assets" / "audio-processor-aj-nodrop.js"
HTML = BASE / "IMTalker" / "static" / "index_v3_binary_fullscreen_aj_nodrop.html"


def _extract_nested(source: str, names: set) -> str:
    """Return the source of the named (possibly nested) functions from `source`."""
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    chunks = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            chunks.append(textwrap.dedent("".join(lines[node.lineno - 1: node.end_lineno])))
    return "\n\n".join(chunks)


class _MockLatency:
    def __init__(self):
        self.calls = 0

    def queue_state(self, *a, **k):
        self.calls += 1


class _MockReplyEngine:
    def __init__(self):
        self.marks = []
        self.conv_logger = type("C", (), {"latency": _MockLatency()})()

    def _mark_first_per_turn(self, name, turn, **k):
        self.marks.append(name)

    def input_backlog_sec(self):
        return 0.0


def _make_loop():
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    threading.Thread(target=_run, daemon=True).start()
    ready.wait(2.0)
    return loop


def _load_batch_fns(frame_q, audio_q, loop, reply_engine):
    src = SERVER.read_text(encoding="utf-8")
    code = _extract_nested(src, {"_enqueue_frame_batch", "_enqueue_audio_batch"})
    assert "def _enqueue_frame_batch" in code, "_enqueue_frame_batch not found"
    assert "def _enqueue_audio_batch" in code, "_enqueue_audio_batch not found"
    ns = {
        "asyncio": asyncio, "frame_q": frame_q, "audio_q": audio_q,
        "event_loop": loop, "reply_engine": reply_engine,
        "args": type("A", (), {"fps": 25.0})(), "FRAME_Q_PUT_TIMEOUT_S": 10.0,
    }
    exec(code, ns)
    return ns["_enqueue_frame_batch"], ns["_enqueue_audio_batch"]


def test_batched_enqueue_real_event_loop():
    """Run the shipped batch functions against a real loop; verify in-order
    enqueue of a whole chunk and the empty-list no-op."""
    loop = _make_loop()
    frame_q = asyncio.Queue(maxsize=512)
    audio_q = asyncio.Queue(maxsize=512)
    reply_engine = _MockReplyEngine()
    enq_frame, enq_audio = _load_batch_fns(frame_q, audio_q, loop, reply_engine)

    # Empty list is a no-op (must not touch the loop).
    enq_frame([])
    enq_audio([])
    assert frame_q.qsize() == 0 and audio_q.qsize() == 0

    # A full 24-frame chunk enqueues in order.
    enq_frame([{"frame_number": i, "turn_id": 1, "t_ready": time.perf_counter()} for i in range(24)])
    enq_audio([{"frame_number": i, "turn_id": 1, "audio_pcm": b"x"} for i in range(24)])

    got_frames = [frame_q.get_nowait()["frame_number"] for _ in range(24)]
    got_audio = [audio_q.get_nowait()["frame_number"] for _ in range(24)]
    assert got_frames == list(range(24)), f"frame order broken: {got_frames}"
    assert got_audio == list(range(24)), f"audio order broken: {got_audio}"
    # Per-packet first-per-turn marks + queue_state logging preserved.
    assert "server_avatar_frame_ready" in reply_engine.marks
    assert "server_audio_queue_enter" in reply_engine.marks
    assert reply_engine.conv_logger.latency.calls == 24, \
        f"expected 24 queue_state logs, got {reply_engine.conv_logger.latency.calls}"

    loop.call_soon_threadsafe(loop.stop)
    print("  batched enqueue: 24 frames + 24 audio in order, marks/logging preserved, "
          "empty no-op  OK")
def test_batched_enqueue_backpressure():
    """A full queue blocks the producer until the consumer drains (real backpressure).

    All queue operations are kept on the loop thread (asyncio.Queue is not
    thread-safe); the producer runs on a separate thread exactly like the real
    GPU producer, which reaches the queue only via run_coroutine_threadsafe.
    """
    loop = _make_loop()
    frame_q = asyncio.Queue(maxsize=4)  # tiny queue to force blocking
    audio_q = asyncio.Queue(maxsize=4)
    enq_frame, _ = _load_batch_fns(frame_q, audio_q, loop, _MockReplyEngine())

    async def _fill():
        for i in range(4):
            await frame_q.put({"frame_number": i})
    asyncio.run_coroutine_threadsafe(_fill(), loop).result(2.0)

    done = threading.Event()

    def _producer():
        enq_frame([{"frame_number": 4, "turn_id": None}, {"frame_number": 5, "turn_id": None}])
        done.set()

    pt = threading.Thread(target=_producer, daemon=True)
    t0 = time.perf_counter()
    pt.start()
    time.sleep(0.2)
    assert not done.is_set(), "producer should block on a full queue"

    # Drain two items on the loop thread -> the batched put can proceed.
    async def _drain():
        a = await frame_q.get()
        b = await frame_q.get()
        return a["frame_number"], b["frame_number"]
    drained = asyncio.run_coroutine_threadsafe(_drain(), loop).result(3.0)
    assert drained == (0, 1), drained

    done.wait(3.0)
    assert done.is_set(), "producer did not resume after drain"
    assert time.perf_counter() - t0 >= 0.2, "producer returned before space was available"

    loop.call_soon_threadsafe(loop.stop)
    print("  batched enqueue: blocks on full queue, resumes on drain (backpressure intact)  OK")


def test_static_invariants():
    srv = SERVER.read_text(encoding="utf-8")
    # Pacing at exactly real time (no 8% catch-up ratchet).
    assert "min_send_interval_s = max(0.001, 1.0 / float(args.fps))" in srv
    assert "0.92 / float(args.fps)" not in srv
    # reply_codes kept on GPU in _step; CPU transfer only in the dump path.
    assert "reply_codes = tokens[:, 1:].detach()\n" in srv
    assert 'reply_codes = tokens[:, 1:].detach().to(device="cpu", dtype=torch.int16)' not in srv
    assert srv.count("if fm_engine.dump_motion:") >= 2, "used_codes not gated behind dump_motion"
    # Real-time budget watchdog + V4 marker present.
    assert "OVER-BUDGET chunk#" in srv
    assert "EMOTION_LIVE_PATCH_V4" in srv
    print("  server static invariants: pacing=1.0, reply_codes deferred, watchdog, V4  OK")

    sh = START_SH.read_text(encoding="utf-8")
    assert '--render_sub_batch "${RENDER_SUB_BATCH:-24}"' in sh
    assert '"${DUMP_MOTION_ARGS[@]}"' in sh and "DUMP_MOTION_ARGS=(--dump_motion)" in sh
    assert "  --dump_motion \\\n" not in sh, "unconditional --dump_motion still present"
    print("  start_winner: render_sub_batch=24, dump_motion gated  OK")

    wk = WORKLET.read_text(encoding="utf-8")
    assert "this.disableEmergencyDrop = false;" in wk
    assert "this.hardMaxBufferSamples = asSamples(1200);" in wk
    assert "AV_SMOOTH_PATCH_V1" in wk
    print("  worklet: bounded emergency drop enabled (1200ms cap)  OK")

    ht = HTML.read_text(encoding="utf-8")
    assert "frameStore.size > 150" in ht
    assert "oldIdx < lastRenderedFrame - 1" in ht
    assert "AV_SMOOTH_PATCH_V1" in ht
    print("  html: frameStore purge + 150-frame cap  OK")


if __name__ == "__main__":
    print("Validating AV_SMOOTH_PATCH_V1 / EMOTION_LIVE_PATCH_V4 against the shipped sources:")
    test_batched_enqueue_real_event_loop()
    test_batched_enqueue_backpressure()
    test_static_invariants()
    print("ALL TESTS PASSED")

