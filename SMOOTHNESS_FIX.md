# Smoothness Fix — Avatar Video Freezing While Audio Continues

## Symptom

During longer PersonaPlex responses the avatar video would freeze for a few
seconds (2–4 times per reply) while the voice kept playing, then recover. The
short greeting was always smooth; the problem grew worse the longer the
conversation ran.

## Root cause

The AH AudioPace server enqueues a chunk's **audio** packets to `audio_q`
*before* rendering, but its **video** packets only exist *after* render + JPEG.
Both senders are paced to the same wall-clock schedule anchored at session start
(`media_epoch`). So whenever the GPU producer thread runs late, audio still goes
out on schedule while video frames arrive late — and the browser (which slaves
video to audio playback time) freezes the avatar, then skips ahead.

Four compounding causes made the GPU producer run late intermittently:

1. **Blocking disk I/O on the GPU thread.** Every Moshi step (12.5/s) and every
   chunk printed with `flush=True`. The notebook redirects stdout to
   `/workspace/.../live_server.log`, and on RunPod `/workspace` is a **network
   volume** with multi-hundred-ms latency spikes. A slow `write()` froze the
   whole pipeline while the browser's audio buffer kept the voice playing.
2. **Zero steady-state jitter buffer.** `media_epoch` was anchored only +80 ms
   after the 1-chunk prebuffer, so `frame_q` hovered near empty; any transient
   > ~0.5 s starved video instantly.
3. **`frame_q` backpressure blocked the producer** when video sending lagged,
   coupling a video backlog to the whole conversation.
4. **Time-dependent load growth:** every frame carried the full session
   transcript (bandwidth grew with conversation length), and `--dump_motion`
   session tensor lists grew ~1.6 MB/s (~100 MB/min) unboundedly → RAM pressure
   and GC pauses.

The greeting is smooth because it is short, starts with fresh state, and the
startup buffers cover initial jitter — there is no time for drift or a
network-volume spike to hit.

## What changed (permanent, structural)

`IMTalker/liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py`
- **Non-blocking hot-path logging** — new `AsyncLogWriter` (daemon thread +
  bounded drop-oldest queue). Per-step / per-chunk / gate / backlog logs go
  through it, so a slow log sink can never stall the GPU thread. Startup and
  health-marker prints stay direct (notebook health check unaffected).
- **End-to-end jitter buffer** — new `--jitter_buffer_sec` (default **1.0**).
  `media_epoch` is anchored this far after prebuffer, banking ~1 s of frames and
  audio in the queues to absorb transients. It is a *constant playback offset*,
  **not** added conversational latency (the model still hears/replies in real
  time).
- **Frame shedding instead of blocking backpressure** — when `frame_q` reaches
  the high-water mark, the oldest queued video frames are dropped (down to the
  jitter depth) on the event loop instead of sleeping. The browser skips missing
  frame numbers and jumps to the newest, so the conversation never stalls behind
  a video backlog.
- **Transcript dedup** — the full transcript is sent only when it changed
  (browser keeps the last non-empty value), removing bandwidth growth.
- **Hidden state stays on GPU** — removed the per-step `.cpu()` (12 GPU→CPU
  syncs per chunk).
- **Bounded session recording** — `--dump_motion` tensor lists are now gated on
  the flag and capped to a rolling `--dump_max_chunks` window (default 900 ≈
  15 min); `avatar_chunk_id` still increments correctly.

`IMTalker/start_winner_live.sh`
- `--frame_q_backpressure 32` → `${FRAME_Q_BACKPRESSURE:-160}` (the code's own
  default) so the producer can bank the jitter buffer without blocking.
- Added `--jitter_buffer_sec "${JITTER_BUFFER_SEC:-1.0}"`.

`RunPod_RTX5090_PersonaPlex_IMTalker_Live_fixed.ipynb`
- New `JITTER_BUFFER_SEC` parameter (blank = script default) wired into
  `env_overrides`, matching the existing pattern.

## Tuning

- `JITTER_BUFFER_SEC` (notebook parameter or env): `1.0` default. Lower to
  `0.5` for a snappier start, raise to `2.0`–`3.0` for maximum smoothness on a
  noisy network/GPU. It only shifts playback by a constant offset.
- `FRAME_Q_BACKPRESSURE` (env): high-water mark for frame shedding; default 160
  (~6.4 s). Rarely needs changing.

## How to verify from the logs

- `[AJ][VIDEO] STARVED` events should disappear (or only appear on genuine
  multi-second network outages).
- `[GPU][chunk#N] ... frame_q=<depth>` should sit around `jitter_buffer_sec × 25`
  (≈ 25 at the default) instead of hovering near 0.
- `[GPU][chunk#N] ... produce_latency=<ms>` should stay well under 960 ms.
- A `video backlog: shed N oldest frame(s)` line appears only when the video
  socket genuinely stalls > ~5 s; the conversation keeps going and video skips
  ahead on recovery.

## Validation

- `python -m py_compile` on the server: OK.
- `python test_smoothness_fixes.py`: extracts the real `AsyncLogWriter` and
  `_shed_oldest_video_frames` from the server via `ast` and verifies
  non-blocking logging, drop-oldest shedding, transcript dedup, and the
  rolling-window session trim. All pass.
- Notebook remains valid JSON.

Nothing in the documented non-negotiable contract was touched: the 12/24/6
timing, CFG/NFE, chunk size, adapter, RMS gate, checkpoints, Opus transport +
separate sockets, FP32 renderer precision, and the AH anti-burst audio pacing
are all unchanged.
