#!/usr/bin/env python3
"""Functional unit tests for IMTalker/auto_emotion.py (no torch required).

Exercises the EnergyVAD segmentation state machine and the AutoEmotionManager
threading/queueing behaviour with a mock classifier.
"""

import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "IMTalker"))

from auto_emotion import AutoEmotionManager, EnergyVAD, EMOTION_NAMES  # noqa: E402

SR = 24_000
FAILURES = []


def check(name, cond, detail=""):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {name}" + (f" - {detail}" if detail else ""))
    if not cond:
        FAILURES.append(name)


def tone(seconds, amp=0.1, freq=220.0):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


# --- 1. basic utterance: speech bracketed by silence closes after 0.7s -------
vad = EnergyVAD()
out = vad.feed(np.concatenate([silence(1.0), tone(2.0), silence(1.0)]))
check("utterance closes after trailing silence", out is not None)
if out is not None:
    dur = len(out) / SR
    check("utterance duration ~2.4s (2s speech + pre-roll + 0.15s tail)", 2.1 < dur < 2.8, f"{dur:.2f}s")

# --- 2. short blips are dropped ----------------------------------------------
vad = EnergyVAD()
out = vad.feed(np.concatenate([tone(0.2), silence(1.0)]))
check("0.2s blip dropped (< min_speech_sec)", out is None)

# --- 3. incremental feeding (packet-sized chunks) -----------------------------
vad = EnergyVAD()
stream = np.concatenate([silence(0.5), tone(1.5), silence(1.0)])
outs = []
step = int(0.02 * SR)  # 20ms browser packets
for i in range(0, len(stream), step):
    o = vad.feed(stream[i:i + step])
    if o is not None:
        outs.append(o)
check("incremental 20ms packets produce exactly one utterance", len(outs) == 1, f"{len(outs)}")

# --- 4. max utterance split ----------------------------------------------------
vad = EnergyVAD(max_utterance_sec=3.0)
outs = []
stream = tone(7.0)
for i in range(0, len(stream), step):
    o = vad.feed(stream[i:i + step])
    if o is not None:
        outs.append(o)
check("7s monologue split into >=2 pieces (max_utterance_sec=3)", len(outs) >= 2, f"{len(outs)}")

# --- 5. hysteresis: brief dip below threshold does not close ------------------
vad = EnergyVAD()
stream = np.concatenate([tone(1.0), silence(0.3), tone(1.0), silence(1.0)])
out = vad.feed(stream)
check("0.3s pause does not close the utterance", out is not None and len(out) / SR > 2.0,
      f"{None if out is None else len(out) / SR:.2f}s")

# --- 6. reset drops partial state ----------------------------------------------
vad = EnergyVAD()
vad.feed(tone(1.0))
vad.reset()
out = vad.feed(silence(1.0))
check("reset clears in-flight utterance", out is None)

# --- 7. manager: feed -> worker -> published emotion id ------------------------
class MockClassifier:
    def __init__(self):
        self.calls = 0

    def predict(self, pcm):
        self.calls += 1
        return 1, "happy", 0.9  # confident happy


args = types.SimpleNamespace(
    emotion_id=0, ser_confidence_threshold=0.5,
    vad_rms_threshold=0.02, vad_silence_sec=0.7,
    vad_min_speech_sec=0.5, vad_max_utterance_sec=12.0,
)
mgr = AutoEmotionManager(args)
mgr._classifier = MockClassifier()
mgr._worker = __import__("threading").Thread(target=mgr._worker_loop, daemon=True)
mgr._worker.start()

pcm16 = (np.concatenate([silence(0.2), tone(1.5), silence(1.0)]) * 32768).astype(np.int16)
chunk = int(0.02 * SR)
for i in range(0, len(pcm16), chunk):
    mgr.feed(pcm16[i:i + chunk], SR)

deadline = time.time() + 5.0
while mgr.current_emotion_id() != 1 and time.time() < deadline:
    time.sleep(0.05)
check("manager publishes detected emotion (0 -> 1)", mgr.current_emotion_id() == 1)
check("classifier ran exactly once", mgr._classifier.calls == 1, f"{mgr._classifier.calls}")
check("no feed errors", mgr.feed_errors == 0)

# --- 8. low confidence keeps previous emotion ----------------------------------
class LowConfClassifier:
    def predict(self, pcm):
        return 3, "angry", 0.3  # below 0.5 threshold


mgr2 = AutoEmotionManager(args)
mgr2._classifier = LowConfClassifier()
mgr2._worker = __import__("threading").Thread(target=mgr2._worker_loop, daemon=True)
mgr2._worker.start()
for i in range(0, len(pcm16), chunk):
    mgr2.feed(pcm16[i:i + chunk], SR)
time.sleep(1.0)
check("low-confidence detection keeps previous emotion", mgr2.current_emotion_id() == 0)

# --- 9. reset_session keeps emotion, clears VAD --------------------------------
mgr.reset_session()
check("reset_session keeps last detected emotion", mgr.current_emotion_id() == 1)
check("reset_session clears VAD state", mgr.vad._in_speech is False)

mgr.stop()
mgr2.stop()

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL UNIT TESTS PASSED")
