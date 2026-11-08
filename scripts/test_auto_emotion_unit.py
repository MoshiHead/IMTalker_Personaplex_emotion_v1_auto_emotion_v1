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

# --- 10. CSV prediction log ----------------------------------------------------
import csv
import os
import tempfile

from auto_emotion import EmotionLogWriter  # noqa: E402

log_dir = Path(tempfile.mkdtemp(prefix="emotion_log_test_"))
os.environ["IMTALKER_EMOTION_LOG_DIR"] = str(log_dir)

mgr3 = AutoEmotionManager(args)
mgr3._classifier = MockClassifier()  # confident happy
mgr3._logger = mgr3._make_logger()
check("logger created from env var dir", mgr3._logger is not None)
mgr3._worker = __import__("threading").Thread(target=mgr3._worker_loop, daemon=True)
mgr3._worker.start()

mgr3.reset_session()  # session 1
for i in range(0, len(pcm16), chunk):
    mgr3.feed(pcm16[i:i + chunk], SR)
deadline = time.time() + 5.0
while mgr3.current_emotion_id() != 1 and time.time() < deadline:
    time.sleep(0.05)
time.sleep(0.3)  # let the log row flush
mgr3.stop()

log_files = list(log_dir.glob("emotion_log_*.csv"))
check("one CSV log file created", len(log_files) == 1, f"{len(log_files)}")
if log_files:
    rows = list(csv.reader(log_files[0].open(encoding="utf-8")))
    check("CSV has header + 1 utterance row", len(rows) == 2, f"{len(rows)} rows")
    if len(rows) == 2:
        header, row = rows
        check("header columns", header == EmotionLogWriter.HEADER, ",".join(header))
        rec = dict(zip(header, row))
        check("row: session 1", rec["session"] == "1", rec["session"])
        check("row: utterance 1", rec["utterance"] == "1", rec["utterance"])
        check("row: detected label", rec["detected_label"] == "happy", rec["detected_label"])
        check("row: detected emotion", rec["detected_emotion"] == "happy", rec["detected_emotion"])
        check("row: confidence", rec["confidence"] == "0.900", rec["confidence"])
        check("row: accepted", rec["accepted"] == "yes", rec["accepted"])
        check("row: previous emotion", rec["previous_emotion"] == "neutral", rec["previous_emotion"])
        check("row: applied emotion (sent to model)", rec["applied_emotion"] == "happy",
              rec["applied_emotion"])

# low-confidence row: detected angry but applied stays neutral
mgr4 = AutoEmotionManager(args)
mgr4._classifier = LowConfClassifier()
mgr4._logger = mgr4._make_logger()
mgr4._worker = __import__("threading").Thread(target=mgr4._worker_loop, daemon=True)
mgr4._worker.start()
mgr4.reset_session()
for i in range(0, len(pcm16), chunk):
    mgr4.feed(pcm16[i:i + chunk], SR)
time.sleep(1.0)
mgr4.stop()
log_files2 = sorted(log_dir.glob("emotion_log_*.csv"))
check("second log file created", len(log_files2) == 2, f"{len(log_files2)}")
if len(log_files2) == 2:
    rows = list(csv.reader(log_files2[-1].open(encoding="utf-8")))
    if len(rows) == 2:
        rec = dict(zip(rows[0], rows[1]))
        check("low-conf row: detected angry", rec["detected_emotion"] == "angry", rec["detected_emotion"])
        check("low-conf row: not accepted", rec["accepted"] == "no", rec["accepted"])
        check("low-conf row: applied stays neutral", rec["applied_emotion"] == "neutral",
              rec["applied_emotion"])

os.environ.pop("IMTALKER_EMOTION_LOG_DIR", None)
import shutil
shutil.rmtree(log_dir, ignore_errors=True)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    sys.exit(1)
print("ALL UNIT TESTS PASSED")
