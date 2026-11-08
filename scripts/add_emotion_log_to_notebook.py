#!/usr/bin/env python3
"""Add the emotion prediction CSV log to the notebook:

  * re-embeds the updated IMTalker/auto_emotion.py (EmotionLogWriter) into
    the Step 12 cell
  * extends the Step 13 PATCH_CHECKS for auto_emotion.py
  * documents the log location in the header and the final cell
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
AE_PATH = ROOT / "IMTalker" / "auto_emotion.py"

ae_content = AE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
if "'''" in ae_content:
    sys.exit("auto_emotion.py must not contain triple single quotes")
if "class EmotionLogWriter" not in ae_content:
    sys.exit("auto_emotion.py has no EmotionLogWriter yet")

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(i):
    return "".join(cells[i]["source"])


def set_src(i, s):
    cells[i]["source"] = s.splitlines(keepends=True)


def find_cell(cell_id):
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            return i
    raise KeyError(cell_id)


def replace_once(text, old, new, label):
    n = text.count(old)
    if n != 1:
        sys.exit(f"anchor for {label} found {n}x (expected exactly 1)")
    return text.replace(old, new)


# --- 1. re-embed AUTO_EMOTION_MODULE in the Step 12 cell ---------------------
i33 = find_cell("emotion-stream-33")
c33 = src(i33)
m = re.search(r"AUTO_EMOTION_MODULE = r'''(.*?)'''", c33, re.DOTALL)
if not m:
    sys.exit("AUTO_EMOTION_MODULE not found in Step 12 cell")
if "class EmotionLogWriter" in m.group(1):
    sys.exit("Step 12 already embeds the EmotionLogWriter version")
c33 = c33.replace(m.group(1), ae_content)
set_src(i33, c33)
print("[ok] Step 12: AUTO_EMOTION_MODULE re-embedded (with EmotionLogWriter)")

# --- 2. Step 13 PATCH_CHECKS: add EmotionLogWriter ----------------------------
i35 = find_cell("emotion-stream-35")
c35 = src(i35)
c35 = replace_once(
    c35,
    '     ["class AutoEmotionManager", "class EnergyVAD", "class SpeechEmotionClassifier",\n'
    '      "get_auto_emotion_manager", "init_auto_emotion_manager"]),',
    '     ["class AutoEmotionManager", "class EnergyVAD", "class SpeechEmotionClassifier",\n'
    '      "class EmotionLogWriter", "get_auto_emotion_manager", "init_auto_emotion_manager"]),',
    "Step 13 auto_emotion.py checks",
)
set_src(i35, c35)
print("[ok] Step 13: PATCH_CHECKS extended")

# --- 3. header point 6: mention the CSV log -----------------------------------
i0 = find_cell("emotion-stream-00")
c0 = src(i0)
c0 = replace_once(
    c0,
    "   so audio/video streaming is untouched; any failure falls back to the fixed\n"
    "   `--emotion_id` behaviour.\n",
    "   so audio/video streaming is untouched; any failure falls back to the fixed\n"
    "   `--emotion_id` behaviour. Every classified utterance is also appended to a\n"
    "   CSV log (`IMTalker/emotion_logs/emotion_log_*.csv`): detected emotion,\n"
    "   confidence, and the emotion actually sent to the avatar model.\n",
    "header point 6 log mention",
)
set_src(i0, c0)
print("[ok] header: log documented")

# --- 4. final cell: print the log location ------------------------------------
i51 = find_cell("emotion-stream-51")
c51 = src(i51)
c51 = replace_once(
    c51,
    '    print("  [AUTO-EMOTION] utterance 2.31s -> happy (p=0.87) [neutral -> happy] switching")\n',
    '    print("  [AUTO-EMOTION] utterance 2.31s -> happy (p=0.87) [neutral -> happy] switching")\n'
    '    print()\n'
    '    print("Prediction log (one CSV row per user utterance):")\n'
    '    print("  IMTalker/emotion_logs/emotion_log_<timestamp>.csv")\n'
    '    print("  columns: timestamp, session, utterance, duration_sec, detected_label,")\n'
    '    print("           detected_emotion, confidence, accepted, previous_emotion,")\n'
    '    print("           applied_emotion  <- the emotion sent to the avatar model")\n',
    "final cell log location",
)
set_src(i51, c51)
print("[ok] final cell: log location documented")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[ok] notebook written: {NB_PATH}")
