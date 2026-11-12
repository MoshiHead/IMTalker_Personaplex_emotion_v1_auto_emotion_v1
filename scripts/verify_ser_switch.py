#!/usr/bin/env python3
"""Verify (and repair) the emotion2vec SER switch in runpod_emotion_streaming.ipynb.

Checks:
  * the notebook parses as JSON
  * AUTO_EMOTION_MODULE embedded in the Step 12 cell is byte-identical to
    IMTalker/auto_emotion.py (re-embeds it from the file when it is not)
  * every code cell touched by the switch still compiles
  * the new model id is in place everywhere expected; the old id may only
    remain in the intentional historical mentions inside auto_emotion.py
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
AE_PATH = ROOT / "IMTalker" / "auto_emotion.py"

OLD_MODEL = "superb/wav2vec2-base-superb-er"
NEW_MODEL = "emotion2vec/emotion2vec_plus_base"
# code cells touched by the switch: Parameters, Step 6 installs, Step 12
# patches/embedded module, Step 17 health-check hints
TOUCHED_CODE_CELLS = [
    "emotion-stream-03",
    "emotion-stream-21",
    "emotion-stream-33",
    "emotion-stream-44",
]

errors = []


def fail(msg):
    errors.append(msg)
    print(f"[FAIL] {msg}")


def ok(msg):
    print(f"[ok] {msg}")


ae_content = AE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]


def src(c):
    return "".join(c["source"])


def set_src(c, s):
    c["source"] = s.splitlines(keepends=True)


by_id = {c.get("id"): c for c in cells}

# 1. embedded module identity (repair in place if the copy is stale)
c33 = by_id["emotion-stream-33"]
s33 = src(c33)
m = re.search(r"AUTO_EMOTION_MODULE = r'''(.*?)'''", s33, re.DOTALL)
if not m:
    fail("AUTO_EMOTION_MODULE not found in Step 12 cell")
elif m.group(1) != ae_content:
    set_src(c33, s33.replace(m.group(1), ae_content))
    NB_PATH.write_text(
        json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    ok("Step 12: AUTO_EMOTION_MODULE re-embedded (was stale)")
else:
    ok("Step 12: AUTO_EMOTION_MODULE byte-identical to IMTalker/auto_emotion.py")

# 2. touched code cells still compile
for cid in TOUCHED_CODE_CELLS:
    try:
        compile(src(by_id[cid]), cid, "exec")
        ok(f"{cid}: compiles")
    except KeyError:
        fail(f"{cid}: cell id not found")
    except SyntaxError as e:
        fail(f"{cid}: SyntaxError: {e}")

# 3. model id placement
all_src = "\n".join(src(c) for c in cells)
# strip the embedded module before counting the old id: the module's two
# mentions of the old model (wav2vec2 fallback comments) are intentional
sans_module = all_src.replace(ae_content, "")
n_old = sans_module.count(OLD_MODEL)
if n_old:
    fail(f"old model id {OLD_MODEL!r} still present {n_old}x outside the module")
else:
    ok("no stale old-model references outside the module")
for cid in ("emotion-stream-03", "emotion-stream-33"):
    if NEW_MODEL in src(by_id[cid]):
        ok(f"{cid}: uses {NEW_MODEL}")
    else:
        fail(f"{cid}: {NEW_MODEL} missing")

# 4. spot-check the switched content
checks = [
    ("emotion-stream-21", "pip install funasr"),
    ("emotion-stream-21", '"transformers==4.30.2"'),
    ("emotion-stream-29b-md", "~1.1 GB download"),
    ("emotion-stream-44", "funasr/transformers import"),
    ("emotion-stream-00", "emotion2vec+ SER model"),
    ("emotion-stream-03", "emotion2vec+ via funasr, 16 kHz input"),
]
for cid, needle in checks:
    if needle in src(by_id[cid]):
        ok(f"{cid}: contains {needle!r}")
    else:
        fail(f"{cid}: missing {needle!r}")

if errors:
    sys.exit(f"{len(errors)} verification error(s)")
print("[ok] all checks passed")
