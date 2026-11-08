#!/usr/bin/env python3
"""Apply the V3 patches to a temp copy of the server + start_winner_live.sh and
print the patched hunks for visual inspection."""

import json
import re
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
SERVER_PATH = ROOT / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
SH_PATH = ROOT / "IMTalker" / "start_winner_live.sh"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
c33 = "".join(next(c for c in nb["cells"] if c.get("id") == "emotion-stream-33")["source"])


def extract(name):
    m = re.search(re.escape(name) + r" = json\.loads\(r'''(.*?)'''\)", c33, re.DOTALL)
    return json.loads(m.group(1))


server_patches = extract("SERVER_PATCHES")
sh_patches = extract("START_WINNER_PATCHES")

tmp = Path(tempfile.mkdtemp(prefix="patched_hunks_"))
server_copy = tmp / SERVER_PATH.name
sh_copy = tmp / SH_PATH.name
shutil.copy(SERVER_PATH, server_copy)
shutil.copy(SH_PATH, sh_copy)

text = server_copy.read_text(encoding="utf-8")
for anchor, repl in server_patches:
    assert text.count(anchor) == 1
    text = text.replace(anchor, repl)
server_copy.write_text(text, encoding="utf-8")

shtext = sh_copy.read_text(encoding="utf-8")
for anchor, repl in sh_patches:
    assert shtext.count(anchor) == 1
    shtext = shtext.replace(anchor, repl)
sh_copy.write_text(shtext, encoding="utf-8")

lines = text.split("\n")


def show(title, needle, before=3, after=14):
    idx = next(i for i, l in enumerate(lines) if needle in l)
    print(f"\n{'=' * 78}\n{title} (line {idx + 1})\n{'=' * 78}")
    for i in range(max(0, idx - before), min(len(lines), idx + after)):
        print(f"{i + 1:5d} | {lines[i]}")


show("IMPORT PATCH", "from auto_emotion import")
show("ARGS PATCH (V3 part)", 'parser.add_argument("--auto_emotion"', before=2, after=8)
show("DYNAMIC emotion_id", "_auto_emotion_mgr.current_emotion_id")
show("BUILD_APP INIT", 'if bool(getattr(args, "auto_emotion", False)):', before=4, after=3)
show("MANAGER FETCH BEFORE LOOP", "fetch the auto-emotion manager once per", before=2, after=6)
show("SESSION RESET", "_auto_emotion_mgr.reset_session()")
show("MIC TAP", "_auto_emotion_mgr.feed(_pcm_i16, int(input_sr))", before=8, after=6)

shlines = shtext.split("\n")
idx = next(i for i, l in enumerate(shlines) if "AUTO_EMOTION=" in l)
print(f"\n{'=' * 78}\nSTART_WINNER AUTO_EMOTION BLOCK (line {idx + 1})\n{'=' * 78}")
for i in range(max(0, idx - 4), min(len(shlines), idx + 18)):
    print(f"{i + 1:5d} | {shlines[i]}")

shutil.rmtree(tmp, ignore_errors=True)
