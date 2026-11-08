#!/usr/bin/env python3
"""Verify the AUTO_EMOTION (V3) notebook changes end-to-end, locally.

  1. Notebook JSON is well-formed; every code cell compiles (except the
     !git clone shell cell).
  2. The AUTO_EMOTION_MODULE embedded in Step 12 matches IMTalker/auto_emotion.py
     and compiles.
  3. The V3 SERVER_PATCHES / START_WINNER_PATCHES apply cleanly to COPIES of
     the pristine local repo files (every anchor exactly once), the patched
     server compiles, and all expected markers are present.
  4. The apply_patches re-patch path works: applying V3 a second time restores
     from the pre-emotion backup and re-applies cleanly (idempotent).
"""

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
AE_PATH = ROOT / "IMTalker" / "auto_emotion.py"
SERVER_PATH = ROOT / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
SH_PATH = ROOT / "IMTalker" / "start_winner_live.sh"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
cells = nb["cells"]
print(f"[ok] notebook JSON valid: {len(cells)} cells")


def src(i):
    return "".join(cells[i]["source"])


def find_cell(cell_id):
    for i, c in enumerate(cells):
        if c.get("id") == cell_id:
            return i
    raise KeyError(cell_id)


# --- 1. compile every code cell ---------------------------------------------
for i, c in enumerate(cells):
    if c.get("cell_type") != "code":
        continue
    s = src(i)
    if s.lstrip().startswith("!"):
        continue  # shell cell
    try:
        compile(s, f"cell[{i}] id={c.get('id')}", "exec")
    except SyntaxError as exc:
        sys.exit(f"cell[{i}] id={c.get('id')} does not compile: {exc}")
print("[ok] all code cells compile")

# --- 2. AUTO_EMOTION_MODULE matches the repo file and compiles ---------------
c33 = src(find_cell("emotion-stream-33"))
m = re.search(r"AUTO_EMOTION_MODULE = r'''(.*?)'''", c33, re.DOTALL)
if not m:
    sys.exit("AUTO_EMOTION_MODULE not found in Step 12 cell")
module_src = m.group(1)
repo_src = AE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
if module_src != repo_src:
    sys.exit("AUTO_EMOTION_MODULE in the notebook differs from IMTalker/auto_emotion.py")
compile(module_src, "auto_emotion.py", "exec")
print("[ok] AUTO_EMOTION_MODULE matches IMTalker/auto_emotion.py and compiles")


def extract_payload(cell_text, name):
    m = re.search(re.escape(name) + r" = json\.loads\(r'''(.*?)'''\)", cell_text, re.DOTALL)
    if not m:
        sys.exit(f"{name} payload not found")
    return json.loads(m.group(1))


server_patches = extract_payload(c33, "SERVER_PATCHES")
sh_patches = extract_payload(c33, "START_WINNER_PATCHES")
print(f"[ok] patch payloads parse: {len(server_patches)} server, {len(sh_patches)} sh patches")


# --- 3. replicate the notebook's apply_patches --------------------------------
def apply_patches(path, patch_list, marker, version_marker=None):
    text = path.read_text(encoding="utf-8")
    if marker in text:
        if version_marker is None or version_marker in text:
            return False
        backup = path.with_suffix(path.suffix + ".bak_pre_emotion")
        if backup.exists():
            text = backup.read_text(encoding="utf-8")
    for i, (anchor, repl) in enumerate(patch_list, 1):
        n = text.count(anchor)
        if n != 1:
            raise RuntimeError(f"anchor found {n}x (expected 1) in {path.name} patch #{i}")
        text = text.replace(anchor, repl)
    backup = path.with_suffix(path.suffix + ".bak_pre_emotion")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    path.write_text(text, encoding="utf-8")
    return True


tmp = Path(tempfile.mkdtemp(prefix="auto_emotion_verify_"))
try:
    server_copy = tmp / SERVER_PATH.name
    sh_copy = tmp / SH_PATH.name
    shutil.copy(SERVER_PATH, server_copy)
    shutil.copy(SH_PATH, sh_copy)

    apply_patches(server_copy, server_patches, "EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V3")
    apply_patches(sh_copy, sh_patches, "EMOTION_ARGS", "AUTO_EMOTION")

    patched = server_copy.read_text(encoding="utf-8")
    compile(patched, str(server_copy), "exec")
    for needle in ["EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V2", "EMOTION_LIVE_PATCH_V3",
                   "--auto_emotion", "get_auto_emotion_manager", "init_auto_emotion_manager",
                   "_auto_emotion_mgr.feed(_pcm_i16, int(input_sr))",
                   "_emotion_id = _auto_emotion_mgr.current_emotion_id(default=_emotion_id)",
                   "apply_emotion_lora_to_model"]:
        if needle not in patched:
            sys.exit(f"patched server missing marker: {needle}")
    patched_sh = sh_copy.read_text(encoding="utf-8")
    for needle in ["EMOTION_ARGS", "AUTO_EMOTION", "--auto_emotion", "--ser_model",
                   "--vad_rms_threshold", '"${EMOTION_ARGS[@]}"']:
        if needle not in patched_sh:
            sys.exit(f"patched start_winner_live.sh missing marker: {needle}")
    print("[ok] V3 patches apply cleanly to pristine copies; patched server compiles")

    # --- 4. re-patch idempotency (restore-from-backup path) -------------------
    again = apply_patches(server_copy, server_patches, "EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V3")
    if again:
        sys.exit("second apply_patches should have been skipped (already V3)")
    # simulate a V2-patched file (V3 marker absent) -> must restore + re-apply
    text_v2ish = patched.replace("EMOTION_LIVE_PATCH_V3", "EMOTION_LIVE_PATCH_V2")
    server_copy.write_text(text_v2ish, encoding="utf-8")
    if not apply_patches(server_copy, server_patches, "EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V3"):
        sys.exit("re-patch from backup did not trigger")
    repatched = server_copy.read_text(encoding="utf-8")
    if repatched != patched:
        sys.exit("re-patched server differs from the first V3 patch result")
    compile(repatched, str(server_copy), "exec")
    print("[ok] re-patch from backup reproduces the identical V3 result")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("ALL CHECKS PASSED")

