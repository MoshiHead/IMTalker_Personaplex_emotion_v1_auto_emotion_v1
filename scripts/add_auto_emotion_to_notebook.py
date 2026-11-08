#!/usr/bin/env python3
"""One-shot surgery: add AUTO_EMOTION (EMOTION_LIVE_PATCH_V3) support to
runpod_emotion_streaming.ipynb.

What it does:
  * Parameters cell: adds AUTO_EMOTION / SER_* / VAD_* parameters
  * New Step 10b cells: pre-download the SER model into the HF cache
  * Step 12: embeds IMTalker/auto_emotion.py (AUTO_EMOTION_MODULE), extends
    SERVER_PATCHES and START_WINNER_PATCHES to V3, bumps version markers
  * Step 13: extends PATCH_CHECKS and adds an SER smoke test
  * Step 15 (env_overrides): exports the new env vars
  * Step 17: adds an AUTO_EMOTION error hint + post-health-check warning
  * Last cell: rewrites the "how to switch emotion" doc for auto mode
  * Header: documents the new automatic mode

Safety: aborts if the notebook already contains V3 changes (restore from git
to re-apply), and verifies every patch anchor occurs exactly once in the
pristine local repo files before writing anything.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
AE_PATH = ROOT / "IMTalker" / "auto_emotion.py"
SERVER_PATH = ROOT / "IMTalker" / "liveTryHeliumFrontendDequeStaticPoseFP32FM_ws_binary_AHAudioPace.py"
SH_PATH = ROOT / "IMTalker" / "start_winner_live.sh"

ae_content = AE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
if "'''" in ae_content:
    sys.exit("auto_emotion.py must not contain triple single quotes")

server_text = SERVER_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
sh_text = SH_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")

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


i33 = find_cell("emotion-stream-33")
if "EMOTION_LIVE_PATCH_V3" in src(i33):
    sys.exit("Notebook already contains EMOTION_LIVE_PATCH_V3 - restore from git to re-apply.")


def extract_payload(cell_text, name):
    m = re.search(re.escape(name) + r" = json\.loads\(r'''(.*?)'''\)", cell_text, re.DOTALL)
    if not m:
        sys.exit(f"{name} payload not found in Step 12 cell")
    return m.group(1), json.loads(m.group(1))


# ---------------------------------------------------------------------------
# Build the V3 SERVER_PATCHES
# ---------------------------------------------------------------------------

V3_ARGS = """        # EMOTION_LIVE_PATCH_V3 (auto emotion from the user's voice)
        parser.add_argument("--auto_emotion", action="store_true",
                            help="Detect the user's emotion from their voice each utterance and drive the avatar's expression automatically (--emotion_id becomes the initial/fallback emotion)")
        parser.add_argument("--ser_model", default="audeering/wav2vec2-large-robust-finetuned-emotion",
                            help="Hugging Face speech-emotion-recognition model (wav2vec2 classification head, 16 kHz input)")
        parser.add_argument("--ser_device", default="",
                            help="Device for SER inference (default: same as --device; use 'cpu' to avoid GPU contention)")
        parser.add_argument("--ser_half", action="store_true",
                            help="Run the SER model in fp16 (halves VRAM)")
        parser.add_argument("--ser_confidence_threshold", type=float, default=0.5,
                            help="Minimum softmax probability to accept a detected emotion; below it the previous emotion is kept")
        parser.add_argument("--ser_disgust_map", type=int, default=0,
                            help="Emotion id the SER 'disgust' class maps to (0=neutral; 3=angry is the closest trained class)")
        parser.add_argument("--vad_rms_threshold", type=float, default=0.02,
                            help="Energy VAD: RMS threshold for speech on the 24kHz mic signal")
        parser.add_argument("--vad_silence_sec", type=float, default=0.7,
                            help="Energy VAD: trailing silence that closes an utterance")
        parser.add_argument("--vad_min_speech_sec", type=float, default=0.5,
                            help="Energy VAD: minimum speech duration for an utterance to be classified")
        parser.add_argument("--vad_max_utterance_sec", type=float, default=12.0,
                            help="Energy VAD: longer utterances are split and classified in pieces")
"""

DATA_DICT_ANCHOR = '        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x}\n'
DATA_DICT_REPL = """        # EMOTION_LIVE_PATCH_V3: emotion_id is resolved per chunk. With
        # --auto_emotion the AutoEmotionManager's latest detection wins;
        # otherwise the fixed --emotion_id is used (V1/V2 behaviour).
        _emotion_id = int(getattr(self.args, "emotion_id", 0))
        _auto_emotion_mgr = get_auto_emotion_manager()
        if _auto_emotion_mgr is not None:
            _emotion_id = _auto_emotion_mgr.current_emotion_id(default=_emotion_id)
        data: dict = {"a_feat": feat_25.unsqueeze(0).float(), "ref_x": self.ref_x,
                      "emotion_id": _emotion_id}
"""

IMPORT_ANCHOR = "from renderer.models import IMTRenderer\n"
IMPORT_REPL = """from renderer.models import IMTRenderer

# EMOTION_LIVE_PATCH_V3: auto emotion detection from the user's voice.
# auto_emotion.py is written next to this file by the deployment notebook;
# if it is missing or fails to import, auto mode degrades to the fixed
# --emotion_id behaviour.
try:
    from auto_emotion import get_auto_emotion_manager, init_auto_emotion_manager
except Exception:  # auto_emotion.py not deployed
    def get_auto_emotion_manager():
        return None

    def init_auto_emotion_manager(args):
        return None
"""

BUILD_APP_ANCHOR = (
    "    engine: LiveHeliumFMEngine | None = LiveHeliumFMEngine(args)\n"
    "    moshi_engine: MoshiOnlyEngine | None = None\n"
)
BUILD_APP_REPL = BUILD_APP_ANCHOR + """
    # EMOTION_LIVE_PATCH_V3: load the SER model once at startup when
    # --auto_emotion is set. Any failure disables auto mode (fixed
    # --emotion_id fallback) instead of crashing the server.
    if bool(getattr(args, "auto_emotion", False)):
        init_auto_emotion_manager(args)
"""


SESSION_RESET_ANCHOR = (
    "        fm_engine = get_engine()\n"
    "        fm_engine.reset_session()\n"
    "        reply_engine = get_moshi_engine() if args.enable_moshi_reply else None\n"
    "        if reply_engine is not None:\n"
    "            reply_engine.reset_session()\n"
)
SESSION_RESET_REPL = SESSION_RESET_ANCHOR + """        # EMOTION_LIVE_PATCH_V3: fresh VAD state per browser session.
        _auto_emotion_mgr = get_auto_emotion_manager()
        if _auto_emotion_mgr is not None:
            _auto_emotion_mgr.reset_session()
"""

MIC_TAP_ANCHOR = """                while not stopped:
                    # --- Phase 1: drain all available mic audio (non-blocking) ---
                    while True:
                        try:
                            item = mic_q.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            stopped = True
                            break
                        raw_bytes, input_sr = item
                        if raw_bytes:
                            reply_engine.append_browser_pcm(
                                np.frombuffer(raw_bytes, dtype=np.int16), input_sr
                            )
"""
MIC_TAP_REPL = """                # EMOTION_LIVE_PATCH_V3: fetch the auto-emotion manager once per
                # session (None when --auto_emotion is off or the SER model
                # failed to load).
                _auto_emotion_mgr = get_auto_emotion_manager()

                while not stopped:
                    # --- Phase 1: drain all available mic audio (non-blocking) ---
                    while True:
                        try:
                            item = mic_q.get_nowait()
                        except queue.Empty:
                            break
                        if item is None:
                            stopped = True
                            break
                        raw_bytes, input_sr = item
                        if raw_bytes:
                            _pcm_i16 = np.frombuffer(raw_bytes, dtype=np.int16)
                            reply_engine.append_browser_pcm(_pcm_i16, input_sr)
                            # EMOTION_LIVE_PATCH_V3: tap the user's mic for auto
                            # emotion detection (VAD + buffering only; SER runs
                            # on a daemon thread). Never allowed to raise here.
                            if _auto_emotion_mgr is not None:
                                try:
                                    _auto_emotion_mgr.feed(_pcm_i16, int(input_sr))
                                except Exception as _ae_exc:
                                    print(f"[AUTO-EMOTION] feed failed: {_ae_exc!r}", flush=True)
"""

# ---------------------------------------------------------------------------
# Build the V3 START_WINNER_PATCHES
# ---------------------------------------------------------------------------

AUTO_EMOTION_SH = """
# AUTO_EMOTION (EMOTION_LIVE_PATCH_V3): detect the user's emotion from their
# voice each utterance and drive the avatar's expression automatically.
# EMOTION_ID above becomes the initial/fallback emotion until the first detection.
AUTO_EMOTION="${AUTO_EMOTION:-0}"
if [[ "$AUTO_EMOTION" == "1" ]]; then
  EMOTION_ARGS+=(
    --auto_emotion
    --ser_model "${SER_MODEL:-audeering/wav2vec2-large-robust-finetuned-emotion}"
    --ser_device "${SER_DEVICE:-}"
    --ser_confidence_threshold "${SER_CONFIDENCE_THRESHOLD:-0.5}"
    --ser_disgust_map "${SER_DISGUST_MAP:-0}"
    --vad_rms_threshold "${VAD_RMS_THRESHOLD:-0.02}"
    --vad_silence_sec "${VAD_SILENCE_SEC:-0.7}"
    --vad_min_speech_sec "${VAD_MIN_SPEECH_SEC:-0.5}"
    --vad_max_utterance_sec "${VAD_MAX_UTTERANCE_SEC:-12.0}"
  )
fi
"""


# ---------------------------------------------------------------------------
# Assemble and verify the new patch lists against the pristine local files
# ---------------------------------------------------------------------------

c33 = src(i33)
old_server_payload, old_server = extract_payload(c33, "SERVER_PATCHES")
old_sh_payload, old_sh = extract_payload(c33, "START_WINNER_PATCHES")
if len(old_server) != 4 or len(old_sh) != 2:
    sys.exit(f"unexpected patch counts: server={len(old_server)} sh={len(old_sh)}")

new_server = [
    [old_server[0][0], old_server[0][1] + V3_ARGS],
    old_server[1],
    old_server[2],
    [DATA_DICT_ANCHOR, DATA_DICT_REPL],
    [IMPORT_ANCHOR, IMPORT_REPL],
    [BUILD_APP_ANCHOR, BUILD_APP_REPL],
    [SESSION_RESET_ANCHOR, SESSION_RESET_REPL],
    [MIC_TAP_ANCHOR, MIC_TAP_REPL],
]
for i, (anchor, _repl) in enumerate(new_server, 1):
    n = server_text.count(anchor)
    if n != 1:
        sys.exit(f"server patch #{i} anchor found {n}x in the pristine server file (expected 1)")

sh_anchor_suffix = "\n# STT + query routing + web search have been removed from this pipeline.\n"
if not old_sh[0][1].endswith(sh_anchor_suffix):
    sys.exit("START_WINNER patch #1 replacement does not end with the expected anchor line")
new_sh_repl = old_sh[0][1][:-len(sh_anchor_suffix)] + AUTO_EMOTION_SH + sh_anchor_suffix
new_sh = [[old_sh[0][0], new_sh_repl], old_sh[1]]
for i, (anchor, _repl) in enumerate(new_sh, 1):
    n = sh_text.count(anchor)
    if n != 1:
        sys.exit(f"sh patch #{i} anchor found {n}x in the pristine start_winner_live.sh (expected 1)")

print("[ok] all V3 patch anchors verified unique against the pristine local files")

# ---------------------------------------------------------------------------
# Step 12 cell: embed auto_emotion.py, swap patch payloads, bump versions
# ---------------------------------------------------------------------------

ae_block = (
    "\n# ---------------------------------------------------------------------------\n"
    "# auto_emotion.py - automatic user-voice emotion detection (V3). Written\n"
    "# next to the live server; the server imports it inside try/except, so a\n"
    "# missing file degrades to the fixed --emotion_id behaviour.\n"
    "# ---------------------------------------------------------------------------\n"
    "AUTO_EMOTION_MODULE = r'''" + ae_content + "'''\n"
    "\n"
    'ae_path = Path(IMTALKER_DIR) / "auto_emotion.py"\n'
    'if ae_path.exists() and ae_path.read_text(encoding="utf-8") == AUTO_EMOTION_MODULE:\n'
    '    print("[skip] auto_emotion.py already up to date")\n'
    "else:\n"
    '    ae_path.write_text(AUTO_EMOTION_MODULE, encoding="utf-8")\n'
    '    print(f"[written] {ae_path} ({len(AUTO_EMOTION_MODULE)} chars)")\n'
)
c33 = replace_once(
    c33,
    'START_WINNER_SH = Path(IMTALKER_DIR) / "start_winner_live.sh"\n',
    'START_WINNER_SH = Path(IMTALKER_DIR) / "start_winner_live.sh"\n' + ae_block,
    "auto_emotion module block",
)
c33 = replace_once(c33, old_server_payload, json.dumps(new_server), "SERVER_PATCHES payload")
c33 = replace_once(c33, old_sh_payload, json.dumps(new_sh), "START_WINNER_PATCHES payload")
c33 = replace_once(
    c33,
    'apply_patches(SERVER_FILE, SERVER_PATCHES, "EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V2")',
    'apply_patches(SERVER_FILE, SERVER_PATCHES, "EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V3")',
    "server apply_patches call",
)
c33 = replace_once(
    c33,
    "#    (EMOTION_LIVE_PATCH_V2 adds --emotion_cfg_scale)",
    "#    (EMOTION_LIVE_PATCH_V3 adds --auto_emotion, the mic tap and per-chunk emotion_id)",
    "server patch comment",
)
c33 = replace_once(
    c33,
    'apply_patches(START_WINNER_SH, START_WINNER_PATCHES, "EMOTION_ARGS", "EMOTION_CFG_SCALE")',
    'apply_patches(START_WINNER_SH, START_WINNER_PATCHES, "EMOTION_ARGS", "AUTO_EMOTION")',
    "start_winner apply_patches call",
)
c33 = replace_once(
    c33,
    "# 5. start_winner_live.sh — EMOTION_ARGS from env vars (EMOTION_CFG_SCALE added)",
    "# 5. start_winner_live.sh — EMOTION_ARGS from env vars (+ AUTO_EMOTION args)",
    "start_winner patch comment",
)
c33 = replace_once(
    c33,
    'for py in [GEN_DIR / "FM.py", GEN_DIR / "FMT.py", GEN_DIR / "train_lora.py", SERVER_FILE]:',
    'for py in [GEN_DIR / "FM.py", GEN_DIR / "FMT.py", GEN_DIR / "train_lora.py", SERVER_FILE,\n'
    '           Path(IMTALKER_DIR) / "auto_emotion.py"]:',
    "syntax check file list",
)
set_src(i33, c33)
print("[ok] Step 12 cell updated (V3 patches + auto_emotion.py writer)")


# ---------------------------------------------------------------------------
# Parameters cell: AUTO_EMOTION / SER_* / VAD_* parameters
# ---------------------------------------------------------------------------

AUTO_EMOTION_PARAMS = """
# --- AUTO_EMOTION - detect the user's emotion from their voice automatically ---
# When True, a speech-emotion-recognition (SER) model listens to the user's mic,
# classifies each utterance, and switches the avatar's expression automatically
# (~1s after the utterance ends). EMOTION above becomes the INITIAL/FALLBACK
# emotion (used until the first confident detection, and kept whenever a
# detection is uncertain). Set EMOTION = "neutral" for the most natural start.
AUTO_EMOTION = True

# Hugging Face SER model (wav2vec2 classification head, 16 kHz input).
# audeering/wav2vec2-large-robust-finetuned-emotion: 7 classes trained on
# MSP-Podcast (real conversational speech): angry, disgust, fear, happy,
# neutral, sad, surprise - covers all 6 avatar emotions (disgust -> neutral).
SER_MODEL = "audeering/wav2vec2-large-robust-finetuned-emotion"
# Device for SER inference. "" = same GPU as the avatar (recommended on the
# 5090: ~50-100ms per utterance on a daemon thread, no streaming impact).
# "cpu" = zero GPU contention at ~0.5-2s per utterance.
SER_DEVICE = ""
# Minimum softmax confidence to accept a detection; below it the previous
# emotion is kept (prevents flicker on ambiguous/short utterances).
SER_CONFIDENCE_THRESHOLD = 0.5
# Energy-VAD tuning (defaults work for a typical headset/laptop mic; watch the
# server's [MIC] rms= logs if utterances are missed or over-split).
VAD_RMS_THRESHOLD = 0.02      # speech onset/offset RMS on the 24kHz mic signal
VAD_SILENCE_SEC = 0.7         # trailing silence that closes an utterance
VAD_MIN_SPEECH_SEC = 0.5      # shorter utterances are ignored (coughs, "yes")
VAD_MAX_UTTERANCE_SEC = 12.0  # longer utterances are classified in pieces
"""

i3 = find_cell("emotion-stream-03")
c3 = src(i3)
c3 = replace_once(c3, "EMOTION_CFG_SCALE = 3.0\n", "EMOTION_CFG_SCALE = 3.0\n" + AUTO_EMOTION_PARAMS,
                  "AUTO_EMOTION parameters")
old_emotion_print = None
for line in c3.split("\n"):
    if "fixed for the whole session" in line:
        old_emotion_print = line
        break
if old_emotion_print is None:
    sys.exit("EMOTION print line not found in Parameters cell")
new_emotion_print = (
    'print(f"EMOTION      = {EMOTION} (id={EMOTION_ID})" + ("\u2014 initial/fallback (AUTO_EMOTION on)" if AUTO_EMOTION else "\u2014 fixed for the whole session"))\n'
    'if AUTO_EMOTION:\n'
    '    print(f"AUTO_EMOTION = ON  (SER={SER_MODEL}, conf>={SER_CONFIDENCE_THRESHOLD}, device={SER_DEVICE or \'same as avatar\'})")'
)
c3 = replace_once(c3, old_emotion_print, new_emotion_print, "EMOTION print line")
set_src(i3, c3)
print("[ok] Parameters cell updated")

# ---------------------------------------------------------------------------
# New Step 10b cells: pre-download the SER model into the HF cache
# ---------------------------------------------------------------------------

md_10b = {
    "cell_type": "markdown",
    "id": "emotion-stream-29b-md",
    "metadata": {},
    "source": [
        "## Step 10b — Pre-download the auto-emotion SER model\n",
        "\n",
        "Downloads the speech-emotion-recognition model into the Hugging Face cache\n",
        "so the server's first startup doesn't stall on a ~1.3 GB download. Skipped\n",
        "when `AUTO_EMOTION = False`.\n",
    ],
}
code_10b = {
    "cell_type": "code",
    "execution_count": None,
    "id": "emotion-stream-29b",
    "metadata": {},
    "outputs": [],
    "source": [
        "if AUTO_EMOTION:\n",
        "    run(f'{VENV_ACTIVATE} && export HF_HUB_ENABLE_HF_TRANSFER=1 && hf download {SER_MODEL}',\n",
        "        retries=3, retry_delay=15)\n",
        "    print(f\"[ok] SER model cached: {SER_MODEL}\")\n",
        "else:\n",
        "    print(\"[skip] AUTO_EMOTION is False - no SER model to download\")\n",
    ],
}
i29 = find_cell("emotion-stream-29")
cells[i29 + 1:i29 + 1] = [md_10b, code_10b]
print("[ok] Step 10b cells inserted (SER model pre-download)")


# ---------------------------------------------------------------------------
# Step 13 cell: extend PATCH_CHECKS + SER smoke test
# ---------------------------------------------------------------------------

OLD_SERVER_CHECKS = """    (SERVER_FILE, ["EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V2", "--emotion_id",
                   "--emotion_lora_path", "--emotion_cfg_scale",
                   "apply_emotion_lora_to_model", "emotion lora loaded",
                   '"emotion_id": int(getattr(self.args, "emotion_id", 0))']),"""
NEW_SERVER_CHECKS = """    (SERVER_FILE, ["EMOTION_LIVE_PATCH_V1", "EMOTION_LIVE_PATCH_V2", "EMOTION_LIVE_PATCH_V3",
                   "--emotion_id", "--emotion_lora_path", "--emotion_cfg_scale",
                   "--auto_emotion", "get_auto_emotion_manager", "init_auto_emotion_manager",
                   "apply_emotion_lora_to_model", "emotion lora loaded",
                   "_auto_emotion_mgr.feed(_pcm_i16, int(input_sr))"]),"""
OLD_SH_CHECKS = """    (START_WINNER_SH, ["EMOTION_ARGS", '--emotion_id "$EMOTION_ID"', '"${EMOTION_ARGS[@]}"',
                       "EMOTION_CFG_SCALE", '--emotion_cfg_scale "$EMOTION_CFG_SCALE"']),"""
NEW_SH_CHECKS = """    (START_WINNER_SH, ["EMOTION_ARGS", '--emotion_id "$EMOTION_ID"', '"${EMOTION_ARGS[@]}"',
                       "EMOTION_CFG_SCALE", '--emotion_cfg_scale "$EMOTION_CFG_SCALE"',
                       "AUTO_EMOTION", "--auto_emotion"]),
    (Path(IMTALKER_DIR) / "auto_emotion.py",
     ["class AutoEmotionManager", "class EnergyVAD", "class SpeechEmotionClassifier",
      "get_auto_emotion_manager", "init_auto_emotion_manager"]),"""

SER_SMOKE_TEST = '''
# --- 3. SER smoke test (auto emotion) ---------------------------------------
if AUTO_EMOTION:
    SER_SMOKE_SRC = r\'\'\'
import sys
import numpy as np

sys.path.insert(0, sys.argv[1])
from auto_emotion import SpeechEmotionClassifier

clf = SpeechEmotionClassifier(model_name=sys.argv[2], device="cpu", half=False, disgust_map=0)
rng = np.random.default_rng(0)
# 1s of synthetic speech-like noise: verifies model load + forward + label map
pcm = (0.05 * rng.standard_normal(24000)).astype(np.float32)
emotion_id, label, conf = clf.predict(pcm)
print(f"smoke: label={label} emotion_id={emotion_id} conf={conf:.3f}")
assert 0 <= int(emotion_id) <= 5, f"invalid emotion_id {emotion_id}"
print("[ok] SER smoke test passed")
\'\'\'
    smoke_path = Path("/tmp/ser_smoke_test.py")
    smoke_path.write_text(SER_SMOKE_SRC, encoding="utf-8")
    proc = subprocess.run(
        [VENV_PYTHON, str(smoke_path), str(IMTALKER_DIR), SER_MODEL],
        capture_output=True, text=True, timeout=600,
    )
    print(proc.stdout)
    assert proc.returncode == 0, proc.stderr[-2000:]
    print(f"[ok] auto-emotion SER model validated: {SER_MODEL}")
else:
    print("[skip] AUTO_EMOTION is False - SER smoke test skipped")
'''

i35 = find_cell("emotion-stream-35")
c35 = src(i35)
c35 = replace_once(c35, OLD_SERVER_CHECKS, NEW_SERVER_CHECKS, "SERVER_FILE patch checks")
c35 = replace_once(c35, OLD_SH_CHECKS, NEW_SH_CHECKS, "START_WINNER_SH patch checks")
c35 = c35.rstrip("\n") + "\n" + SER_SMOKE_TEST
set_src(i35, c35)
print("[ok] Step 13 cell updated (patch checks + SER smoke test)")

# ---------------------------------------------------------------------------
# Step 15 cell (env_overrides): export the new env vars
# ---------------------------------------------------------------------------

ENV_ADDITIONS = """    # AUTO_EMOTION (V3) - voice-based automatic emotion detection.
    "AUTO_EMOTION": "1" if AUTO_EMOTION else "0",
    "SER_MODEL": SER_MODEL,
    "SER_DEVICE": SER_DEVICE,
    "SER_CONFIDENCE_THRESHOLD": str(SER_CONFIDENCE_THRESHOLD),
    "VAD_RMS_THRESHOLD": str(VAD_RMS_THRESHOLD),
    "VAD_SILENCE_SEC": str(VAD_SILENCE_SEC),
    "VAD_MIN_SPEECH_SEC": str(VAD_MIN_SPEECH_SEC),
    "VAD_MAX_UTTERANCE_SEC": str(VAD_MAX_UTTERANCE_SEC),
"""

i40 = find_cell("emotion-stream-40")
c40 = src(i40)
c40 = replace_once(c40, '    "EMOTION_CFG_SCALE": str(EMOTION_CFG_SCALE),\n',
                   '    "EMOTION_CFG_SCALE": str(EMOTION_CFG_SCALE),\n' + ENV_ADDITIONS,
                   "env_overrides additions")
set_src(i40, c40)
print("[ok] env_overrides cell updated")


# ---------------------------------------------------------------------------
# Step 17 cell: AUTO_EMOTION error hint + post-health-check warning
# ---------------------------------------------------------------------------

i44 = find_cell("emotion-stream-44")
c44 = src(i44)
old_hint_line = None
for line in c44.split("\n"):
    if '"anchor found":' in line:
        old_hint_line = line
        break
if old_hint_line is None:
    sys.exit("ERROR_HINTS 'anchor found' line not found in Step 17 cell")
new_hint_line = old_hint_line + "\n" + (
    '    "[AUTO-EMOTION] DISABLED": "the SER model failed to load - the server is running with the fixed EMOTION fallback; check the traceback above (HF download, VRAM, transformers version)",'
)
c44 = replace_once(c44, old_hint_line, new_hint_line, "ERROR_HINTS addition")
c44 = replace_once(
    c44,
    "wait_for_healthy()",
    """wait_for_healthy()

# Auto-emotion sanity: warn (don't fail) if the SER model didn't come up.
if AUTO_EMOTION and os.path.exists(LOG_PATH):
    with open(LOG_PATH, "r", errors="ignore") as f:
        _log = f.read()
    if "[AUTO-EMOTION] DISABLED" in _log:
        print("[warn] AUTO_EMOTION is on but the SER model failed to load - the avatar will use")
        print("       the fixed EMOTION fallback. See '[AUTO-EMOTION] DISABLED' in the log above.")
    elif "[AUTO-EMOTION] enabled:" in _log:
        print("[ok] auto-emotion SER model is active (voice-driven expressions)")""",
    "post-health-check auto-emotion warning",
)
set_src(i44, c44)
print("[ok] Step 17 cell updated")

# ---------------------------------------------------------------------------
# Final cell: rewrite the "how to switch emotion" doc for auto mode
# ---------------------------------------------------------------------------

CELL51_NEW = '''# How emotion control works with AUTO_EMOTION
# --------------------------------------------
# AUTO_EMOTION = True (default): the avatar's expression is driven by the USER'S
# VOICE. Each user utterance is classified by the SER model and the face
# switches automatically (~1s after the utterance ends). EMOTION in the
# Parameters cell is only the initial/fallback emotion - no restart needed,
# no manual selection, works for the whole conversation.
#
# AUTO_EMOTION = False: the old manual behaviour - EMOTION is baked in at
# launch. To change it: run the stop cell, edit EMOTION in Parameters, re-run
# Parameters, then re-run Launch (Step 16).
NEW_EMOTION = "sad"  # manual mode only: "neutral", "happy", "sad", "angry", "surprise", "fear"

print(f"AUTO_EMOTION = {AUTO_EMOTION}")
if AUTO_EMOTION:
    print("The avatar mirrors the user's detected voice emotion automatically.")
    print(f"Initial/fallback emotion: {EMOTION} (id={EMOTION_ID})")
    print(f"SER model: {SER_MODEL} | confidence threshold: {SER_CONFIDENCE_THRESHOLD}")
    print("Watch the server log for lines like:")
    print("  [AUTO-EMOTION] utterance 2.31s -> happy (p=0.87) [neutral -> happy] switching")
else:
    print(f"Currently running: EMOTION={EMOTION} (id={EMOTION_ID}) - fixed for the session")
    print()
    print(f"To switch to '{NEW_EMOTION}':")
    print(f"  1. Run the STOP SERVER cell above and wait for the port to free")
    print(f"  2. In the Parameters cell, set:  EMOTION = \\"{NEW_EMOTION}\\"")
    print(f"  3. Re-run the Parameters cell, then the Launch cell (Step 16)")
print()
print("Emotion reference:")
for _name, _id in EMOTION_MAP.items():
    marker = "  <-- initial" if _id == EMOTION_ID else ""
    print(f"  {_id}: {_name}{marker}")
print()
print(f"Expression strength: EMOTION_CFG_SCALE={EMOTION_CFG_SCALE} (edit in Parameters;")
print("  raise for a stronger expression, lower toward 1.0 for subtler, then re-launch)")
'''

i51 = find_cell("emotion-stream-51")
set_src(i51, CELL51_NEW)
print("[ok] final doc cell rewritten")


# ---------------------------------------------------------------------------
# Header cell: document the automatic mode
# ---------------------------------------------------------------------------

i0 = find_cell("emotion-stream-00")
c0 = src(i0)
old_intro = (
    "**Select the emotion before launching** (Parameters cell): `happy`, `sad`,\n"
    "`angry`, `surprise`, `fear` (or `neutral`). The choice is baked into the GPU\n"
    "weights at startup and never changes mid-conversation.\n"
)
new_intro = (
    "**Emotion can now be fully automatic** (Parameters cell): with `AUTO_EMOTION = True`\n"
    "(default), a speech-emotion-recognition model listens to the **user's voice**,\n"
    "classifies every utterance, and switches the avatar's expression automatically\n"
    "throughout the conversation — no manual selection needed. `EMOTION` is only the\n"
    "initial/fallback emotion. With `AUTO_EMOTION = False` the old behaviour applies:\n"
    "the emotion is fixed at launch (`happy`, `sad`, `angry`, `surprise`, `fear` or\n"
    "`neutral`) and never changes mid-conversation.\n"
)
c0 = replace_once(c0, old_intro, new_intro, "header intro paragraph")
point6 = (
    "6. **Automatic emotion (V3, optional).** Step 12 also writes `IMTalker/auto_emotion.py`\n"
    "   and patches the live server (`--auto_emotion` args, a mic tap in the GPU\n"
    "   producer thread, per-chunk `emotion_id` resolution). An energy VAD segments\n"
    "   the user's mic into utterances; a daemon thread classifies each one with a\n"
    "   wav2vec2 SER model (`audeering/wav2vec2-large-robust-finetuned-emotion`,\n"
    "   7 classes covering all 6 avatar emotions) and publishes the detected emotion\n"
    "   id, which the next 0.96s FM chunk picks up. Detection runs off the hot path,\n"
    "   so audio/video streaming is untouched; any failure falls back to the fixed\n"
    "   `--emotion_id` behaviour.\n"
)
c0 = replace_once(c0, "**Emotion adapter:**", point6 + "\n**Emotion adapter:**",
                  "header point 6")
set_src(i0, c0)
print("[ok] header cell updated")

# ---------------------------------------------------------------------------
# Write the notebook back (1-space indent, LF, trailing newline - as before)
# ---------------------------------------------------------------------------

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[ok] notebook written: {NB_PATH}")
print("DONE - run scripts/verify_auto_emotion_notebook.py next")

