#!/usr/bin/env python3
"""Switch the default SER model from the removed
audeering/wav2vec2-large-robust-finetuned-emotion (404 on HF) to the small
superb/wav2vec2-base-superb-er (95M params, IEMOCAP English, 4 classes).

  * re-embeds the updated IMTalker/auto_emotion.py into the Step 12 cell
  * replaces the model id everywhere in the notebook
  * rewrites the Parameters SER_MODEL comment (4-class default + 8-class
    ehcalabres alternative for surprise/fear coverage)
  * updates the header point-6 description and the Step 10b download size
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_PATH = ROOT / "runpod_emotion_streaming.ipynb"
AE_PATH = ROOT / "IMTalker" / "auto_emotion.py"

OLD_MODEL = "audeering/wav2vec2-large-robust-finetuned-emotion"
NEW_MODEL = "superb/wav2vec2-base-superb-er"

ae_content = AE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
if "'''" in ae_content:
    sys.exit("auto_emotion.py must not contain triple single quotes")
if NEW_MODEL not in ae_content:
    sys.exit("auto_emotion.py has not been updated to the new SER model yet")

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
c33 = c33.replace(m.group(1), ae_content)
set_src(i33, c33)
print("[ok] Step 12: AUTO_EMOTION_MODULE re-embedded")

# --- 2. swap the model id everywhere -----------------------------------------
total = 0
for i, c in enumerate(cells):
    s = src(i)
    n = s.count(OLD_MODEL)
    if n:
        set_src(i, s.replace(OLD_MODEL, NEW_MODEL))
        total += n
print(f"[ok] model id swapped in {total} place(s)")

# --- 3. Parameters cell: rewrite the SER_MODEL comment ------------------------
i3 = find_cell("emotion-stream-03")
c3 = src(i3)
old_comment = (
    "# Hugging Face SER model (wav2vec2 classification head, 16 kHz input).\n"
    f"# {NEW_MODEL}: 7 classes trained on\n"
    "# MSP-Podcast (real conversational speech): angry, disgust, fear, happy,\n"
    "# neutral, sad, surprise - covers all 6 avatar emotions (disgust -> neutral).\n"
)
new_comment = (
    "# Hugging Face SER model (wav2vec2 classification head, 16 kHz input).\n"
    "# Default: superb/wav2vec2-base-superb-er - only ~95M params (~360MB),\n"
    "# trained on IEMOCAP (English conversational speech): 4 classes\n"
    "# neu/hap/ang/sad -> neutral/happy/angry/sad. ~10-25ms per utterance on the\n"
    "# 5090 - zero streaming impact. (surprise/fear are not auto-detected with\n"
    "# this small model; for all 6 avatar emotions use the ~1.3GB alternative:\n"
    "#   ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition\n"
    "#   8 classes: angry, calm, disgust, fearful, happy, neutral, sad, surprised)\n"
)
c3 = replace_once(c3, old_comment, new_comment, "Parameters SER_MODEL comment")
set_src(i3, c3)
print("[ok] Parameters: SER_MODEL comment rewritten")

# --- 4. header point 6: update the model description --------------------------
i0 = find_cell("emotion-stream-00")
c0 = src(i0)
old_p6 = (
    "   wav2vec2 SER model (`superb/wav2vec2-base-superb-er`,\n"
    "   7 classes covering all 6 avatar emotions) and publishes the detected emotion\n"
)
new_p6 = (
    "   wav2vec2 SER model (`superb/wav2vec2-base-superb-er`, ~95M params,\n"
    "   IEMOCAP English: neutral/happy/angry/sad) and publishes the detected emotion\n"
)
c0 = replace_once(c0, old_p6, new_p6, "header point 6 model description")
set_src(i0, c0)
print("[ok] header: point 6 updated")

# --- 5. Step 10b md: download size --------------------------------------------
i10b = find_cell("emotion-stream-29b-md")
c10b = src(i10b)
c10b = replace_once(c10b, "~1.3 GB download", "~360 MB download", "Step 10b size")
set_src(i10b, c10b)
print("[ok] Step 10b: download size updated")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[ok] notebook written: {NB_PATH}")
