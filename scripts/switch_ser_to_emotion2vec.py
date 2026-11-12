#!/usr/bin/env python3
"""Switch the default SER model from superb/wav2vec2-base-superb-er to
emotion2vec/emotion2vec_plus_base (funasr AutoModel, 9 classes, ~90M params).

  * re-embeds the updated IMTalker/auto_emotion.py into the Step 12 cell
  * rewrites the Parameters SER_MODEL / SER_DEVICE comments
  * updates the header point-6 description and the Step 10b download size
  * installs funasr in Step 6 and re-asserts the project pins afterwards
  * updates the Step 12 --ser_model help text and the Step 17 ERROR_HINTS
  * swaps the model id everywhere else (argparse + start_winner_live.sh
    defaults, the SER_MODEL assignment)
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

ae_content = AE_PATH.read_text(encoding="utf-8").replace("\r\n", "\n")
if "'''" in ae_content:
    sys.exit("auto_emotion.py must not contain triple single quotes")
if NEW_MODEL not in ae_content:
    sys.exit("auto_emotion.py has not been updated to the emotion2vec model yet")

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


# --- 1. Parameters cell: rewrite the SER_MODEL / SER_DEVICE comments ---------
i3 = find_cell("emotion-stream-03")
c3 = src(i3)
old_comment = (
    "# Hugging Face SER model (wav2vec2 classification head, 16 kHz input).\n"
    "# Default: superb/wav2vec2-base-superb-er - only ~95M params (~360MB),\n"
    "# trained on IEMOCAP (English conversational speech): 4 classes\n"
    "# neu/hap/ang/sad -> neutral/happy/angry/sad. ~10-25ms per utterance on the\n"
    "# 5090 - zero streaming impact. (surprise/fear are not auto-detected with\n"
    "# this small model; for all 6 avatar emotions use the ~1.3GB alternative:\n"
    "#   ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition\n"
    "#   8 classes: angry, calm, disgust, fearful, happy, neutral, sad, surprised)\n"
)
new_comment = (
    "# Hugging Face SER model (emotion2vec+ via funasr, 16 kHz input).\n"
    "# Default: emotion2vec/emotion2vec_plus_base - ~90M params (~1.1GB checkpoint),\n"
    "# trained on ~4.8k hours of pseudo-labelled speech: 9 classes\n"
    "# angry/disgusted/fearful/happy/neutral/other/sad/surprised/unknown ->\n"
    "# covers all 6 avatar emotions (disgust maps to --ser_disgust_map;\n"
    "# other/unknown fall back to neutral). ~20-50ms per utterance on the\n"
    "# 5090 - zero streaming impact. Requires the funasr package (installed in\n"
    "# Step 6). Wav2Vec2 classification-head repos remain supported as\n"
    "# alternatives (auto_emotion.py detects them from the model name).\n"
)
c3 = replace_once(c3, old_comment, new_comment, "Parameters SER_MODEL comment")
c3 = replace_once(
    c3,
    "# ~10-25ms per utterance on a daemon thread, so the emotion switch lands ~1s",
    "# ~20-50ms per utterance on a daemon thread, so the emotion switch lands ~1s",
    "Parameters SER_DEVICE latency",
)
c3 = replace_once(
    c3,
    "# NOT recommended on a shared-CPU pod: wav2vec2 inference takes seconds there",
    "# NOT recommended on a shared-CPU pod: SER inference takes seconds there",
    "Parameters SER_DEVICE cpu warning",
)
set_src(i3, c3)
print("[ok] Parameters: SER_MODEL / SER_DEVICE comments rewritten")

# --- 2. header point 6: update the model description --------------------------
i0 = find_cell("emotion-stream-00")
c0 = src(i0)
old_p6 = (
    "   the user's mic into utterances; a daemon thread classifies each one with a\n"
    "   wav2vec2 SER model (`superb/wav2vec2-base-superb-er`, ~95M params,\n"
    "   IEMOCAP English: neutral/happy/angry/sad) and publishes the detected emotion\n"
)
new_p6 = (
    "   the user's mic into utterances; a daemon thread classifies each one with the\n"
    "   emotion2vec+ SER model (`emotion2vec/emotion2vec_plus_base`, ~90M params,\n"
    "   9 classes covering all 6 avatar emotions) and publishes the detected emotion\n"
)
c0 = replace_once(c0, old_p6, new_p6, "header point 6 model description")
set_src(i0, c0)
print("[ok] header: point 6 updated")

# --- 3. Step 6: install funasr and re-assert the project pins ------------------
i21 = find_cell("emotion-stream-21")
c21 = src(i21)
old_install = (
    "    f'{VENV_ACTIVATE} && pip install \"sphn>=0.2.0,<0.3.0\" einops "
    "sentencepiece aiohttp av aiortc bitsandbytes',\n"
    "    retries=2, retry_delay=15,\n"
    ")\n"
    "assert_torch_pin()\n"
)
new_install = (
    "    f'{VENV_ACTIVATE} && pip install \"sphn>=0.2.0,<0.3.0\" einops "
    "sentencepiece aiohttp av aiortc bitsandbytes',\n"
    "    retries=2, retry_delay=15,\n"
    ")\n"
    "# funasr: runtime for the emotion2vec SER model (AUTO_EMOTION; Step 10b / 12).\n"
    "# funasr does not depend on torch, so the cu128 pin is safe; its other deps\n"
    "# (modelscope, umap_learn, ...) are unpinned, so the project pins are\n"
    "# re-asserted afterwards in case pip resolved anything newer.\n"
    'run(f"{VENV_ACTIVATE} && pip install funasr", retries=2, retry_delay=15)\n'
    "run(\n"
    "    f'{VENV_ACTIVATE} && pip install \"numpy<2.0.0\" \"transformers==4.30.2\" '\n"
    "    f'\"pandas<2.2.0\" \"matplotlib<3.9.0\"',\n"
    "    retries=2, retry_delay=15,\n"
    ")\n"
    "assert_torch_pin()\n"
)
c21 = replace_once(c21, old_install, new_install, "Step 6 funasr install")
set_src(i21, c21)
print("[ok] Step 6: funasr install added (project pins re-asserted)")

# --- 4. Step 10b md: download size ---------------------------------------------
i10b = find_cell("emotion-stream-29b-md")
c10b = src(i10b)
c10b = replace_once(c10b, "~360 MB download", "~1.1 GB download", "Step 10b size")
set_src(i10b, c10b)
print("[ok] Step 10b: download size updated")

# --- 5. Step 12: --ser_model argparse help text --------------------------------
i33 = find_cell("emotion-stream-33")
c33 = src(i33)
c33 = replace_once(
    c33,
    "Hugging Face speech-emotion-recognition model (wav2vec2 classification head, 16 kHz input)",
    "Hugging Face speech-emotion-recognition model (emotion2vec+ via funasr, "
    "or a wav2vec2 classification head; 16 kHz input)",
    "Step 12 --ser_model help",
)
set_src(i33, c33)
print("[ok] Step 12: --ser_model help updated")

# --- 6. Step 17: ERROR_HINTS wording -------------------------------------------
i44 = find_cell("emotion-stream-44")
c44 = src(i44)
c44 = replace_once(
    c44,
    "check the traceback above (HF download, VRAM, transformers version)",
    "check the traceback above (HF download, VRAM, funasr/transformers import)",
    "Step 17 ERROR_HINTS",
)
set_src(i44, c44)
print("[ok] Step 17: ERROR_HINTS updated")

# --- 7. swap the model id everywhere else --------------------------------------
total = 0
for i, c in enumerate(cells):
    s = src(i)
    n = s.count(OLD_MODEL)
    if n:
        set_src(i, s.replace(OLD_MODEL, NEW_MODEL))
        total += n
print(f"[ok] model id swapped in {total} place(s)")

# --- 8. re-embed AUTO_EMOTION_MODULE in the Step 12 cell -----------------------
# Done LAST so the global model-id swap cannot touch the module text: the
# embedded copy must stay byte-identical to IMTalker/auto_emotion.py.
c33 = src(i33)
m = re.search(r"AUTO_EMOTION_MODULE = r'''(.*?)'''", c33, re.DOTALL)
if not m:
    sys.exit("AUTO_EMOTION_MODULE not found in Step 12 cell")
c33 = c33.replace(m.group(1), ae_content)
set_src(i33, c33)
print("[ok] Step 12: AUTO_EMOTION_MODULE re-embedded")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"[ok] notebook written: {NB_PATH}")
