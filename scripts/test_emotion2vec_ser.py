#!/usr/bin/env python3
"""Offline unit test for the emotion2vec SER path in IMTalker/auto_emotion.py.

Runs without torch / funasr / a GPU: both packages are stubbed in sys.modules
so the SpeechEmotionClassifier constructor, the 24->16 kHz resampling
plumbing, the bilingual label mapping and the funasr generate() response
parsing can be exercised locally.
"""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AE_PATH = ROOT / "IMTalker" / "auto_emotion.py"

# ------------------------------------------------------------------ stubs --

class _FakeDevice:
    def __init__(self, s):
        self._s = str(s)
        self.type = self._s.split(":")[0]

    def __str__(self):
        return self._s


class _FakeTensor:
    def __init__(self, arr):
        self.arr = np.asarray(arr)

    def reshape(self, *shape):
        return _FakeTensor(self.arr.reshape(*shape))

    def numpy(self):
        return self.arr

    def __getitem__(self, i):
        return _FakeTensor(self.arr[i])


torch_stub = types.ModuleType("torch")
torch_stub.from_numpy = lambda a: _FakeTensor(a)
torch_stub.device = _FakeDevice

torchaudio_stub = types.ModuleType("torchaudio")
torchaudio_stub.functional = types.SimpleNamespace(
    resample=lambda wav, src, dst: _FakeTensor(wav.arr[:: max(1, src // dst)])
)

sys.modules["torch"] = torch_stub
sys.modules["torchaudio"] = torchaudio_stub

FUNASR_LABELS = [
    "生气/angry", "厌恶/disgusted", "恐惧/fearful", "开心/happy",
    "中立/neutral", "其他/other", "难过/sad", "吃惊/surprised", "<unk>",
]
FUNASR_SCORES = [0.05, 0.02, 0.03, 0.75, 0.05, 0.03, 0.04, 0.02, 0.01]

funasr_state = {"init_kwargs": None, "generate_calls": [], "results": None}


class _FakeAutoModel:
    def __init__(self, **kwargs):
        funasr_state["init_kwargs"] = kwargs
        funasr_state["results"] = [
            {"key": "utt", "labels": FUNASR_LABELS, "scores": FUNASR_SCORES}
        ]

    def generate(self, *args, **kwargs):
        funasr_state["generate_calls"].append((args, kwargs))
        if funasr_state.get("raise_on_array") and isinstance(args[0], np.ndarray):
            raise RuntimeError("stub: array input not supported")
        return funasr_state["results"]


funasr_stub = types.ModuleType("funasr")
funasr_stub.AutoModel = _FakeAutoModel
sys.modules["funasr"] = funasr_stub

soundfile_stub = types.ModuleType("soundfile")
soundfile_stub.write = (
    lambda path, data, sr: funasr_state.setdefault("wav_writes", []).append((path, sr))
)
sys.modules["soundfile"] = soundfile_stub

# -------------------------------------------------------- import the module --

spec = importlib.util.spec_from_file_location("auto_emotion", AE_PATH)
ae = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ae)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"[ok] {name}")
    else:
        failures.append(name)
        print(f"[FAIL] {name} {detail}")


# -------------------------------------------------------------- the tests --

clf = ae.SpeechEmotionClassifier(
    "emotion2vec/emotion2vec_plus_base", device="cuda:0", half=True, disgust_map=2
)

kw = funasr_state["init_kwargs"]
check("funasr.AutoModel load kwargs (hub=hf, disable_update)",
      kw == {"model": "emotion2vec/emotion2vec_plus_base", "hub": "hf",
             "device": "cuda:0", "disable_update": True}, repr(kw))
check("--ser_half ignored for the emotion2vec backend", clf.half is False)
check("no transformers feature extractor on the funasr path",
      clf.feature_extractor is None and clf.id2label == {})

emotion_id, label, conf = clf.predict(np.zeros(24_000, dtype=np.float32))
check("predict returns (emotion_id, raw label, confidence)",
      (emotion_id, label, conf) == (1, "开心/happy", 0.75),
      repr((emotion_id, label, conf)))

args, kwargs = funasr_state["generate_calls"][-1]
check("generate called with 16 kHz float32 array + utterance granularity",
      isinstance(args[0], np.ndarray) and args[0].dtype == np.float32
      and kwargs == {"fs": 16_000, "granularity": "utterance",
                     "extract_embedding": False}, repr((type(args[0]), kwargs)))

label_cases = {
    "生气/angry": 3, "厌恶/disgusted": 2, "恐惧/fearful": 5,
    "开心/happy": 1, "中立/neutral": 0, "其他/other": 0, "难过/sad": 2,
    "吃惊/surprised": 4, "<unk>": 0, "unknown": 0, "  Angry ": 3,
    "disgust": 2, "sad": 2, "hap": 1, "completely-unknown-label": 0,
}
for raw, want in label_cases.items():
    got = clf._label_to_emotion_id(raw)
    check(f"label {raw!r} -> emotion {want}", got == want, f"got {got}")

# empty / malformed results fall back to neutral
funasr_state["results"] = []
check("empty results -> neutral/0.0",
      clf._predict_emotion2vec(np.zeros(1600, np.float32)) == (0, "neutral", 0.0))
funasr_state["results"] = [{"labels": ["开心/happy"], "scores": []}]
check("missing scores -> neutral/0.0",
      clf._predict_emotion2vec(np.zeros(1600, np.float32)) == (0, "neutral", 0.0))

# temp-wav fallback for funasr builds that reject array input
funasr_state["results"] = [
    {"key": "utt", "labels": FUNASR_LABELS, "scores": FUNASR_SCORES}
]
funasr_state["raise_on_array"] = True
emotion_id, label, conf = clf._predict_emotion2vec(np.zeros(1600, np.float32))
check("temp-wav fallback recovers the prediction",
      (emotion_id, label, conf) == (1, "开心/happy", 0.75),
      repr((emotion_id, label, conf)))
check("fallback wrote a 16 kHz wav and removed it",
      bool(funasr_state["wav_writes"])
      and funasr_state["wav_writes"][0][1] == 16_000
      and not Path(funasr_state["wav_writes"][0][0]).exists())

# wav2vec2 repos must still take the transformers path
check("non-emotion2vec name skips the funasr backend",
      not ae.SpeechEmotionClassifier._is_emotion2vec(
          types.SimpleNamespace(model_name="superb/wav2vec2-base-superb-er")))

if failures:
    sys.exit(f"{len(failures)} test(s) failed")
print("[ok] all emotion2vec SER tests passed")
