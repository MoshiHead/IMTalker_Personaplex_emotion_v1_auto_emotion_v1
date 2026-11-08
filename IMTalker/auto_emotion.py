"""Automatic user-voice emotion detection for the live avatar (EMOTION_LIVE_PATCH_V3).

Taps the user's microphone PCM (already decoded to 24 kHz int16 in the GPU
producer thread), segments it into utterances with an energy-based VAD, runs a
wav2vec2 speech-emotion-recognition (SER) model on each completed utterance in
a background daemon thread, and publishes the detected emotion id for the FM
generator to pick up on its next 0.96 s chunk.

Design rules (mirrors the server's own defensive patterns):
  * feed() runs on the GPU producer thread and must stay in the microsecond
    range -- buffering + VAD only, never inference, never blocking.
  * All SER inference happens on a daemon worker thread with its own CUDA
    stream; a slow or failed inference can never stall the streaming loop.
  * Any failure (model load, inference, feed) degrades to the fixed
    --emotion_id behaviour instead of raising.
"""

from __future__ import annotations

import collections
import queue
import threading
import time
import traceback

import numpy as np

# Emotion ids must match generator/FM.py EMOTION_MAP and train_emotion_lora.ipynb.
EMOTION_NAMES = {0: "neutral", 1: "happy", 2: "sad", 3: "angry", 4: "surprise", 5: "fear"}

TARGET_SR = 24_000  # server mic PCM rate (Mimi sample rate)
SER_SR = 16_000     # wav2vec2 SER models expect 16 kHz

DEFAULT_SER_MODEL = "superb/wav2vec2-base-superb-er"

# Fallback label order for the default SER model (used only when the model
# config carries no id2label). superb/wav2vec2-base-superb-er: 4 classes
# trained on IEMOCAP (English conversational speech), ~95M params.
DEFAULT_SER_LABELS = ["neu", "hap", "ang", "sad"]

# SER label name -> avatar emotion id (generator/FM.py EMOTION_MAP).
# Covers both the IEMOCAP abbreviations (neu/hap/ang/sad) and the full-word
# labels used by e.g. ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
# (angry, calm, disgust, fearful, happy, neutral, sad, surprised).
# "disgust" is handled separately via the --ser_disgust_map arg because the
# emotion LoRA was not trained on disgust.
SER_LABEL_TO_EMOTION_ID = {
    "neutral": 0, "neutrality": 0, "calm": 0, "neu": 0,
    "happy": 1, "happiness": 1, "joy": 1, "excited": 1, "hap": 1, "exc": 1,
    "sad": 2, "sadness": 2,
    "angry": 3, "anger": 3, "frustrated": 3, "ang": 3,
    "surprise": 4, "surprised": 4,
    "fear": 5, "fearful": 5, "fearfulness": 5,
}



# ---------------------------------------------------------------------------
# Energy-based VAD / utterance segmenter
# ---------------------------------------------------------------------------

class EnergyVAD:
    """Splits a continuous mono stream into utterances (zero dependencies).

    State machine with hysteresis:
      * speech starts when a frame's RMS >= rms_threshold
      * an utterance closes after `silence_sec` of sub-threshold audio
      * utterances shorter than `min_speech_sec` are dropped (coughs, "yes")
      * utterances longer than `max_utterance_sec` are split so SER latency
        stays bounded on long monologues
      * `pre_roll_sec` of audio before speech onset is kept so the classifier
        sees the true utterance start
    """

    def __init__(self, rms_threshold: float = 0.02, silence_sec: float = 0.7,
                 min_speech_sec: float = 0.5, max_utterance_sec: float = 12.0,
                 pre_roll_sec: float = 0.25, sample_rate: int = TARGET_SR,
                 frame_sec: float = 0.02) -> None:
        self.sample_rate = int(sample_rate)
        self.frame_sec = float(frame_sec)
        self.frame_len = max(1, int(round(self.sample_rate * self.frame_sec)))
        self.rms_threshold = float(rms_threshold)
        self.silence_frames = max(1, int(round(float(silence_sec) / self.frame_sec)))
        self.min_speech_frames = max(1, int(round(float(min_speech_sec) / self.frame_sec)))
        self.max_utterance_frames = max(
            self.min_speech_frames, int(round(float(max_utterance_sec) / self.frame_sec))
        )
        self.pre_roll_frames = max(0, int(round(float(pre_roll_sec) / self.frame_sec)))
        self.reset()

    def reset(self) -> None:
        self._pending = np.zeros(0, dtype=np.float32)  # partial-frame carry-over
        self._pre_roll: collections.deque = collections.deque(maxlen=self.pre_roll_frames)
        self._speech: list = []
        self._in_speech = False
        self._silence_run = 0    # consecutive sub-threshold frames while in speech
        self._speech_frames = 0  # speech (non-silence) frames in current utterance

    def feed(self, pcm: np.ndarray):
        """Feed float32 mono samples @ sample_rate; return an utterance or None."""
        pcm = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if pcm.size:
            self._pending = np.concatenate([self._pending, pcm])
        utterance = None
        n_frames = self._pending.shape[0] // self.frame_len
        for i in range(n_frames):
            frame = self._pending[i * self.frame_len:(i + 1) * self.frame_len]
            out = self._process_frame(frame)
            if out is not None:
                utterance = out  # keep the last completed utterance of this call
        if n_frames:
            self._pending = self._pending[n_frames * self.frame_len:]
        return utterance

    def _process_frame(self, frame: np.ndarray):
        rms = float(np.sqrt(np.mean(np.square(frame, dtype=np.float32))))
        if rms >= self.rms_threshold:
            if not self._in_speech:
                self._in_speech = True
                self._speech = list(self._pre_roll)
                self._speech_frames = 0
                self._silence_run = 0
            self._speech.append(frame)
            self._speech_frames += 1
            self._silence_run = 0
            if self._speech_frames >= self.max_utterance_frames:
                return self._close_utterance(trim=False)
            return None
        # sub-threshold frame
        if self._in_speech:
            self._speech.append(frame)
            self._silence_run += 1
            if self._silence_run >= self.silence_frames:
                return self._close_utterance(trim=True)
        else:
            self._pre_roll.append(frame)
        return None

    def _close_utterance(self, trim: bool):
        frames = self._speech
        if trim and self._silence_run:
            # Keep a short natural tail (~0.15 s) instead of the full silence run.
            keep_tail = min(self._silence_run, int(round(0.15 / self.frame_sec)))
            frames = frames[:len(frames) - self._silence_run + keep_tail]
        speech_ok = self._speech_frames >= self.min_speech_frames
        self._in_speech = False
        self._speech = []
        self._speech_frames = 0
        self._silence_run = 0
        self._pre_roll.clear()
        if not speech_ok or not frames:
            return None
        return np.concatenate(frames).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Speech emotion recognition (SER) classifier
# ---------------------------------------------------------------------------

class SpeechEmotionClassifier:
    """wav2vec2-based speech emotion classifier (16 kHz mono input).

    Supports two checkpoint layouts:
      * standard Wav2Vec2ForSequenceClassification repos (architectures field
        names a *ForSequenceClassification / *ForAudioClassification class,
        e.g. superb/wav2vec2-base-superb-er) -> loaded via
        AutoModelForAudioClassification
      * audeering-style repos (wav2vec2 backbone + mean-pool + linear `head`)
        -> loaded via the small documented EmotionModel wrapper
    """

    def __init__(self, model_name: str, device: str = "cuda", half: bool = False,
                 disgust_map: int = 0) -> None:
        import torch
        from transformers import AutoConfig, AutoFeatureExtractor

        self.torch = torch
        self.device = torch.device(device)
        self.half = bool(half) and self.device.type == "cuda"
        self.disgust_map = int(disgust_map)
        self.model_name = str(model_name)

        config = AutoConfig.from_pretrained(self.model_name)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        architectures = [str(a) for a in (getattr(config, "architectures", None) or [])]

        if any(("ForSequenceClassification" in a) or ("ForAudioClassification" in a)
               for a in architectures):
            self._load_standard()
        else:
            self._load_audeering_style(config)

        if self.half:
            self.model = self.model.half()
        self.model.to(self.device).eval()

        id2label = getattr(self.model.config, "id2label", None) or {}
        self.id2label = {int(k): str(v).lower() for k, v in id2label.items()}
        if not self.id2label:
            self.id2label = {i: n for i, n in enumerate(DEFAULT_SER_LABELS)}
        self._warned_labels: set = set()

    def _load_standard(self) -> None:
        from transformers import AutoModelForAudioClassification

        self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)

        def forward_logits(input_values):
            return self.model(input_values=input_values).logits

        self._forward_logits = forward_logits

    def _load_audeering_style(self, config) -> None:
        import torch.nn as nn
        from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel

        class EmotionModel(Wav2Vec2PreTrainedModel):
            """Wrapper from the audeering model card: wav2vec2 + mean-pool + linear head."""

            def __init__(self, cfg):
                super().__init__(cfg)
                self.config = cfg
                self.wav2vec2 = Wav2Vec2Model(cfg)
                num_labels = len(getattr(cfg, "id2label", None) or {}) or int(
                    getattr(cfg, "num_labels", len(DEFAULT_SER_LABELS))
                )
                self.head = nn.Linear(cfg.hidden_size, num_labels)
                self.init_weights()

            def forward(self, input_values):
                hidden = self.wav2vec2(input_values).last_hidden_state
                return self.head(hidden.mean(dim=1))

        self.model = EmotionModel.from_pretrained(self.model_name)

        def forward_logits(input_values):
            return self.model(input_values)

        self._forward_logits = forward_logits

    def _label_to_emotion_id(self, label: str) -> int:
        key = str(label).strip().lower()
        if key in ("disgust", "disgusted"):
            return self.disgust_map
        if key in SER_LABEL_TO_EMOTION_ID:
            return SER_LABEL_TO_EMOTION_ID[key]
        if key not in self._warned_labels:
            self._warned_labels.add(key)
            print(f"[AUTO-EMOTION] unknown SER label {label!r}; mapping to neutral", flush=True)
        return 0

    def predict(self, pcm_24k: np.ndarray):
        """pcm_24k: float32 mono @ 24 kHz. Returns (emotion_id, label, confidence)."""
        torch = self.torch
        import torchaudio

        wav = torch.from_numpy(np.asarray(pcm_24k, dtype=np.float32).reshape(1, -1))
        if TARGET_SR != SER_SR:
            wav = torchaudio.functional.resample(wav, TARGET_SR, SER_SR)
        inputs = self.feature_extractor(
            wav[0].numpy(), sampling_rate=SER_SR, return_tensors="pt"
        )
        input_values = inputs["input_values"].to(self.device)
        if self.half:
            input_values = input_values.half()
        with torch.inference_mode():
            logits = self._forward_logits(input_values)
            probs = torch.softmax(logits.float(), dim=-1)[0]
        conf, idx = torch.max(probs, dim=0)
        label = self.id2label.get(int(idx), str(int(idx)))
        return self._label_to_emotion_id(label), label, float(conf)



# ---------------------------------------------------------------------------
# Manager: VAD on the GPU thread, SER on a daemon worker thread
# ---------------------------------------------------------------------------

class AutoEmotionManager:
    """Owns the VAD, the utterance queue, the SER worker and the published id.

    feed() is called from the GPU producer thread for every mic packet and is
    deliberately cheap (resample + RMS + buffering). Completed utterances are
    classified on a daemon worker thread; the result is published as a plain
    int that the FM chunk builder reads once per 0.96 s chunk.
    """

    def __init__(self, args) -> None:
        self.args = args
        self.initial_emotion_id = int(getattr(args, "emotion_id", 0))
        self._emotion_id = self.initial_emotion_id
        self.confidence_threshold = float(getattr(args, "ser_confidence_threshold", 0.5))
        self.vad = EnergyVAD(
            rms_threshold=float(getattr(args, "vad_rms_threshold", 0.02)),
            silence_sec=float(getattr(args, "vad_silence_sec", 0.7)),
            min_speech_sec=float(getattr(args, "vad_min_speech_sec", 0.5)),
            max_utterance_sec=float(getattr(args, "vad_max_utterance_sec", 12.0)),
        )
        self._queue: queue.Queue = queue.Queue(maxsize=4)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._classifier: SpeechEmotionClassifier | None = None
        self.utterances_classified = 0
        self.utterances_dropped = 0
        self.feed_errors = 0

    def start(self) -> None:
        """Load the SER model (may raise -> caller disables auto mode) and
        start the daemon worker thread."""
        model_name = str(getattr(self.args, "ser_model", "") or DEFAULT_SER_MODEL)
        device = str(getattr(self.args, "ser_device", "") or getattr(self.args, "device", "cuda"))
        half = bool(getattr(self.args, "ser_half", False))
        disgust_map = int(getattr(self.args, "ser_disgust_map", 0))
        t0 = time.perf_counter()
        self._classifier = SpeechEmotionClassifier(
            model_name, device=device, half=half, disgust_map=disgust_map
        )
        self._worker = threading.Thread(
            target=self._worker_loop, name="auto-emotion-ser", daemon=True
        )
        self._worker.start()
        print(
            f"[AUTO-EMOTION] enabled: model={model_name} device={device} half={half} "
            f"confidence_threshold={self.confidence_threshold:.2f} "
            f"initial_emotion={EMOTION_NAMES.get(self.initial_emotion_id, self.initial_emotion_id)} "
            f"(SER model loaded in {time.perf_counter() - t0:.1f}s)",
            flush=True,
        )

    def feed(self, pcm_i16: np.ndarray, input_sr: int) -> None:
        """GPU-producer-thread entry point. Never raises, never blocks."""
        try:
            pcm = np.asarray(pcm_i16, dtype=np.float32) / 32768.0
            if int(input_sr) != TARGET_SR:
                import torch
                import torchaudio

                wav = torch.from_numpy(pcm).view(1, -1)
                pcm = torchaudio.functional.resample(wav, int(input_sr), TARGET_SR)[0].numpy()
            with self._lock:
                utterance = self.vad.feed(pcm)
            if utterance is None:
                return
            try:
                self._queue.put_nowait(utterance)
            except queue.Full:
                # Drop the OLDEST queued utterance (the newest user speech is
                # what a live conversation cares about), same policy as the
                # server's own mic backlog handling.
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(utterance)
                except Exception:
                    self.utterances_dropped += 1
        except Exception as exc:
            self.feed_errors += 1
            if self.feed_errors <= 5:
                print(f"[AUTO-EMOTION] feed error #{self.feed_errors}: {exc!r}", flush=True)


    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            try:
                utterance = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            dur = utterance.shape[0] / float(TARGET_SR)
            try:
                emotion_id, label, conf = self._classifier.predict(utterance)
            except Exception as exc:
                print(
                    f"[AUTO-EMOTION] inference failed ({dur:.2f}s utterance): {exc!r}",
                    flush=True,
                )
                continue
            self.utterances_classified += 1
            prev = self._emotion_id
            if conf >= self.confidence_threshold:
                self._emotion_id = int(emotion_id)
                action = "switching" if prev != emotion_id else "confirmed"
            else:
                action = f"below threshold {self.confidence_threshold:.2f}; keeping"
            print(
                f"[AUTO-EMOTION] utterance {dur:.2f}s -> {label} (p={conf:.2f}) "
                f"[{EMOTION_NAMES.get(prev, prev)} -> "
                f"{EMOTION_NAMES.get(self._emotion_id, self._emotion_id)}] {action}",
                flush=True,
            )

    def current_emotion_id(self, default: int = 0) -> int:
        return int(self._emotion_id)

    def reset_session(self) -> None:
        """Fresh VAD state per browser session; the last detected emotion
        carries over (same user), falling back to --emotion_id until the
        first detection of the session."""
        with self._lock:
            self.vad.reset()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop.set()


# ---------------------------------------------------------------------------
# Module-level singleton (created once at server startup)
# ---------------------------------------------------------------------------

_MANAGER: AutoEmotionManager | None = None


def init_auto_emotion_manager(args) -> AutoEmotionManager | None:
    """Create and start the singleton. On ANY failure, auto mode is disabled
    and the server keeps the fixed --emotion_id behaviour."""
    global _MANAGER
    if _MANAGER is not None:
        return _MANAGER
    try:
        mgr = AutoEmotionManager(args)
        mgr.start()
    except Exception as exc:
        print(
            f"[AUTO-EMOTION] DISABLED - failed to initialise the SER model; "
            f"falling back to fixed --emotion_id={int(getattr(args, 'emotion_id', 0))}: "
            f"{exc!r}\n{traceback.format_exc()}",
            flush=True,
        )
        _MANAGER = None
        return None
    _MANAGER = mgr
    return _MANAGER


def get_auto_emotion_manager() -> AutoEmotionManager | None:
    return _MANAGER

