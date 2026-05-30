"""Groq cloud Speech-to-Text (Whisper large-v3-turbo).

Groq hosts OpenAI Whisper large-v3-turbo and runs it at a very high
real-time factor, so a 30-second interview clip transcribes in a
fraction of a second - typically far faster than the same model on a
laptop CPU, and more accurate than the local 'small' model.

Tradeoffs vs local:
  + Much faster on a typical machine; no local CPU load.
  + Same/better accuracy (full turbo, not int8-quantized).
  - Requires internet and sends the captured audio to Groq.
  - Uses your Groq free-tier quota (shared with the LLM if you also
    use Groq for chat). When you run out, switch STT back to 'local'
    in Settings -> Audio/STT.

API: OpenAI-compatible audio/transcriptions multipart endpoint. We
encode the float32 buffer to an in-memory 16 kHz mono WAV and POST it.
``initial_prompt`` biasing is supported via the 'prompt' form field,
mirroring the local engine's resume/JD keyword biasing.
"""
from __future__ import annotations

import io
import time
import wave

import httpx
import numpy as np

from .base import STTEngine

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "whisper-large-v3-turbo"

# Cap the audio we upload. The Groq dashboard showed every call sending a
# fixed 30s blob, which bloats the upload and exposes the request to the
# free-tier STT queue (latency spiked to 6-13s). One interview question
# almost never exceeds ~15s of actual speech, so we trim silence and cap
# to this. Smaller payload = faster upload + less compute + less queue
# exposure.
MAX_UPLOAD_SECONDS = 15.0


def _isolate_last_utterance(
    audio: np.ndarray,
    samplerate: int,
    max_seconds: float,
    min_seconds: float = 6.0,
    pause_seconds: float = 1.2,
) -> np.ndarray:
    """Return only the most-recent UTTERANCE, silence-trimmed and capped.

    A press almost always lands right after the interviewer finishes
    asking, so the audio looks like:
        [candidate's previous answer] ... <pause> ... [the question] (press)
    We only want the question. This:
      1. Trims leading/trailing silence (frame RMS gate, 50ms frames).
      2. Within the spoken region, finds the LAST internal pause >=
         ``pause_seconds`` (a real turn boundary, not a mid-sentence
         breath) and keeps only what comes AFTER it.
      3. Never returns less than ``min_seconds`` (so a question that
         itself contains a short pause isn't truncated), and never more
         than ``max_seconds``.

    Sending the question alone instead of the whole 15s window roughly
    halves Groq STT time (latency scales with audio length) and removes
    cross-talk from the candidate's own speech, improving accuracy.

    Conservative fallback: if no clear speech is found, return the last
    ``max_seconds`` so we never accidentally send empty/clipped audio.
    """
    if audio.size == 0:
        return audio
    cap = int(max(1.0, max_seconds) * samplerate)
    floor = int(max(1.0, min_seconds) * samplerate)
    frame = max(1, int(0.05 * samplerate))  # 50 ms frames
    n_frames = audio.size // frame
    if n_frames < 2:
        return audio[-cap:]
    trimmed_len = n_frames * frame
    frames = audio[:trimmed_len].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0:
        return audio[-cap:]
    thresh = max(peak * 0.18, 0.004)
    voiced_mask = rms > thresh
    voiced = np.where(voiced_mask)[0]
    if voiced.size == 0:
        return audio[-cap:]

    # Trim to the contiguous speech region first.
    pad = int(0.2 * samplerate)
    region_start = max(0, voiced[0] * frame - pad)
    region_end = min(audio.size, (voiced[-1] + 1) * frame + pad)

    # Find the last internal pause >= pause_seconds within the region.
    pause_frames = max(1, int(pause_seconds / 0.05))  # frames of silence
    last_voiced = voiced[-1]
    # Walk backwards over the voiced frames looking for a gap.
    cut_frame = voiced[0]
    run = 0
    prev = voiced[0]
    # Build the set of silent frames between first and last voiced frame.
    for f in range(last_voiced, voiced[0], -1):
        if not voiced_mask[f]:
            run += 1
            if run >= pause_frames:
                cut_frame = f + run  # start of speech after this pause
                break
        else:
            run = 0
    utt_start = max(region_start, cut_frame * frame)
    utt = audio[utt_start:region_end]

    # Enforce the min floor: if the isolated utterance is shorter than
    # min_seconds, widen the window backwards (the question may contain
    # its own short pause we cut on).
    if utt.size < floor:
        utt = audio[max(region_start, region_end - floor):region_end]
    # Enforce the max cap.
    if utt.size > cap:
        utt = utt[-cap:]
    return utt


def _trim_silence(audio: np.ndarray, samplerate: int, max_seconds: float) -> np.ndarray:
    """Backwards-compatible alias: trim silence + cap (no utterance split).

    Retained for callers/tests that want the simple behavior.
    """
    if audio.size == 0:
        return audio
    cap = int(max(1.0, max_seconds) * samplerate)
    frame = max(1, int(0.05 * samplerate))
    n_frames = audio.size // frame
    if n_frames < 2:
        return audio[-cap:]
    trimmed_len = n_frames * frame
    frames = audio[:trimmed_len].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    peak = float(rms.max()) if rms.size else 0.0
    if peak <= 0:
        return audio[-cap:]
    thresh = max(peak * 0.18, 0.004)
    voiced = np.where(rms > thresh)[0]
    if voiced.size == 0:
        return audio[-cap:]
    pad = int(0.2 * samplerate)
    start = max(0, voiced[0] * frame - pad)
    end = min(audio.size, (voiced[-1] + 1) * frame + pad)
    speech = audio[start:end]
    return speech[-cap:] if speech.size > cap else speech


def _float32_to_wav_bytes(audio: np.ndarray, samplerate: int) -> bytes:
    """Encode a mono float32 [-1, 1] array as 16-bit PCM WAV in memory."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(samplerate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class GroqSTTEngine(STTEngine):
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
    ):
        if not api_key:
            raise ValueError(
                "Groq API key is missing. Get a free one at "
                "https://console.groq.com/keys and paste it in "
                "Settings -> AI Provider, then set STT backend to 'cloud'."
            )
        self.samplerate = 16000
        self.api_key = api_key
        self.model = (model or DEFAULT_MODEL).strip()
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._prompt: str | None = None
        # Persistent keep-alive client: reuses one TLS connection across
        # presses, removing a fresh handshake (~0.3-0.8s) every call.
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=4, keepalive_expiry=300.0),
        )
        print(
            f"[groq-stt] cloud STT ready (model={self.model!r}); "
            f"audio will be uploaded to Groq for transcription.",
            flush=True,
        )

    def set_bias(self, keywords: list[str]) -> None:
        from .biasing import build_initial_prompt

        self._prompt = build_initial_prompt(keywords)
        n = len(keywords)
        print(
            f"[groq-stt] bias vocabulary set: {n} term(s)"
            + (f" e.g. {keywords[:6]}" if n else ""),
            flush=True,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None or audio.size < self.samplerate * 0.5:
            return ""
        audio = audio.astype(np.float32, copy=False)
        # Isolate just the interviewer's LAST utterance (the question),
        # silence-trimmed and capped. Sending the question alone instead
        # of the whole window roughly halves STT time (latency scales
        # with audio length) and removes the candidate's own speech.
        raw_secs = audio.size / self.samplerate
        audio = _isolate_last_utterance(audio, self.samplerate, MAX_UPLOAD_SECONDS)
        sent_secs = audio.size / self.samplerate
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak == 0.0:
            return ""
        if peak < 0.2:
            audio = audio / peak * 0.6

        wav_bytes = _float32_to_wav_bytes(audio, self.samplerate)

        files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
        data = {
            "model": self.model,
            "language": "en",
            "response_format": "json",
            "temperature": "0",
        }
        if self._prompt:
            data["prompt"] = self._prompt

        t0 = time.monotonic()
        try:
            resp = self._client.post(
                "/audio/transcriptions",
                files=files,
                data=data,
            )
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Groq STT connection failed: {type(e).__name__}: {e}"
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Groq STT HTTP {resp.status_code}: {resp.text[:300]}"
            )
        elapsed = time.monotonic() - t0
        # Wall-clock receipt so latency is visible (the dashboard showed
        # STT, not the LLM, was the bottleneck).
        print(
            f"[groq-stt] uploaded {sent_secs:.1f}s audio "
            f"(isolated from {raw_secs:.1f}s, {len(wav_bytes)//1024} KB) -> "
            f"{elapsed:.2f}s round-trip",
            flush=True,
        )
        try:
            return (resp.json().get("text") or "").strip()
        except Exception as e:
            raise RuntimeError(f"Groq STT bad response: {e}")
