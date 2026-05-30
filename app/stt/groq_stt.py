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
import wave

import httpx
import numpy as np

from .base import STTEngine

DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "whisper-large-v3-turbo"


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

        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            resp = httpx.post(
                f"{self.base_url}/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Groq STT connection failed: {type(e).__name__}: {e}"
            )

        if resp.status_code != 200:
            raise RuntimeError(
                f"Groq STT HTTP {resp.status_code}: {resp.text[:300]}"
            )
        try:
            return (resp.json().get("text") or "").strip()
        except Exception as e:
            raise RuntimeError(f"Groq STT bad response: {e}")
