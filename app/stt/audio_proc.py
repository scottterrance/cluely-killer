"""Audio pre-processing for STT: silence trimming + last-utterance isolation.

These were originally embedded in the (now-removed) Groq cloud STT engine.
They matter MORE for the local path: on a CPU, faster-whisper latency is
roughly linear in audio length, so feeding it only the interviewer's last
question (~5-8 s) instead of the whole capture window (~15-25 s) is the
single biggest local-STT speedup available - it cuts transcription time
by 2-4x AND improves accuracy by removing the candidate's own speech.

All functions operate on mono float32 [-1, 1] numpy arrays.
"""
from __future__ import annotations

import numpy as np

_FRAME_SECONDS = 0.05  # 50 ms analysis frames


def rms(audio: np.ndarray) -> float:
    """Root-mean-square level of a buffer (0.0 for empty/None)."""
    if audio is None or audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))


def _frame_rms(audio: np.ndarray, samplerate: int) -> tuple[np.ndarray, int]:
    """Return (per-frame RMS array, frame_size_in_samples)."""
    frame = max(1, int(_FRAME_SECONDS * samplerate))
    n_frames = audio.size // frame
    if n_frames < 1:
        return np.zeros(0, dtype=np.float64), frame
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    return np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)), frame


def trim_silence(audio: np.ndarray, samplerate: int, max_seconds: float) -> np.ndarray:
    """Trim leading/trailing silence and cap to the most-recent
    ``max_seconds`` of the contiguous spoken region.

    Conservative: if no clear speech is found, returns the last
    ``max_seconds`` so we never accidentally send empty/clipped audio.
    """
    if audio is None or audio.size == 0:
        return audio
    cap = int(max(1.0, max_seconds) * samplerate)
    rms_frames, frame = _frame_rms(audio, samplerate)
    if rms_frames.size < 2:
        return audio[-cap:]
    peak = float(rms_frames.max()) if rms_frames.size else 0.0
    if peak <= 0:
        return audio[-cap:]
    thresh = max(peak * 0.18, 0.004)
    voiced = np.where(rms_frames > thresh)[0]
    if voiced.size == 0:
        return audio[-cap:]
    pad = int(0.2 * samplerate)
    start = max(0, voiced[0] * frame - pad)
    end = min(audio.size, (voiced[-1] + 1) * frame + pad)
    speech = audio[start:end]
    return speech[-cap:] if speech.size > cap else speech


def isolate_last_utterance(
    audio: np.ndarray,
    samplerate: int,
    max_seconds: float,
    min_seconds: float = 5.0,
    pause_seconds: float = 1.0,
) -> np.ndarray:
    """Return only the most-recent UTTERANCE, silence-trimmed and capped.

    A press almost always lands right after the interviewer finishes
    asking, so the captured audio looks like:
        [candidate's previous answer] ... <pause> ... [the question] (press)
    We only want the question. This:
      1. Trims leading/trailing silence (frame RMS gate).
      2. Within the spoken region, finds the LAST internal pause >=
         ``pause_seconds`` (a real turn boundary, not a mid-sentence
         breath) and keeps only what comes AFTER it.
      3. Never returns less than ``min_seconds`` (so a question that
         itself contains a short pause isn't truncated), and never more
         than ``max_seconds``.

    Sending the question alone instead of the whole window roughly halves
    local Whisper time (latency scales with audio length) and removes
    cross-talk from the candidate's own speech, improving accuracy.

    Conservative fallback: if no clear speech is found, return the last
    ``max_seconds``.
    """
    if audio is None or audio.size == 0:
        return audio
    cap = int(max(1.0, max_seconds) * samplerate)
    floor = int(max(1.0, min_seconds) * samplerate)
    rms_frames, frame = _frame_rms(audio, samplerate)
    if rms_frames.size < 2:
        return audio[-cap:]
    peak = float(rms_frames.max()) if rms_frames.size else 0.0
    if peak <= 0:
        return audio[-cap:]
    thresh = max(peak * 0.18, 0.004)
    voiced_mask = rms_frames > thresh
    voiced = np.where(voiced_mask)[0]
    if voiced.size == 0:
        return audio[-cap:]

    pad = int(0.2 * samplerate)
    region_start = max(0, voiced[0] * frame - pad)
    region_end = min(audio.size, (voiced[-1] + 1) * frame + pad)

    # Walk backwards from the last voiced frame; the first silent run of
    # >= pause_frames marks the boundary between the previous turn and
    # the current question.
    pause_frames = max(1, int(pause_seconds / _FRAME_SECONDS))
    cut_frame = voiced[0]
    run = 0
    for fidx in range(voiced[-1], voiced[0], -1):
        if not voiced_mask[fidx]:
            run += 1
            if run >= pause_frames:
                cut_frame = fidx + run  # start of speech after the pause
                break
        else:
            run = 0
    utt_start = max(region_start, cut_frame * frame)
    utt = audio[utt_start:region_end]

    # Min floor: if the isolated utterance is too short, widen backwards.
    if utt.size < floor:
        utt = audio[max(region_start, region_end - floor):region_end]
    # Max cap.
    if utt.size > cap:
        utt = utt[-cap:]
    return utt
