"""Audio pre-processing for STT: silence trimming + last-utterance isolation
+ signal cleanup (high-pass, AGC, light denoise).

These were originally embedded in the (now-removed) Groq cloud STT engine.
They matter MORE for the local path: on a CPU, faster-whisper latency is
roughly linear in audio length, so feeding it only the interviewer's last
question (~5-8 s) instead of the whole capture window (~15-25 s) is the
single biggest local-STT speedup available - it cuts transcription time
by 2-4x AND improves accuracy by removing the candidate's own speech.

Roadmap #6 adds ``clean_audio()`` (high-pass filter + automatic gain
control + spectral-floor denoise). It's a pure-numpy, vectorized chain -
no scipy/torch - so it adds negligible latency and zero new
dependencies, and only improves WER on quiet/noisy input. Whisper still
runs its own VAD; this just hands it a cleaner signal.

All functions operate on mono float32 [-1, 1] numpy arrays.
"""
from __future__ import annotations

import numpy as np

_FRAME_SECONDS = 0.05  # 50 ms analysis frames


def high_pass(audio: np.ndarray, samplerate: int, cutoff_hz: float = 80.0) -> np.ndarray:
    """Remove low-frequency rumble (AC hum, desk thumps, mic handling).

    Implemented as (signal - low-pass(signal)), where the low-pass is a
    centered moving average. This is numerically stable for any length
    (unlike the exponential one-pole 'a^k' trick, which underflows/over-
    flows on multi-second buffers), O(n) via a cumulative-sum sliding
    window, vectorized, and needs no scipy.

    The moving-average window length is chosen so its -3 dB point is
    near ``cutoff_hz`` (window ~ samplerate / cutoff).
    """
    if audio is None or audio.size < 4:
        return audio
    x = audio.astype(np.float32, copy=False)
    win = max(3, int(samplerate / max(1.0, cutoff_hz)))
    if win >= x.size:
        # Window longer than the clip: just remove the mean (DC).
        return (x - float(np.mean(x))).astype(np.float32)
    # Sliding-window mean via prefix sums (odd window, centered).
    if win % 2 == 0:
        win += 1
    half = win // 2
    # Reflect-pad so the moving average is defined at the edges.
    padded = np.pad(x.astype(np.float64), half, mode="reflect")
    csum = np.cumsum(padded)
    # mean[i] = (csum[i+win] - csum[i]) / win  for i in 0..len(x)-1
    csum = np.concatenate(([0.0], csum))
    moving = (csum[win:] - csum[:-win]) / win  # length == x.size
    out = x - moving[: x.size].astype(np.float32)
    return out.astype(np.float32)


def agc(audio: np.ndarray, target_rms: float = 0.08, max_gain: float = 8.0) -> np.ndarray:
    """Automatic gain control: normalize toward a target RMS.

    A quiet interviewer and a loud one both arrive at a consistent level,
    which helps Whisper. Gain is capped so we don't blow up pure noise,
    and the result is soft-clipped to stay within [-1, 1].
    """
    if audio is None or audio.size == 0:
        return audio
    cur = rms(audio)
    if cur < 1e-6:
        return audio  # essentially silence; don't amplify hiss
    gain = min(max_gain, target_rms / cur)
    if gain <= 1.01:
        return audio  # already loud enough
    out = audio.astype(np.float32, copy=False) * gain
    # Soft clip (tanh) only if we'd exceed unity, to avoid harsh edges.
    peak = float(np.max(np.abs(out))) if out.size else 0.0
    if peak > 1.0:
        out = np.tanh(out).astype(np.float32)
    return out


def denoise(audio: np.ndarray, samplerate: int, strength: float = 0.6) -> np.ndarray:
    """Light spectral-floor denoise (Wiener-style, time-domain framing).

    Estimates a noise floor from the quietest frames and attenuates
    frames near that floor, leaving speech frames intact. This is a cheap
    approximation of spectral subtraction that needs no FFT/scipy and is
    safe (never removes speech, just softens steady background hiss).
    ``strength`` in [0,1] scales how aggressively low-energy frames are
    attenuated.
    """
    if audio is None or audio.size == 0:
        return audio
    strength = float(min(1.0, max(0.0, strength)))
    if strength <= 0.0:
        return audio
    rms_frames, frame = _frame_rms(audio, samplerate)
    if rms_frames.size < 4:
        return audio
    # Noise floor = 20th percentile frame energy; speech ref = 90th.
    noise = float(np.percentile(rms_frames, 20))
    speech = float(np.percentile(rms_frames, 90))
    if speech <= noise or speech < 1e-6:
        return audio
    # Per-frame gain: ~1 for speech-level frames, ~ (1-strength) for
    # noise-level frames, smoothly interpolated (Wiener-like ratio).
    ratio = (rms_frames - noise) / (speech - noise)
    ratio = np.clip(ratio, 0.0, 1.0)
    gains = (1.0 - strength) + strength * ratio  # in [1-strength, 1]
    n_frames = rms_frames.size
    out = audio.astype(np.float32, copy=True)
    # Apply per-frame gain to the framed region; tail (partial frame)
    # left untouched.
    framed = out[: n_frames * frame].reshape(n_frames, frame)
    framed *= gains[:, None].astype(np.float32)
    return out


def clean_audio(
    audio: np.ndarray,
    samplerate: int,
    *,
    hpf_cutoff_hz: float = 80.0,
    agc_target_rms: float = 0.08,
    denoise_strength: float = 0.5,
) -> np.ndarray:
    """Full cleanup chain for STT input: high-pass -> denoise -> AGC.

    Order matters: HPF first removes rumble so the noise-floor estimate
    in denoise() isn't skewed by low-frequency energy; AGC last so the
    final level is normalized after noise has been attenuated. Pure
    numpy, vectorized, no new deps. Safe no-op on empty/tiny input.
    """
    if audio is None or audio.size < samplerate // 10:  # <100 ms
        return audio
    x = high_pass(audio, samplerate, hpf_cutoff_hz)
    x = denoise(x, samplerate, denoise_strength)
    x = agc(x, target_rms=agc_target_rms)
    return x.astype(np.float32, copy=False)


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
