"""Audio renderer boundary.

The built-in renderer is deliberately a deterministic WAV fixture generator for
tests. Production callers plug in a native audio model and never expose source
text to the policy.
"""

from __future__ import annotations

import hashlib
import io
import math
import random
import struct
import wave
from typing import Protocol

from .spec import AudioProfile


class AudioRenderer(Protocol):
    def render(self, private_text: str, profile: AudioProfile, seed: int) -> bytes: ...


class DeterministicWavFixtureRenderer:
    """Create deterministic valid WAV fixtures; not a speech synthesizer."""

    duration_seconds = 0.08

    def render(self, private_text: str, profile: AudioProfile, seed: int) -> bytes:
        validate_audio_profile(profile)
        digest = hashlib.sha256(f"{seed}:{private_text}:{profile.id}".encode()).digest()
        frequency = 300 + int.from_bytes(digest[:2], "big") % 700
        frames = int(profile.sample_rate_hz * self.duration_seconds / profile.speaking_rate)
        rng = random.Random(int.from_bytes(digest[2:10], "big"))
        signal_amplitude = 5_000.0
        noise_amplitude = 0.0
        if profile.noise_snr_db is not None:
            noise_amplitude = signal_amplitude / (10 ** (profile.noise_snr_db / 20.0))
        payload = bytearray()
        for i in range(frames):
            signal = signal_amplitude * math.sin(2 * math.pi * frequency * i / profile.sample_rate_hz)
            sample = int(max(-32_768, min(32_767, signal + rng.gauss(0, noise_amplitude))))
            payload.extend(struct.pack("<h", sample))
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(profile.sample_rate_hz)
            wav.writeframes(bytes(payload))
        return output.getvalue()


def validate_audio_profile(profile: AudioProfile) -> None:
    if profile.sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")
    if profile.speaking_rate <= 0:
        raise ValueError("speaking_rate must be positive")
    if profile.noise_snr_db is not None and not math.isfinite(profile.noise_snr_db):
        raise ValueError("noise_snr_db must be finite")
    if profile.codec != "pcm_s16le":
        raise ValueError(f"unsupported codec without an external codec provider: {profile.codec}")
    if profile.accent is not None:
        raise ValueError("accent transformation requires an external speech renderer")
    if profile.prosody != "neutral":
        raise ValueError("prosody transformation requires an external speech renderer")

