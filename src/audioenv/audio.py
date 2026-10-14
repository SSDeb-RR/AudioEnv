"""Audio renderer boundary.

The built-in renderer is deliberately a deterministic WAV fixture generator for
tests. Production callers plug in a native audio model and never expose source
text to the policy.
"""

from __future__ import annotations

import hashlib
import io
import math
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
        digest = hashlib.sha256(f"{seed}:{private_text}:{profile.id}".encode()).digest()
        frequency = 300 + int.from_bytes(digest[:2], "big") % 700
        frames = int(profile.sample_rate_hz * self.duration_seconds)
        payload = bytearray()
        for i in range(frames):
            sample = int(5_000 * math.sin(2 * math.pi * frequency * i / profile.sample_rate_hz))
            payload.extend(struct.pack("<h", sample))
        output = io.BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(profile.sample_rate_hz)
            wav.writeframes(bytes(payload))
        return output.getvalue()

