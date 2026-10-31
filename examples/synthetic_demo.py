#!/usr/bin/env python3
"""Synthetic end-to-end check: no audio files needed.

Generates a short signal with a plausible syllable rate, aligns a fixed lyrics
line to it, and asserts the result is a valid, monotonic, gap-free segmentation.
Used by CI to catch packaging and API regressions.

    python examples/synthetic_demo.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import pyshiro
except ModuleNotFoundError:  # running from a checkout without installing
    import sys
    from pathlib import Path as _Path
    sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
    import pyshiro

SR = 16_000
DURATION = 4.0
LYRICS = "あかいぐつ\nあるいてた\n"


def make_audio() -> np.ndarray:
    t = np.arange(int(SR * DURATION)) / SR

    # pitch contour, wobbly like a sung phrase
    f0 = 220.0 * 2 ** (0.35 * np.sin(2 * np.pi * 0.8 * t))
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 10))

    # syllable-rate amplitude envelope + a little noise
    env = 0.55 + 0.45 * np.sin(2 * np.pi * 2.2 * t) ** 2
    rng = np.random.default_rng(0)
    sig = sig * env + 0.02 * rng.standard_normal(len(t))

    return (sig / np.max(np.abs(sig)) * 0.9).astype(np.float32)


def main() -> None:
    audio = make_audio()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        wav_path = tmp_dir / "synthetic.wav"
        txt_path = tmp_dir / "synthetic.txt"
        lab_path = tmp_dir / "synthetic.lab"

        sf.write(wav_path, audio, SR)
        txt_path.write_text(LYRICS, encoding="utf-8")

        model, phonemap = pyshiro.load_bundled_model()
        streams = pyshiro.extract_mfcc(audio, SR)
        phonemes = pyshiro.convert_lyric_file(txt_path)

        frames = streams[0].shape[0]
        state_seq = pyshiro.build_state_sequence(phonemes, phonemap, frames)
        segments, loglik = pyshiro.forced_align_2pass(
            model, streams, state_seq, nodur_phonemes={"pau", "br"})

        intervals = pyshiro.segments_to_phoneme_intervals(phonemes, segments)
        pyshiro.write_lab(intervals, lab_path)

        lab_size = lab_path.stat().st_size
        lab_head = lab_path.read_text(encoding="utf-8").splitlines()[:3]

    # --- checks ---------------------------------------------------------
    assert len(segments) == len(state_seq), "one segment per state"
    assert len(intervals) == len(phonemes), "one interval per phoneme"
    assert intervals[0][0] == 0, "first interval must start at frame 0"
    assert intervals[-1][1] == frames, f"last interval must end at {frames}"

    for (s0, e0, _), (s1, _, _) in zip(intervals, intervals[1:]):
        assert e0 == s1, f"gap or overlap at frame {e0} vs {s1}"
        assert e0 >= s0, "non-monotonic interval"

    assert lab_size > 0, "lab file is empty"
    assert np.isfinite(loglik), "log-likelihood is not finite"

    print(f"model      : {model.nstate} GMMs, {len(phonemap)} phonemes")
    print(f"audio      : {DURATION:.1f} s -> {frames} frames")
    print(f"phonemes   : {' '.join(phonemes)}")
    print(f"loglik     : {loglik / frames:.3f} nats/frame")
    print(f"lab        : {len(intervals)} intervals, {lab_size} bytes")
    for line in lab_head:
        print(f"             {line}")
    print("OK")


if __name__ == "__main__":
    main()
