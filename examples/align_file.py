#!/usr/bin/env python3
"""Align a WAV file with a lyrics file and write labels.

    python examples/align_file.py song.wav song.txt [--format lab|textgrid|audacity]

Same pipeline as the `pyshiro-align` command, written out so it can be edited:
change the model, post-process the boundaries, or write your own output format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import pyshiro
except ModuleNotFoundError:  # running from a checkout without installing
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import pyshiro


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("wav", type=Path)
    ap.add_argument("lyrics", type=Path)
    ap.add_argument("--model", type=Path, default=None,
                    help="default: bundled jp-v2")
    ap.add_argument("--phonemap", type=Path, default=None,
                    help="default: bundled jp-v2")
    ap.add_argument("--format", choices=["lab", "textgrid", "audacity"], default="lab")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--refine-pau", action="store_true",
                    help="apply RMS onset correction to pau/consonant boundaries")
    args = ap.parse_args()

    if args.model and args.phonemap:
        model = pyshiro.load_hsmm(args.model)
        phonemap = pyshiro.load_phonemap(args.phonemap)
    else:
        model, phonemap = pyshiro.load_bundled_model()

    streams = pyshiro.extract_mfcc_from_file(args.wav)
    phonemes = pyshiro.convert_lyric_file(args.lyrics)
    T = streams[0].shape[0]

    state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
    segments, loglik = pyshiro.forced_align_2pass(
        model, streams, state_seq, nodur_phonemes={"pau", "br"})
    print(f"{T} frames, {len(phonemes)} phonemes, "
          f"{loglik / T:.3f} nats/frame")

    intervals = pyshiro.segments_to_phoneme_intervals(phonemes, segments)

    if args.refine_pau:
        import soundfile as sf
        audio, sr = sf.read(args.wav)
        intervals = pyshiro.refine_pau_boundaries(intervals, audio, sr)

    out = args.out or args.wav.with_suffix(
        {"lab": ".lab", "textgrid": ".TextGrid", "audacity": ".txt"}[args.format])

    if args.format == "lab":
        pyshiro.write_lab(intervals, out)
    elif args.format == "textgrid":
        pyshiro.write_textgrid(intervals, out)
    else:
        pyshiro.write_audacity(intervals, out)

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
