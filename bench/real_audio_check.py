#!/usr/bin/env python3
"""Align one real recording and print sanity statistics.

This is the script used for the real-singing check in docs/performance.md, but it
takes a plain phone list instead of a dataset, so it works with any annotated
corpus (or with a phone list you produced yourself).

    python bench/real_audio_check.py --wav song.wav \
        --phones "pau,a,k,a,i,pau" \
        --ref-durs "0.30,0.05,0.12,0.40,0.08,0.25"

Phones must be spelled in the pyshiro inventory (see the phonemap in
pyshiro/data/models/). --ref-durs is optional; when given, the phone boundaries
produced by the aligner are compared against the cumulative reference durations.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OBS = {"p", "b", "t", "d", "k", "g", "s", "sh", "z", "j", "ts", "ch", "h", "cl",
       "ky", "gy", "by", "py"}
FPS = 200.0
MS = 1000.0 / FPS


def parse_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", type=Path, required=True)
    ap.add_argument("--phones", required=True, help="comma-separated phone list")
    ap.add_argument("--ref-durs", default="", help="comma-separated reference durations (s)")
    ap.add_argument("--model", type=Path, default=None)
    ap.add_argument("--phonemap", type=Path, default=None)
    args = ap.parse_args()

    if not args.wav.is_file():
        sys.exit(f"no such file: {args.wav}")

    import pyshiro
    from pyshiro.align import FPS as _FPS, build_state_sequence, forced_align_2pass, load_phonemap
    from pyshiro.features import extract_mfcc_from_file
    from pyshiro.model import load_hsmm

    if args.model and args.phonemap:
        hsmm, pmap = args.model, args.phonemap
    else:
        hsmm, pmap = pyshiro.bundled_model_path("jp-v2")

    model = load_hsmm(hsmm)
    phonemap = load_phonemap(pmap)

    t0 = time.perf_counter()
    streams = extract_mfcc_from_file(args.wav)
    t_feat = time.perf_counter() - t0
    T = streams[0].shape[0]

    phones = parse_list(args.phones)
    missing = [p for p in phones if p not in phonemap]
    if missing:
        sys.exit(f"phones not in the phonemap: {sorted(set(missing))}")

    state_seq = build_state_sequence(phones, phonemap, T)

    t0 = time.perf_counter()
    segs, ll = forced_align_2pass(model, streams, state_seq,
                                  nodur_phonemes={"pau", "br"})
    t_align = time.perf_counter() - t0

    seg = np.array(segs, dtype=np.int64)
    n_states = len(state_seq)
    # first state index of every phoneme (a phoneme can span a variable number of states)
    first = [i for i in range(n_states)
             if i == 0 or state_seq[i].phoneme != state_seq[i - 1].phoneme]
    starts = seg[first, 0]
    ends = np.append(starts[1:], seg[-1, 1])
    durs = (ends - starts) / _FPS

    print(f"audio        : {args.wav}  ({T / _FPS:.2f} s, {T} frames)")
    print(f"phones       : {len(phones)}  ({n_states} states)")
    print(f"features     : {t_feat:.2f} s")
    print(f"alignment    : {t_align:.2f} s  ({T / _FPS / t_align:.1f}x real time)")
    print(f"loglik       : {ll / T:.2f} nats/frame")

    obs = np.array([p in OBS for p in phones])
    if obs.any():
        print(f"obstruents   : median {np.median(durs[obs]) * 1000:.0f} ms, "
              f"p90 {np.percentile(durs[obs], 90) * 1000:.0f} ms, "
              f"max {durs[obs].max() * 1000:.0f} ms")
    print(f"phone durs   : median {np.median(durs) * 1000:.0f} ms, "
          f"max {durs.max() * 1000:.0f} ms")

    if args.ref_durs:
        ref = np.array([float(x) for x in parse_list(args.ref_durs)])
        if len(ref) != len(phones):
            sys.exit(f"--ref-durs has {len(ref)} values, expected {len(phones)}")
        ref_b = np.round(np.concatenate([[0.0], np.cumsum(ref)]) / ref.sum() * T)
        d = np.abs(starts[1:] - ref_b[1:-1]) * MS
        print(f"vs reference : median {np.median(d):.0f} ms, "
              f"mean {d.mean():.0f} ms, within 50 ms {100 * (d <= 50).mean():.0f}%")


if __name__ == "__main__":
    main()
