#!/usr/bin/env python3
"""
bench/bench_align_speed.py

pyshiro の HSMM 強制アライメントの速度・メモリ計測。

2 通りの使い方がある:

  1) 実音声で計測
     python bench/bench_align_speed.py --wav song.wav --lyric song.txt \
         --model models/xxx.hsmm --phonemap models/xxx_phonemap.json

  2) 長さを変えてスケーリングを計測（モデルの GMM から決定的に合成した特徴量）
     python bench/bench_align_speed.py --sweep 10,20,60,120,300 \
         --model pyshiro/data/models/pyshiro-jp-v2.hsmm \
         --phonemap pyshiro/data/models/pyshiro-jp-v2_phonemap.json

`--mem-mb` に小さい値を渡すと Hirschberg 分割統治経路を強制できる
（--mem-mb 0 で常に分割統治）。
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pyshiro
from pyshiro.align import FPS, build_state_sequence, forced_align_2pass, load_phonemap
from pyshiro.model import load_hsmm


def _peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _synth(model, state_seq, T: int, seed: int = 0):
    """状態 GMM から決定的に特徴量を合成する（速度計測用）。"""
    rng = np.random.default_rng(seed)
    w = np.array([max(1.0, model.durations[s.dur_idx].mean) for s in state_seq])
    dur = np.maximum(1, np.round(w / w.sum() * T)).astype(np.int64)
    while dur.sum() > T:
        i = int(np.argmax(dur))
        if dur[i] <= 1:
            raise RuntimeError("状態数が多すぎます")
        dur[i] -= 1
    dur[int(np.argmax(dur))] += T - dur.sum()
    bounds = np.concatenate([[0], np.cumsum(dur)])

    streams = []
    for k in range(model.nstream):
        x = np.zeros((T, model.ndim), dtype=np.float64)
        for n, s in enumerate(state_seq):
            t0, t1 = bounds[n], bounds[n + 1]
            if t1 <= t0:
                continue
            g = model.streams[k].gmms[s.out_idx]
            sd = np.sqrt(np.maximum(g.vars[0], 1e-6))
            x[t0:t1] = g.means[0] + sd * rng.standard_normal((t1 - t0, model.ndim))
        x += 0.5 * rng.standard_normal((T, model.ndim))
        streams.append(x)
    return streams


def _phonemes_for(T: int, table) -> list:
    base = pyshiro.text_to_phonemes("あかいぐつ\nあるいてた\nゆきのあさ\nひとりきり", table)
    body = base[1:-1]
    target = max(4, T // 12)          # 1 音素あたり平均 12 フレーム
    out: list = []
    while len(out) < target:
        out.extend(body)
    return ["pau"] + out[:target] + ["pau"]


def run_case(model, phonemap, table, streams, mem_mb: int, label: str):
    T = streams[0].shape[0]
    phonemes = _phonemes_for(T, table)
    state_seq = build_state_sequence(phonemes, phonemap, T)

    t0 = time.perf_counter()
    segs, ll = forced_align_2pass(model, streams, state_seq,
                                  nodur_phonemes={"pau", "br"},
                                  backtrace_mem_mb=mem_mb)
    dt = time.perf_counter() - t0

    print(f"{label:>10}  T={T:>7}  N={len(state_seq):>6}  "
          f"{dt:8.2f} s  {T/dt:9.0f} frame/s  "
          f"peak={_peak_mb():8.1f} MB  loglik/frame={ll/T:9.3f}")
    return dt, _peak_mb(), segs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--phonemap", type=Path, required=True)
    ap.add_argument("--wav", type=Path, default=None)
    ap.add_argument("--lyric", type=Path, default=None)
    ap.add_argument("--sweep", type=str, default="10,20,60",
                    help="合成音声の長さ（秒、カンマ区切り）")
    ap.add_argument("--mem-mb", type=int, default=1024,
                    help="バックポインタに許すメモリ上限（MiB）。0 で分割統治を強制")
    args = ap.parse_args()

    model = load_hsmm(args.model)
    phonemap = load_phonemap(args.phonemap)
    table = pyshiro.load_table()

    if args.wav is not None:
        if args.lyric is None:
            ap.error("--wav と --lyric はセットで指定してください")
        streams = pyshiro.extract_mfcc_from_file(args.wav)
        phonemes = pyshiro.convert_lyric_file(args.lyric, table)
        T = streams[0].shape[0]
        state_seq = build_state_sequence(phonemes, phonemap, T)
        t0 = time.perf_counter()
        segs, ll = forced_align_2pass(model, streams, state_seq,
                                      nodur_phonemes={"pau", "br"},
                                      backtrace_mem_mb=args.mem_mb)
        dt = time.perf_counter() - t0
        print(f"{args.wav.name}: {dt:.2f} s ({T/dt:.0f} frame/s), "
              f"peak={_peak_mb():.1f} MB, loglik/frame={ll/T:.3f}")
        return

    print(f"{'case':>10}  {'T':>8}  {'N':>7}  {'time':>9}  {'speed':>12}"
          f"  {'peak mem':>12}")
    for sec in [float(s) for s in args.sweep.split(",") if s.strip()]:
        T = int(round(sec * FPS))
        phonemes = _phonemes_for(T, table)
        state_seq = build_state_sequence(phonemes, phonemap, T)
        streams = _synth(model, state_seq, T)
        run_case(model, phonemap, table, streams, args.mem_mb, f"{sec:.0f}s")


if __name__ == "__main__":
    main()
