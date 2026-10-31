"""
Viterbi カーネルの厳密一致テスト。

最適化後の 3 経路
  (1) 通常 DP（バックポインタ int32 1 枚）
  (2) Hirschberg 分割統治（メモリ O(K·T)）
  (3) 参照実装 `_viterbi_dp`（最適化前のコード）
が **同一の区間・同一の対数尤度** を返すことを確認する。

topology（type-b / type-c / skip-boundary）と pskip を含む音素列も対象。
"""

from pathlib import Path

import numpy as np
import pytest

import pyshiro
from pyshiro.align import (
    build_state_sequence, forced_align, load_phonemap,
    _viterbi_dp, _decode_bp,
)
from pyshiro.model import load_hsmm
from pyshiro.bundled import bundled_model_path

HSMM, PHONEMAP = bundled_model_path("jp-v2")


# ---------------------------------------------------------------------------
# 参照実装（最適化前）のバックトラック
# ---------------------------------------------------------------------------

def _reference_align(model, streams, state_seq, use_duration,
                     initial_segments=None, window=60,
                     nodur_phonemes=None, hmm_cap=None):
    from pyshiro.align import compute_frame_loglik

    T = streams[0].shape[0]
    N = len(state_seq)

    unique_out = sorted({s.out_idx for s in state_seq})
    out_to_row = {idx: r for r, idx in enumerate(unique_out)}
    cum = np.zeros((len(unique_out), T + 1), dtype=np.float64)
    for idx in unique_out:
        cum[out_to_row[idx], 1:] = np.cumsum(compute_frame_loglik(model, idx, streams))

    floors = np.array([s.floor for s in state_seq], dtype=np.int32)
    ceils  = np.array([min(s.ceil, T) for s in state_seq], dtype=np.int32)
    if not use_duration and hmm_cap is not None:
        eff = max(hmm_cap, int(np.ceil(T / N)))
        for n in range(N):
            if nodur_phonemes and state_seq[n].phoneme in nodur_phonemes:
                continue
            ceils[n] = min(ceils[n], eff)

    out_row   = np.array([out_to_row[s.out_idx] for s in state_seq], dtype=np.int32)
    dur_means = np.array([model.durations[s.dur_idx].mean for s in state_seq], dtype=np.float64)
    dur_vars  = np.array([model.durations[s.dur_idx].var  for s in state_seq], dtype=np.float64)
    if nodur_phonemes:
        for n, s in enumerate(state_seq):
            if s.phoneme in nodur_phonemes:
                dur_vars[n] = 0.0

    max_topo  = max(max((len(s.topo_sources) for s in state_seq), default=0), 1)
    max_pskip = max(max((len(s.pskip_sources) for s in state_seq), default=0), 1)
    topo_src  = np.full((N, max_topo), -1, dtype=np.int32)
    topo_log  = np.zeros((N, max_topo), dtype=np.float64)
    pskip_src = np.full((N, max_pskip), -1, dtype=np.int32)
    pskip_log = np.zeros((N, max_pskip), dtype=np.float64)
    for n, s in enumerate(state_seq):
        for ki, (src, lp) in enumerate(s.topo_sources):
            topo_src[n, ki] = src
            topo_log[n, ki] = lp
        for ki, (src, lp) in enumerate(s.pskip_sources):
            pskip_src[n, ki] = src
            pskip_log[n, ki] = lp

    t_lo = np.empty(N, dtype=np.int32)
    t_hi = np.empty(N, dtype=np.int32)
    for n, sn in enumerate(state_seq):
        if initial_segments is None:
            t_lo[n] = sn.floor
            t_hi[n] = T
        elif nodur_phonemes and sn.phoneme in nodur_phonemes:
            t_lo[n] = sn.floor
            t_hi[n] = T
        else:
            e0 = initial_segments[n][1]
            t_lo[n] = max(sn.floor, e0 - window)
            t_hi[n] = min(T, e0 + window if n < N - 1 else T)

    score, back_d, back_src = _viterbi_dp(
        N, T, cum, out_row, floors, ceils, dur_means, dur_vars,
        t_lo, t_hi, use_duration, topo_src, topo_log, pskip_src, pskip_log)

    segs = [None] * N
    t, n = T, N - 1
    while n >= 0:
        d, src = int(back_d[n, t]), int(back_src[n, t])
        if d == 0:
            for k in range(src + 1, n + 1):
                segs[k] = (t, t)
            n = src
        else:
            start = t - d
            segs[n] = (start, t)
            for k in range(src + 1, n):
                segs[k] = (start, start)
            n, t = src, start
    return segs, float(score[N - 1, T])


# ---------------------------------------------------------------------------
# テスト用データ
# ---------------------------------------------------------------------------

def _make_case(seconds=8.0, seed=0, lyric="あかいぐつ あるいてた"):
    """モデル GMM から決定的に合成した特徴量を作る。"""
    model = load_hsmm(HSMM)
    phonemap = load_phonemap(PHONEMAP)
    table = pyshiro.load_table()

    phonemes = pyshiro.text_to_phonemes(lyric, table)
    T = int(round(seconds * 200))
    state_seq = build_state_sequence(phonemes, phonemap, T)

    rng = np.random.default_rng(seed)
    streams = []
    for k in range(model.nstream):
        base = np.zeros((T, model.ndim), dtype=np.float64)
        # 状態ごとに平均を切り替えた緩やかな軌跡 + ノイズ
        gmm = model.streams[k].gmms[state_seq[0].out_idx]
        mu = gmm.means[0]
        base[:] = mu
        step = max(1, T // len(state_seq))
        for i, s in enumerate(state_seq):
            t0 = i * step
            t1 = min(T, t0 + step)
            if t1 <= t0:
                break
            base[t0:t1] = model.streams[k].gmms[s.out_idx].means[0]
        # 時間方向にぼかして連続性を持たせる
        for d in range(1, 4):
            base[d:] += 0.3 * base[:-d]
        base /= 1.9
        base += 0.8 * rng.standard_normal((T, model.ndim))
        streams.append(base)
    return model, phonemes, phonemap, state_seq, streams


def _variant_phonemap(phonemap, topo=None, pskip=()):
    pm = {k: dict(v) for k, v in phonemap.items()}
    if topo:
        for ph, tp in topo.items():
            if ph in pm:
                pm[ph]["topology"] = tp
    for ph in pskip:
        if ph in pm:
            pm[ph]["pskip"] = 0.5
    return pm


CASES = [
    ("default",        {},                        ()),
    ("type-b",         {"k": "type-b", "a": "type-b"}, ()),
    ("type-c",         {"k": "type-c", "a": "type-c"}, ()),
    ("skip-boundary",  {"k": "skip-boundary", "a": "skip-boundary"}, ()),
    ("pskip",          {},                        ("cl", "t")),
    ("topo+pskip",     {"k": "type-b", "a": "type-c"}, ("cl", "t")),
]


# ---------------------------------------------------------------------------
# テスト本体
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,topo,pskip", CASES)
def test_two_pass_matches_reference(name, topo, pskip):
    model, phonemes, phonemap, _, streams = _make_case()
    pm = _variant_phonemap(phonemap, topo, pskip)
    T  = streams[0].shape[0]
    state_seq = build_state_sequence(phonemes, pm, T)

    nodur = {"pau", "br"}
    ref, ref_ll = _reference_align(model, streams, state_seq, False,
                                   nodur_phonemes=nodur)
    ref2, ref_ll2 = _reference_align(model, streams, state_seq, True,
                                     initial_segments=ref, nodur_phonemes=nodur)
    got, got_ll = forced_align(model, streams, state_seq, use_duration=False,
                               nodur_phonemes={"pau", "br"})
    assert got == ref, f"{name}: pass1 不一致"
    assert got_ll == ref_ll, f"{name}: pass1 対数尤度不一致"

    got2, got_ll2 = forced_align(model, streams, state_seq, use_duration=True,
                                 initial_segments=got, nodur_phonemes={"pau", "br"})
    assert got2 == ref2, f"{name}: pass2 不一致"
    assert got_ll2 == ref_ll2, f"{name}: pass2 対数尤度不一致"


@pytest.mark.parametrize("name,topo,pskip", CASES)
def test_hirschberg_matches_full(name, topo, pskip):
    """分割統治経路（メモリ上限 0）が通常 DP と完全一致すること。"""
    model, phonemes, phonemap, _, streams = _make_case(6.0, seed=1)
    pm = _variant_phonemap(phonemap, topo, pskip)
    T  = streams[0].shape[0]
    state_seq = build_state_sequence(phonemes, pm, T)

    for use_dur in (False, True):
        init = None
        if use_dur:
            init, _ = forced_align(model, streams, state_seq, use_duration=False,
                                   nodur_phonemes={"pau", "br"})
        a, la = forced_align(model, streams, state_seq, use_duration=use_dur,
                             initial_segments=init, nodur_phonemes={"pau", "br"},
                             backtrace_mem_mb=512)
        b, lb = forced_align(model, streams, state_seq, use_duration=use_dur,
                             initial_segments=init, nodur_phonemes={"pau", "br"},
                             backtrace_mem_mb=0)
        assert a == b, f"{name} use_dur={use_dur}: 区間不一致"
        assert la == lb, f"{name} use_dur={use_dur}: 対数尤度不一致"


def test_no_skip_edges():
    """スキップ辺が無い素朴なケースでも一致すること。"""
    model, phonemes, phonemap, _, streams = _make_case(4.0, seed=2)
    T = streams[0].shape[0]
    state_seq = build_state_sequence(phonemes, phonemap, T)
    for s in state_seq:
        s.topo_sources = []
        s.pskip_sources = []
    ref, ref_ll = _reference_align(model, streams, state_seq, True)
    got, got_ll = forced_align(model, streams, state_seq, use_duration=True)
    assert got == ref
    assert got_ll == ref_ll
