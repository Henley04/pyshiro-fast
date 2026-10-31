"""
align.py

SHIRO 互換 HSMM 強制アライメント（Viterbi デコーディング）

phonemap.json で定義された音素→状態マッピングと
.hsmm モデルの GMM + 継続時間分布を使い、
音素列の各音素がどのフレーム区間に対応するかを推定する。
"""

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

import numpy as np
import numba

from pyshiro.model import HSMMModel, GMM

# フレームレート (hop=80, sr=16000)
HOP_TIME = 80 / 16000   # 0.005 s/frame
FPS      = 1.0 / HOP_TIME  # 200 frames/sec

NEG_INF = -1e30


# ---------------------------------------------------------------------------
# phonemap 読み込み
# ---------------------------------------------------------------------------

def load_phonemap(path: Path) -> dict:
    """phonemap.json を読み込み phone_map dict を返す。"""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data['phone_map']


# ---------------------------------------------------------------------------
# 状態列の構築
# ---------------------------------------------------------------------------

@dataclass
class StateInfo:
    phoneme:       str
    state_idx:     int   # 音素内での状態番号 (0/1/2)
    dur_idx:       int   # モデルの継続時間状態インデックス
    out_idx:       int   # GMM インデックス（全ストリーム共通）
    floor:         int   # 最小継続フレーム数
    ceil:          int   # 最大継続フレーム数（0=無制限）
    pskip_sources: list  = field(default_factory=list)
    # ゼロフレームスキップ: [(src_state_idx, log_p), ...]  ← pskip 用
    topo_sources:  list  = field(default_factory=list)
    # 継続時間保存スキップ: [(src_state_idx, log_p), ...]  ← topology 用


def build_state_sequence(phonemes: List[str], phonemap: dict,
                         total_frames: int) -> List[StateInfo]:
    """
    音素列 → StateInfo のリストに展開する。
    各音素は 3 状態（デフォルト）を持つ。

    スキップエッジ:
    - pskip (ゼロフレーム) : phonemap に "pskip" > 0 が指定された音素を丸ごと省略。
      前の音素の最終状態 → 次の音素の先頭状態（フレーム消費なし）。
    - topology（継続時間保存）: phonemap に "topology" が指定された音素の内部構造。
      - "type-a" (デフォルト): スキップなし
      - "type-b": 各状態 k から最終状態へのスキップ
      - "type-c": 各状態 k から状態 k+2 へのスキップ
      - "skip-boundary": 前の音素最終状態→現音素状態1、現音素状態(nst-2)→次音素先頭
    """
    states: List[StateInfo] = []
    phoneme_starts: List[int] = []

    def _get_entry(ph: str) -> dict:
        """phonemap からエントリを取得する。
        トライフォン 'L-ph+R' が無ければモノフォン 'ph' にフォールバックする。"""
        if ph in phonemap:
            return phonemap[ph]
        # 中心音素を抽出してフォールバック
        core = ph
        if '+' in core:
            core = core.split('+')[0]
        if '-' in core:
            core = core.split('-')[1]
        if core in phonemap:
            return phonemap[core]
        raise ValueError(
            f"音素 '{ph}' が phonemap に存在しません"
            + (f"（モノフォン '{core}' も見つかりません）" if core != ph else "")
        )

    # パス1: 全状態を構築（スキップエッジなし）
    for ph in phonemes:
        entry  = _get_entry(ph)
        nst    = len(entry['states'])
        floors = entry.get('durfloor', [0.0] * nst)
        ceils  = entry.get('durceil',  [0.0] * nst)

        phoneme_starts.append(len(states))

        for k, st in enumerate(entry['states']):
            fl_sec = floors[k] if k < len(floors) else 0.0
            ce_sec = ceils[k]  if k < len(ceils)  else 0.0

            floor_f = max(1, round(fl_sec * FPS)) if fl_sec > 0 else 1
            ceil_f  = round(ce_sec * FPS)          if ce_sec > 0 else total_frames

            states.append(StateInfo(
                phoneme   = ph,
                state_idx = k,
                dur_idx   = st['dur'],
                out_idx   = st['out'][0],
                floor     = floor_f,
                ceil      = ceil_f,
            ))

    # パス2: スキップエッジを設定
    for i, ph in enumerate(phonemes):
        entry    = _get_entry(ph)
        nst      = len(entry['states'])
        ph_start = phoneme_starts[i]

        # --- pskip（ゼロフレームスキップ）---
        # 音素 i に pskip > 0 があれば前の音素最終状態 → 次の音素先頭状態
        pskip = float(entry.get('pskip', 0.0))
        if pskip > 0.0 and 0 < i < len(phonemes) - 1:
            src = phoneme_starts[i] - 1       # 前の音素の最終状態
            dst = phoneme_starts[i + 1]       # 次の音素の先頭状態
            states[dst].pskip_sources.append((src, math.log(max(pskip, 1e-30))))

        # --- topology（継続時間保存スキップ）---
        topology = entry.get('topology', 'type-a')

        if topology == 'type-b':
            # 状態 k → 最終状態（状態 nst-1）へのスキップ（k=0..nst-3）
            # SHIRO: for k=1..nst-2: state[ph_start+k-1].jmp += {d=nst-k}
            for k in range(nst - 2):          # k=0..nst-3 (0-indexed)
                src = ph_start + k
                dst = ph_start + nst - 1
                states[dst].topo_sources.append((src, 0.0))

        elif topology == 'type-c':
            # 状態 k → 状態 k+2 へのスキップ（k=0..nst-3）
            # SHIRO: for k=1..nst-2: state[ph_start+k-1].jmp += {d=2}
            for k in range(nst - 2):
                src = ph_start + k
                dst = ph_start + k + 2
                states[dst].topo_sources.append((src, 0.0))

        elif topology == 'skip-boundary':
            # 前の音素最終状態 → 現音素状態1（現音素の先頭をスキップ）
            # SHIRO: states[#states-nst].jmp += {d=2}
            if i > 0:
                src = phoneme_starts[i] - 1   # 前の音素最終状態
                dst = ph_start + 1            # 現音素状態1
                if dst < len(states):
                    states[dst].topo_sources.append((src, 0.0))
            # 現音素状態(nst-2) → 次の音素先頭（現音素末尾をスキップ）
            # SHIRO: states[#states-1].jmp += {d=2}
            if i + 1 < len(phonemes):
                src = ph_start + nst - 2      # 現音素の後ろから2番目の状態
                dst = phoneme_starts[i + 1]   # 次の音素先頭
                states[dst].topo_sources.append((src, 0.0))

    return states


# ---------------------------------------------------------------------------
# フレームレベル対数尤度の計算
# ---------------------------------------------------------------------------

def _gmm_loglik_frames(gmm: GMM, obs: np.ndarray) -> np.ndarray:
    """
    単一 GMM の全フレームに対する対数尤度を返す。

    Parameters
    ----------
    gmm : GMM  (nmix=1 を想定)
    obs : (T, ndim)

    Returns
    -------
    loglik : (T,)
    """
    T = obs.shape[0]
    # log-sum-exp の accumulator は log(0) = -inf で初期化する。
    # 0（= log 1）で始めると log(Σ w·N) ではなく log(1 + Σ w·N) を計算してしまい、
    # 対数尤度が +0.0 で下げ止まる。すると「このフレームはこの状態ではない」という
    # 強い否定ができなくなり、識別が必要な音素境界でのみアライメントが破綻する。
    loglik = np.full(T, -np.inf, dtype=np.float64)

    for m in range(gmm.nmix):
        mu  = gmm.means[m]        # (ndim,)
        var = gmm.vars[m]         # (ndim,)
        var = np.maximum(var, 1e-6)  # ゼロ分散を防ぐ

        diff = obs - mu           # (T, ndim)
        # log N(x | mu, diag(var)) = -0.5 * [sum_j (x_j-mu_j)^2/var_j + log(2pi*var_j)]
        log_norm = -0.5 * (np.sum(diff ** 2 / var, axis=1)
                           + np.sum(np.log(2 * math.pi * var)))
        if gmm.nmix == 1:
            loglik = log_norm
        else:
            log_w = math.log(max(gmm.weights[m], 1e-30))
            loglik = np.logaddexp(loglik, log_w + log_norm)

    return loglik


def compute_frame_loglik(model: HSMMModel, out_idx: int,
                         streams: List[np.ndarray]) -> np.ndarray:
    """
    あるモデル状態（GMM インデックス = out_idx）における
    全フレームの対数尤度（全ストリーム加重和）を返す。

    Parameters
    ----------
    model   : HSMMModel
    out_idx : int  GMM インデックス
    streams : list of (T, 12)  各ストリームの特徴量

    Returns
    -------
    loglik : (T,)
    """
    T      = streams[0].shape[0]
    loglik = np.zeros(T, dtype=np.float64)

    for l, stream in enumerate(model.streams):
        gmm = stream.gmms[out_idx]
        ll  = _gmm_loglik_frames(gmm, streams[l].astype(np.float64))
        loglik += stream.weight * ll

    return loglik


# ---------------------------------------------------------------------------
# 継続時間対数尤度
# ---------------------------------------------------------------------------

def _log_dur_prob(d: int, dur_mean: float, dur_var: float) -> float:
    """
    ガウス継続時間分布の対数確率。
    var <= 0 の場合は一様分布（0 を返す）。
    """
    if dur_var <= 0:
        return 0.0
    return -0.5 * ((d - dur_mean) ** 2 / dur_var
                   + math.log(2 * math.pi * dur_var))


# ---------------------------------------------------------------------------
# Viterbi DP カーネル（Numba JIT）
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _viterbi_dp(N, T,
                cum_ll_mat,   # (n_unique_out, T+1) float64
                out_row,      # (N,) int32  — cum_ll_mat の行インデックス
                floors,       # (N,) int32
                ceils,        # (N,) int32
                dur_means,    # (N,) float64
                dur_vars,     # (N,) float64
                t_lo_arr,     # (N,) int32
                t_hi_arr,     # (N,) int32
                use_duration,
                topo_src,     # (N, max_topo) int32  -1=番兵  topology スキップ起点
                topo_log,     # (N, max_topo) float64          log(skip prob)
                pskip_src,    # (N, max_pskip) int32 -1=番兵  pskip ゼロフレーム起点
                pskip_log):   # (N, max_pskip) float64         log(pskip)
    """HSMM Viterbi DP の内側ループ（純粋 numpy 配列のみ受け取る）。

    スキップ遷移:
    - topo: 継続時間保存。状態 n が通常の d フレームを消費しつつ、
            n-1 以外の先行状態から遷移する。back_src に先行状態を記録。
    - pskip: ゼロフレーム。フレーム消費なしで src → n に遷移。
             back_d=0 かつ back_src=src で識別する。

    Returns
    -------
    score    : (N, T+1) float64
    back_d   : (N, T+1) int32  — 継続フレーム数（pskip 時は 0）
    back_src : (N, T+1) int32  — 先行状態インデックス（通常時は n-1）
    """
    NEG_INF  = -1e30
    TWO_PI   = 2.0 * math.pi
    max_topo  = topo_src.shape[1]
    max_pskip = pskip_src.shape[1]

    score    = np.full((N, T + 1), NEG_INF)
    back_d   = np.zeros((N, T + 1), dtype=np.int32)
    back_src = np.full((N, T + 1), -1, dtype=np.int32)

    # --- 初期化（状態 0、フレーム 0 から始まる） ---
    fl0 = floors[0]
    ce0 = ceils[0]
    for t in range(t_lo_arr[0], t_hi_arr[0] + 1):
        d = t
        if d < fl0 or d > ce0:
            continue
        s = cum_ll_mat[out_row[0], t] - cum_ll_mat[out_row[0], 0]
        if use_duration:
            dv = dur_vars[0]
            if dv > 0.0:
                dm = dur_means[0]
                s += -0.5 * ((d - dm) ** 2 / dv + math.log(TWO_PI * dv))
        score[0, t]    = s
        back_d[0, t]   = d
        back_src[0, t] = -1   # 先行なし（始端）

    # --- 再帰 ---
    for n in range(1, N):
        fl  = floors[n]
        ce  = ceils[n]
        row = out_row[n]
        dm  = dur_means[n]
        dv  = dur_vars[n]
        use_dur_n = use_duration and dv > 0.0
        log_denom = math.log(TWO_PI * dv) if use_dur_n else 0.0

        for t in range(t_lo_arr[n], t_hi_arr[n] + 1):
            best_score = NEG_INF
            best_d     = 0
            best_src   = n - 1
            d_max = ce if ce < t else t

            # 通常遷移 + topology スキップ（どちらも d フレームを消費）
            for d in range(fl, d_max + 1):
                prev_t = t - d
                ll = cum_ll_mat[row, t] - cum_ll_mat[row, prev_t]
                if use_dur_n:
                    ll += -0.5 * ((d - dm) ** 2 / dv + log_denom)

                # 前の状態 n-1 から（通常遷移）
                ps = score[n - 1, prev_t]
                if ps > NEG_INF:
                    lp = ps + ll
                    if lp > best_score:
                        best_score = lp
                        best_d     = d
                        best_src   = n - 1

                # topology スキップ: 別の先行状態から同じ d フレーム消費で遷移
                for ki in range(max_topo):
                    src = topo_src[n, ki]
                    if src < 0:
                        break
                    ps = score[src, prev_t]
                    if ps > NEG_INF:
                        lp = ps + topo_log[n, ki] + ll
                        if lp > best_score:
                            best_score = lp
                            best_d     = d
                            best_src   = src

            # pskip ゼロフレームスキップ（フレーム消費なし）
            for ki in range(max_pskip):
                src = pskip_src[n, ki]
                if src < 0:
                    break
                ps = score[src, t]
                if ps > NEG_INF:
                    lp = ps + pskip_log[n, ki]
                    if lp > best_score:
                        best_score = lp
                        best_d     = 0   # ゼロフレーム
                        best_src   = src

            score[n, t]    = best_score
            back_d[n, t]   = best_d
            back_src[n, t] = best_src

    return score, back_d, back_src


# ---------------------------------------------------------------------------
# 高速 Viterbi DP カーネル（Numba JIT）
#
# 上の `_viterbi_dp` は参照実装として残してある（互換性検証用）。
# (N, T+1) の score / back_d / back_src を 3 枚確保し、内側ループが
# d ∈ [floor, min(ceil, t)] を総当りするため、長尺音声では
#   時間 O(N·T²)、メモリ O(N·T)·16 byte
# になる。以下は **結果を 1 フレームも変えずに** 次を適用した実装:
#
#   1. d の探索範囲を「先行状態が到達しうる時間窓」との積に限定（厳密な枝刈り）
#   2. HMM パス（継続時間分布なし）は窓付き最大値の逐次更新で 1 セル O(1)
#      → 総当り O(T) を O(1) に。値は原式 ps + (cum[t]-cum[p]) のまま計算する
#   3. バックポインタを int32 1 枚に詰める（d:29bit + 先行状態オフセット:3bit）
#      score 行列は直前 K 行のリングバッファのみ
#   4. それでも収まらない巨大入力では Hirschberg 型の分割統治で
#      バックトラックを実行し、メモリを O(K·T)（系列長に線形）に落とす
# ---------------------------------------------------------------------------

BP_SHIFT = 3                     # bp 下位ビットに (n - src) を格納
BP_MASK  = (1 << BP_SHIFT) - 1


@numba.njit(cache=True)
def _dp_forward(n_base, N, W, t_base,
                cum_ll,        # (n_row, T_total+1) 累積対数尤度
                out_row,       # (N_total,) int32
                floors,        # (N_total,) int32
                ceils,         # (N_total,) int32
                dur_means,     # (N_total,) float64
                dur_vars,      # (N_total,) float64
                t_lo,          # (N_total,) int32  絶対フレーム
                t_hi,          # (N_total,) int32
                use_duration,
                topo_src,      # (N_total, max_topo) int32
                topo_log,      # (N_total, max_topo) float64
                pskip_src,     # (N_total, max_pskip) int32
                pskip_log,     # (N_total, max_pskip) float64
                K,             # リングバッファ行数（= 最大参照段数 + 1）
                bp,            # (N, W+1) int32  ローカル添字
                store_bp):
    """状態 n_base .. n_base+N-1、時刻 [t_base, t_base+W] の前向き Viterbi。

    戻り値は最終状態（n_base+N-1）のスコア列（長さ W+1、q = 0..W）。
    bp には (d << BP_SHIFT) | (n - src) を格納する。
    """
    NEG    = -1e30
    TWO_PI = 2.0 * math.pi
    max_topo  = topo_src.shape[1]
    max_pskip = pskip_src.shape[1]
    ncol      = W + 1
    nsrc_max  = 1 + max_topo

    sbuf  = np.full((K, ncol), NEG)
    last  = np.full(ncol, NEG)
    srcs  = np.empty(nsrc_max, dtype=np.int32)
    slgs  = np.empty(nsrc_max, dtype=np.float64)
    bestv = np.empty(nsrc_max, dtype=np.float64)
    bestp = np.empty(nsrc_max, dtype=np.int32)
    bests = np.empty(nsrc_max, dtype=np.float64)   # score[bestp]
    bestc = np.empty(nsrc_max, dtype=np.float64)   # cum[bestp]
    lastp = np.empty(nsrc_max, dtype=np.int32)

    # --- 初期状態: n_base が t_base から始まり q フレーム伸びる ---
    g0  = n_base
    r0  = out_row[g0]
    fl0 = floors[g0]
    ce0 = ceils[g0]
    lo0 = t_lo[g0] - t_base
    if lo0 < 0:
        lo0 = 0
    hi0 = t_hi[g0] - t_base
    if hi0 > W:
        hi0 = W
    dv0     = dur_vars[g0]
    dm0     = dur_means[g0]
    use_d0  = use_duration and dv0 > 0.0
    ld0     = math.log(TWO_PI * dv0) if use_d0 else 0.0
    for q in range(lo0, hi0 + 1):
        d = q
        if d < fl0 or d > ce0:
            continue
        sc = cum_ll[r0, t_base + q] - cum_ll[r0, t_base]
        if use_d0:
            sc += -0.5 * ((d - dm0) * (d - dm0) / dv0 + ld0)
        sbuf[0, q] = sc
        if store_bp:
            bp[0, q] = np.int32((d << BP_SHIFT) | 1)

    # --- 再帰 ---
    for lni in range(1, N):
        g  = n_base + lni
        fl = floors[g]
        ce = ceils[g]
        rw = out_row[g]
        dm = dur_means[g]
        dv = dur_vars[g]
        use_dur = use_duration and dv > 0.0
        ld = math.log(TWO_PI * dv) if use_dur else 0.0

        lo_n = t_lo[g] - t_base
        if lo_n < 0:
            lo_n = 0
        hi_n = t_hi[g] - t_base
        if hi_n > W:
            hi_n = W

        if hi_n < lo_n:
            continue

        for q in range(ncol):
            sbuf[lni % K, q] = NEG

        # 先行状態（通常遷移 + topology スキップ）
        ns = 1
        srcs[0] = g - 1
        slgs[0] = 0.0
        for ki in range(max_topo):
            s = topo_src[g, ki]
            if s < 0:
                break
            srcs[ns] = s
            slgs[ns] = topo_log[g, ki]
            ns += 1

        lo_min = t_lo[srcs[0]] - t_base
        hi_max = t_hi[srcs[0]] - t_base
        for k in range(1, ns):
            v = t_lo[srcs[k]] - t_base
            if v < lo_min:
                lo_min = v
            v = t_hi[srcs[k]] - t_base
            if v > hi_max:
                hi_max = v
        if lo_min < 0:
            lo_min = 0
        if hi_max > W:
            hi_max = W

        if (not use_dur) and ce >= hi_n - lo_min:
            # --- 継続時間分布なし: 窓付き最大値の逐次更新（1 セル O(1)） ---
            # 窓は p ∈ [t_lo[src], min(t - fl, t_hi[src])]。
            # ce >= hi_n - lo_min より下限は常に t_lo[src] 以下なので
            # 「p を増やしながらの最大値更新」で厳密に一致する。
            for k in range(ns):
                bestv[k] = NEG
                bestp[k] = -1
                bests[k] = NEG
                bestc[k] = 0.0
                v = t_lo[srcs[k]] - t_base
                if v < 0:
                    v = 0
                lastp[k] = v - 1

            for q in range(lo_n, hi_n + 1):
                cum_t    = cum_ll[rw, t_base + q]
                best_sc  = NEG
                best_d   = 0
                best_src = g - 1
                best_pp  = -1
                for k in range(ns):
                    s    = srcs[k]
                    sr   = (s - n_base) % K
                    hi_s = t_hi[s] - t_base
                    if hi_s > q:
                        hi_s = q
                    lim = q - fl
                    if lim > hi_s:
                        lim = hi_s
                    p = lastp[k]
                    while p < lim:
                        p += 1
                        sv = sbuf[sr, p]
                        if sv > NEG:
                            vv = sv - cum_ll[rw, t_base + p]
                            if vv >= bestv[k] or bestp[k] < 0:
                                # 最大値更新時に score / cum を控えておく。
                                # 毎セル bestp を再参照するとランダムアクセスに
                                # なるため（長尺音声で顕著なボトルネック）。
                                bestv[k] = vv
                                bestp[k] = p
                                bests[k] = sv
                                bestc[k] = cum_ll[rw, t_base + p]
                    lastp[k] = p
                    if bestp[k] >= 0:
                        # 値は原式と同じ順序で計算する（丸め誤差を持ち込まない）
                        cand = bests[k] + (cum_t - bestc[k]) + slgs[k]
                        if cand > best_sc or (cand == best_sc and bestp[k] > best_pp):
                            best_sc  = cand
                            best_d   = q - bestp[k]
                            best_src = s
                            best_pp  = bestp[k]

                for ki in range(max_pskip):
                    s = pskip_src[g, ki]
                    if s < 0:
                        break
                    ps = sbuf[(s - n_base) % K, q]
                    if ps > NEG:
                        cand = ps + pskip_log[g, ki]
                        if cand > best_sc:
                            best_sc  = cand
                            best_d   = 0
                            best_src = s
                            best_pp  = q

                sbuf[lni % K, q] = best_sc
                if store_bp:
                    bp[lni, q] = np.int32((best_d << BP_SHIFT) | (g - best_src))
        else:
            # --- 継続時間分布あり、または窓が広い場合: d を直接走査 ---
            # ループ範囲は「先行状態の時間窓」との積に限定（厳密）。
            for q in range(lo_n, hi_n + 1):
                cum_t    = cum_ll[rw, t_base + q]
                best_sc  = NEG
                best_d   = 0
                best_src = g - 1

                dlo = fl
                v = q - hi_max
                if v > dlo:
                    dlo = v
                dhi = ce
                v = q - lo_min
                if v < dhi:
                    dhi = v
                if dhi > q:
                    dhi = q

                for d in range(dlo, dhi + 1):
                    p  = q - d
                    ll = cum_t - cum_ll[rw, t_base + p]
                    if use_dur:
                        ll += -0.5 * ((d - dm) * (d - dm) / dv + ld)
                    ps = sbuf[(g - 1 - n_base) % K, p]
                    if ps > NEG:
                        cand = ps + ll
                        if cand > best_sc:
                            best_sc  = cand
                            best_d   = d
                            best_src = g - 1
                    for k in range(1, ns):
                        s  = srcs[k]
                        ps = sbuf[(s - n_base) % K, p]
                        if ps > NEG:
                            cand = ps + slgs[k] + ll
                            if cand > best_sc:
                                best_sc  = cand
                                best_d   = d
                                best_src = s

                for ki in range(max_pskip):
                    s = pskip_src[g, ki]
                    if s < 0:
                        break
                    ps = sbuf[(s - n_base) % K, q]
                    if ps > NEG:
                        cand = ps + pskip_log[g, ki]
                        if cand > best_sc:
                            best_sc  = cand
                            best_d   = 0
                            best_src = s

                sbuf[lni % K, q] = best_sc
                if store_bp:
                    bp[lni, q] = np.int32((best_d << BP_SHIFT) | (g - best_src))

    for q in range(ncol):
        last[q] = sbuf[(N - 1) % K, q]
    return last


@numba.njit(cache=True)
def _decode_bp(bp, N, W, seg_start, seg_end):
    """パック済みバックポインタから状態ごとの区間を復元する（ローカル添字）。"""
    n = N - 1
    t = W
    while n >= 0:
        v   = int(bp[n, t])
        d   = v >> BP_SHIFT
        off = v & BP_MASK
        src = n - off
        if off == 0:            # 未到達セル（本来起こらない）: 無限ループ防止
            seg_start[n] = t
            seg_end[n]   = t
            break
        if d == 0:
            # pskip ゼロフレームスキップ: src -> n
            k = src + 1
            while k <= n:
                seg_start[k] = t
                seg_end[k]   = t
                k += 1
            n = src
        else:
            start = t - d
            seg_start[n] = start
            seg_end[n]   = t
            # topology スキップ: 中間状態は空区間
            k = src + 1
            while k < n:
                seg_start[k] = start
                seg_end[k]   = start
                k += 1
            n = src
            t = start


# ---------------------------------------------------------------------------
# Hirschberg 型 分割統治バックトラック
#
# (N, T) のバックポインタ行列を持たずに済ませるため、状態軸の中点 m で
# 問題を分割する。m を跨ぐ経路は必ず「状態 m-1 が時刻 t* で終わり、
# 状態 m が時刻 t* から始まる」形になるので、
#   前向き DP で F(t) = best(状態 m-1 が t で終わる)
#   後向き DP で B(t) = best(状態 m が t から始まる)
# を求め、F(t)+B(t) を最大化する t* で左右に再帰すれば全体最適が復元できる。
# 後向き DP は「状態順と時間軸を反転した前向き DP」に等しいので、
# `_dp_forward` をそのまま再利用する（遷移・継続時間の向きだけ反転）。
# ---------------------------------------------------------------------------

class _DPCtx:
    """1 回の forced_align 呼び出しで共有する DP メタデータ。"""

    __slots__ = ("cum_ll", "out_row", "floors", "ceils", "dur_means", "dur_vars",
                 "t_lo", "t_hi", "use_duration", "topo_src", "topo_log",
                 "pskip_src", "pskip_log", "K", "max_topo", "max_pskip",
                 "valid_split")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)

    def pick_split(self, n0: int, n1: int):
        """[n0, n1) を分割する境界 m（n0 < m < n1）を中点に最も近いものから選ぶ。

        スキップ辺（topology / pskip）が跨ぐ境界は選べない。
        """
        vs  = self.valid_split
        mid = (n0 + n1) // 2
        i   = int(np.searchsorted(vs, mid))
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(vs):
                m = int(vs[j])
                if n0 < m < n1 and (best is None or abs(m - mid) < abs(best - mid)):
                    best = m
        return best


def _dp_forward_call(ctx, n0, n1, q0, q1, bp, store_bp):
    """ctx を使って [n0,n1) × [q0,q1] の前向き DP を回す。"""
    return _dp_forward(n0, n1 - n0, q1 - q0, q0,
                       ctx.cum_ll, ctx.out_row, ctx.floors, ctx.ceils,
                       ctx.dur_means, ctx.dur_vars, ctx.t_lo, ctx.t_hi,
                       ctx.use_duration, ctx.topo_src, ctx.topo_log,
                       ctx.pskip_src, ctx.pskip_log, ctx.K, bp, store_bp)


def _prepare_reverse(ctx, m, n1, q1, W):
    """[m, n1) × [q0, q1] を逆向きに解くための問題データを組み立てる。

    戻り値の配列は逆順インデックス u = (n1-1) - n で並んでいる。
    """
    nst  = n1 - m
    gidx = np.arange(n1 - 1, m - 1, -1, dtype=np.int64)

    out_row = ctx.out_row[gidx].astype(np.int32)
    floors  = ctx.floors[gidx].astype(np.int32)
    ceils   = np.minimum(ctx.ceils[gidx], W).astype(np.int32)
    dmeans  = ctx.dur_means[gidx].copy()
    dvars   = ctx.dur_vars[gidx].copy()
    # 絶対フレーム -> 反転後の局所時刻 q = q1 - t
    t_lo = np.clip(q1 - ctx.t_hi[gidx], 0, W).astype(np.int32)
    t_hi = np.clip(q1 - ctx.t_lo[gidx], 0, W).astype(np.int32)

    # 逆向きの隣接リスト（元の辺 src -> dst は 逆問題では u_dst -> u_src）
    deg_t = np.zeros(nst, dtype=np.int64)
    deg_p = np.zeros(nst, dtype=np.int64)
    for dst in range(m, n1):
        for ki in range(ctx.max_topo):
            s = int(ctx.topo_src[dst, ki])
            if s < 0:
                break
            if s < m:
                continue
            deg_t[(n1 - 1) - s] += 1
        for ki in range(ctx.max_pskip):
            s = int(ctx.pskip_src[dst, ki])
            if s < 0:
                break
            if s < m:
                continue
            deg_p[(n1 - 1) - s] += 1

    mt = max(1, int(deg_t.max()) if nst else 1)
    mp = max(1, int(deg_p.max()) if nst else 1)
    topo_src  = np.full((nst, mt), -1, dtype=np.int32)
    topo_log  = np.zeros((nst, mt), dtype=np.float64)
    pskip_src = np.full((nst, mp), -1, dtype=np.int32)
    pskip_log = np.zeros((nst, mp), dtype=np.float64)
    nt  = np.zeros(nst, dtype=np.int64)
    nps = np.zeros(nst, dtype=np.int64)

    for dst in range(m, n1):
        u = (n1 - 1) - dst
        for ki in range(ctx.max_topo):
            s = int(ctx.topo_src[dst, ki])
            if s < 0:
                break
            if s < m:
                continue
            v = (n1 - 1) - s
            topo_src[v, nt[v]] = u
            topo_log[v, nt[v]] = ctx.topo_log[dst, ki]
            nt[v] += 1
        for ki in range(ctx.max_pskip):
            s = int(ctx.pskip_src[dst, ki])
            if s < 0:
                break
            if s < m:
                continue
            v = (n1 - 1) - s
            pskip_src[v, nps[v]] = u
            pskip_log[v, nps[v]] = ctx.pskip_log[dst, ki]
            nps[v] += 1

    # 逆向きの累積対数尤度: cum_r[u, q] = Σ_{τ=t1-q}^{t1-1} ll
    used  = np.unique(out_row)
    remap = {int(v): i for i, v in enumerate(used)}
    orow  = np.array([remap[int(v)] for v in out_row], dtype=np.int32)
    qs    = np.arange(W + 1, dtype=np.int64)
    cum_r = np.empty((len(used), W + 1), dtype=np.float64)
    for i, urow in enumerate(used):
        cum_r[i, :] = ctx.cum_ll[int(urow), q1] - ctx.cum_ll[int(urow), q1 - qs]

    return (cum_r, orow, floors, ceils, dmeans, dvars, t_lo, t_hi,
            topo_src, topo_log, pskip_src, pskip_log)


def _solve_block(ctx, n0, n1, q0, q1, seg_start, seg_end):
    """分割できない（スキップ辺が跨ぐ）小区間を通常 DP で解く。

    戻り値は区間終端 (n1-1, q1) のスコア。
    """
    nloc = n1 - n0
    W    = q1 - q0
    bp   = np.zeros((nloc, W + 1), dtype=np.int32)
    row  = _dp_forward_call(ctx, n0, n1, q0, q1, bp, True)
    ss = np.zeros(nloc, dtype=np.int64)
    se = np.zeros(nloc, dtype=np.int64)
    _decode_bp(bp, nloc, W, ss, se)
    for i in range(nloc):
        seg_start[n0 + i] = ss[i] + q0
        seg_end[n0 + i]   = se[i] + q0
    return float(row[W])


def _hirschberg_backtrace(ctx, N, T):
    """メモリ O(K·T) の分割統治バックトラック。"""
    seg_start = np.zeros(N, dtype=np.int64)
    seg_end   = np.zeros(N, dtype=np.int64)
    dummy     = np.zeros((1, 1), dtype=np.int32)

    # 最適スコアは通常 DP を 1 回だけ回して取得する（バックポインタは捨てる）。
    # こうすると分割統治経路でも対数尤度が原実装とビット単位で一致する。
    row0 = _dp_forward_call(ctx, 0, N, 0, T, dummy, False)
    best_score = float(row0[T])

    stack = [(0, N, 0, T)]

    while stack:
        n0, n1, q0, q1 = stack.pop()
        if n1 - n0 == 1:
            seg_start[n0] = q0
            seg_end[n0]   = q1
            continue

        m = ctx.pick_split(n0, n1)
        if m is None:
            _solve_block(ctx, n0, n1, q0, q1, seg_start, seg_end)
            continue

        W = q1 - q0
        F = _dp_forward_call(ctx, n0, m, q0, q1, dummy, False)
        (cum_r, orow_r, fl_r, ce_r, dm_r, dv_r, tlo_r, thi_r,
         ts_r, tl_r, ps_r, pl_r) = _prepare_reverse(ctx, m, n1, q1, W)
        BRev = _dp_forward(0, n1 - m, W, 0, cum_r, orow_r, fl_r, ce_r,
                           dm_r, dv_r, tlo_r, thi_r, ctx.use_duration,
                           ts_r, tl_r, ps_r, pl_r, ctx.K, dummy, False)

        NEG = -1e30
        best_t = -1
        best_v = NEG
        for t in range(q0, q1 + 1):          # 同値なら t が大きい方（原実装の d 最小優先）
            v = F[t - q0] + BRev[q1 - t]
            if v >= best_v:
                best_v = v
                best_t = t

        if best_t < 0 or best_v <= NEG:
            _solve_block(ctx, n0, n1, q0, q1, seg_start, seg_end)
            continue

        # 先に右を積む（左から処理したいので LIFO 順を調整）
        stack.append((m, n1, best_t, q1))
        stack.append((n0, m, q0, best_t))

    return seg_start, seg_end, best_score


# ---------------------------------------------------------------------------
# Viterbi 強制アライメント
# ---------------------------------------------------------------------------

def forced_align(model: HSMMModel, streams: List[np.ndarray],
                 state_seq: List[StateInfo],
                 use_duration: bool = True,
                 initial_segments: List[Tuple[int, int]] = None,
                 window: int = 60,
                 daem_temp: float = 1.0,
                 nodur_phonemes: set = None,
                 hmm_cap: int = None,
                 backtrace_mem_mb: int = 1024) -> List[Tuple[int, int]]:
    """
    HSMM Viterbi 強制アライメント。

    Parameters
    ----------
    model            : HSMMModel
    streams          : list of (T, 12)  MFCC / delta / delta-delta
    state_seq        : StateInfo のリスト（音素 × 3状態）
    use_duration     : False のとき継続時間分布を無視（HMM モード）
    initial_segments : 第2パス用の初期境界 [(start, end), ...]
    window           : initial_segments からの探索範囲（フレーム数）
    daem_temp        : DAEM 温度 (0, 1]。観測対数尤度をこの値でスケール。
                       1.0 のとき通常の Viterbi（デフォルト）。
    nodur_phonemes   : 継続時間モデルを無効化する音素集合（例: {"pau", "br"}）。
                       指定した音素の dur_var を 0 とみなし一様分布扱いにする。
    hmm_cap          : HMM モード（pass 1）での状態あたり最大フレーム数。
                       None（デフォルト）のとき制限なし。訓練の初期段階で
                       収束を助けるために小さい値（例: 200）を指定できる。
    backtrace_mem_mb : バックポインタ行列に許すメモリ上限（MiB）。
                       これを超える規模では Hirschberg 型の分割統治に切り替え、
                       使用メモリを O(状態参照段数 × フレーム数) に抑える。
                       結果はどちらの経路でも同一。

    Returns
    -------
    segments : list of (start_frame, end_frame)  各状態の区間（end は exclusive）
    """
    T = streams[0].shape[0]
    N = len(state_seq)

    # --- ユニークな out_idx についてのみフレーム尤度を計算 ---
    unique_out  = sorted({s.out_idx for s in state_seq})
    out_to_row  = {idx: r for r, idx in enumerate(unique_out)}
    n_unique    = len(unique_out)

    # cum_ll_mat[row, :] = 累積対数尤度（行 = unique_out の順）
    # DAEM: 観測対数尤度を daem_temp でスケールして分布を平滑化
    cum_ll_mat = np.zeros((n_unique, T + 1), dtype=np.float64)
    for idx in unique_out:
        ll = compute_frame_loglik(model, idx, streams)
        if daem_temp != 1.0:
            ll = ll * daem_temp
        cum_ll_mat[out_to_row[idx], 1:] = np.cumsum(ll)

    # --- 状態ごとの配列を組み立て ---
    floors    = np.array([s.floor for s in state_seq], dtype=np.int32)
    ceils     = np.array([min(s.ceil, T) for s in state_seq], dtype=np.int32)
    # HMM モード（第1パス）は継続時間を探索しないため上限を制限。
    # これにより計算量が O(T²×N) から O(T×N×cap) に削減される。
    # cap は少なくとも T/N（全状態が T フレームをカバーできる最小値）を確保する。
    # nodur_phonemes の状態は継続時間が不定のためキャップしない。
    if not use_duration and hmm_cap is not None:
        effective_cap = max(hmm_cap, math.ceil(T / N))
        for n in range(N):
            if nodur_phonemes and state_seq[n].phoneme in nodur_phonemes:
                pass  # cap しない
            else:
                ceils[n] = min(ceils[n], effective_cap)
    out_row   = np.array([out_to_row[s.out_idx] for s in state_seq], dtype=np.int32)
    dur_means = np.array([model.durations[s.dur_idx].mean for s in state_seq], dtype=np.float64)
    dur_vars  = np.array([model.durations[s.dur_idx].var  for s in state_seq], dtype=np.float64)
    # nodur_phonemes に含まれる音素の dur_var を 0 にして一様分布扱いにする
    if nodur_phonemes:
        for n, s in enumerate(state_seq):
            if s.phoneme in nodur_phonemes:
                dur_vars[n] = 0.0

    # topo / pskip の 2D 配列を組み立て（-1 で番兵）
    max_topo  = max((len(s.topo_sources)  for s in state_seq), default=0)
    max_pskip = max((len(s.pskip_sources) for s in state_seq), default=0)
    max_topo  = max(max_topo,  1)   # Numba に 0 列配列を渡さないようにする
    max_pskip = max(max_pskip, 1)

    topo_src  = np.full((N, max_topo),  -1, dtype=np.int32)
    topo_log  = np.zeros((N, max_topo),     dtype=np.float64)
    pskip_src = np.full((N, max_pskip), -1, dtype=np.int32)
    pskip_log = np.zeros((N, max_pskip),    dtype=np.float64)

    for n, s in enumerate(state_seq):
        for ki, (src, lp) in enumerate(s.topo_sources):
            topo_src[n, ki]  = src
            topo_log[n, ki]  = lp
        for ki, (src, lp) in enumerate(s.pskip_sources):
            pskip_src[n, ki] = src
            pskip_log[n, ki] = lp

    # --- t の探索範囲 ---
    t_lo_arr = np.empty(N, dtype=np.int32)
    t_hi_arr = np.empty(N, dtype=np.int32)
    for n, sn in enumerate(state_seq):
        if initial_segments is None:
            t_lo_arr[n] = sn.floor
            t_hi_arr[n] = T
        else:
            # nodur_phonemes の状態は pass1 の境界が大幅にずれうるため
            # ウィンドウを使わず全範囲を探索する
            if nodur_phonemes and sn.phoneme in nodur_phonemes:
                t_lo_arr[n] = sn.floor
                t_hi_arr[n] = T
            else:
                e0 = initial_segments[n][1]
                lo = max(sn.floor, e0 - window)
                hi = e0 + window if n < N - 1 else T
                t_lo_arr[n] = lo
                t_hi_arr[n] = min(T, hi)

    # --- DP メタデータ ---
    # バックポインタの下位ビットに「何段前の状態から来たか」を詰めるため、
    # スキップ辺の最大段数を求める。
    max_delta = 1
    for n, s in enumerate(state_seq):
        for src, _ in s.topo_sources:
            if n - src > max_delta:
                max_delta = n - src
        for src, _ in s.pskip_sources:
            if n - src > max_delta:
                max_delta = n - src
    if max_delta > BP_MASK:
        raise ValueError(f"skip edge spans {max_delta} states (limit {BP_MASK})")
    K = max_delta + 1

    # スキップ辺が跨ぐ境界は分割統治の分割点に使えない
    blocked = np.zeros(N, dtype=bool)
    for n, s in enumerate(state_seq):
        for src, _ in s.topo_sources:
            blocked[src + 1:n + 1] = True
        for src, _ in s.pskip_sources:
            blocked[src + 1:n + 1] = True
    valid_split = np.nonzero(~blocked)[0].astype(np.int64)
    valid_split = valid_split[(valid_split >= 1) & (valid_split <= N - 1)]

    ctx = _DPCtx(cum_ll=cum_ll_mat, out_row=out_row,
                 floors=floors, ceils=ceils,
                 dur_means=dur_means, dur_vars=dur_vars,
                 t_lo=t_lo_arr, t_hi=t_hi_arr,
                 use_duration=use_duration,
                 topo_src=topo_src, topo_log=topo_log,
                 pskip_src=pskip_src, pskip_log=pskip_log,
                 K=K, max_topo=max_topo, max_pskip=max_pskip,
                 valid_split=valid_split)

    # --- DP 実行 ---
    # バックポインタ行列（int32 1 枚）が閾値に収まるなら通常 DP、
    # 超えるなら Hirschberg 型の分割統治でメモリ O(K·T) にする。
    est_bytes = 4 * N * (T + 1)
    if est_bytes <= backtrace_mem_mb * 1024 * 1024:
        bp = np.zeros((N, T + 1), dtype=np.int32)
        row = _dp_forward_call(ctx, 0, N, 0, T, bp, True)
        best_score = float(row[T])
        if best_score <= -1e30:
            raise RuntimeError("alignment failed: no valid path found. "
                               "Check that the phoneme sequence length fits the audio.")
        ss = np.zeros(N, dtype=np.int64)
        se = np.zeros(N, dtype=np.int64)
        _decode_bp(bp, N, T, ss, se)
    else:
        ss, se, best_score = _hirschberg_backtrace(ctx, N, T)
        if best_score is None or best_score <= -1e30:
            raise RuntimeError("alignment failed: no valid path found. "
                               "Check that the phoneme sequence length fits the audio.")

    segments: List[Tuple[int, int]] = [(int(ss[i]), int(se[i])) for i in range(N)]
    return segments, best_score


def forced_align_2pass(model: HSMMModel, streams: List[np.ndarray],
                       state_seq: List[StateInfo],
                       window: int = 60,
                       daem_temp: float = 1.0,
                       nodur_phonemes: set = None,
                       hmm_cap: int = None,
                       backtrace_mem_mb: int = 1024) -> List[Tuple[int, int]]:
    """
    2パス強制アライメント（SHIRO 方式）。

    1パス目: HMM モード（継続時間分布なし）で粗いアライメント
    2パス目: HSMM モードで 1パス目の境界付近を精細化

    Parameters
    ----------
    window          : 2パス目で 1パス目境界から探索するフレーム幅（デフォルト 60 = 0.3s）
    daem_temp       : DAEM 温度（forced_align に渡す）
    nodur_phonemes  : 継続時間モデルを無効化する音素集合（forced_align に渡す）
    hmm_cap         : HMM pass 1 の状態あたり最大フレーム数（forced_align に渡す）
    """
    initial, _ = forced_align(model, streams, state_seq, use_duration=False,
                              daem_temp=daem_temp, nodur_phonemes=nodur_phonemes,
                              hmm_cap=hmm_cap, backtrace_mem_mb=backtrace_mem_mb)
    refined, loglik = forced_align(model, streams, state_seq, use_duration=True,
                                   initial_segments=initial, window=window,
                                   daem_temp=daem_temp, nodur_phonemes=nodur_phonemes,
                                   hmm_cap=hmm_cap, backtrace_mem_mb=backtrace_mem_mb)
    return refined, loglik


# ---------------------------------------------------------------------------
# pau/子音境界エネルギーオンセット補正
# ---------------------------------------------------------------------------

SOFT_ONSET_PHONES: frozenset = frozenset({
    # 鼻音・接近音（有声、漸進的オンセット）
    'n', 'ny', 'm', 'my', 'y', 'w', 'r', 'ry',
    # 無声摩擦音（エネルギー立ち上がりが緩やか）
    's', 'sh', 'h', 'hy', 'f',
})


def refine_pau_boundaries(
    intervals: List[Tuple[int, int, str]],
    audio: np.ndarray,
    sr: int,
    soft_onset_phones: frozenset = SOFT_ONSET_PHONES,
    search_sec: float = 0.05,
    rms_hop_sec: float = 0.0025,
    threshold_ratio: float = 0.15,
    min_pau_frames: int = 2,
) -> List[Tuple[int, int, str]]:
    """
    pau の直後に来るソフトオンセット子音の pau/子音境界を
    エネルギーオンセット検出で修正する。

    HSMM アライメントでは pau に隣接する鼻音・摩擦音・接近音の
    境界が無音区間側にずれやすい。本関数は境界付近の短時間 RMS を
    走査し、子音区間ピーク RMS の threshold_ratio 倍を超える
    最初の点を新しい境界として採用する。

    Parameters
    ----------
    intervals         : segments_to_phoneme_intervals() の出力
                        [(start_frame, end_frame, phoneme), ...]  ← フレーム単位
    audio             : モノラル float32/float64 音声（raw samples）
    sr                : サンプルレート
    soft_onset_phones : 補正対象音素セット
    search_sec        : 境界前方に探索する時間（秒）
    rms_hop_sec       : RMS 計算のホップ長（秒）
    threshold_ratio   : オンセット閾値 = 子音ピーク RMS × この値
    min_pau_frames    : pau の最小長（フレーム数）。これを下回る短縮は行わない

    Returns
    -------
    補正済み intervals（フレーム単位、同一フォーマット）
    """
    result  = [list(iv) for iv in intervals]
    rms_hop = max(1, round(rms_hop_sec * sr))
    n_audio = len(audio)

    def _rms_frames(seg: np.ndarray) -> np.ndarray:
        n = len(seg) // rms_hop
        if n == 0:
            return np.array([])
        return np.sqrt(np.mean(
            seg[:n * rms_hop].reshape(n, rms_hop).astype(np.float64) ** 2,
            axis=1,
        ))

    for i in range(1, len(result)):
        ph      = result[i][2]
        prev_ph = result[i - 1][2]

        if prev_ph != 'pau' or ph not in soft_onset_phones:
            continue

        f_bound = result[i][0]
        f_end   = result[i][1]

        s_bound = int(f_bound / FPS * sr)
        s_hi    = min(n_audio, s_bound + round(search_sec * sr))
        consnt_seg = audio[s_bound:s_hi]
        if len(consnt_seg) < rms_hop * 2:
            continue

        rms_c    = _rms_frames(consnt_seg)
        if rms_c.size == 0:
            continue
        peak_rms = float(rms_c.max())
        if peak_rms < 1e-7:
            continue

        threshold = peak_rms * threshold_ratio
        onset_hop = None
        for fi, r in enumerate(rms_c):
            if r >= threshold:
                onset_hop = fi
                break

        if onset_hop is None or onset_hop == 0:
            continue

        s_onset = s_bound + onset_hop * rms_hop
        f_new   = round(s_onset / sr * FPS)
        f_new   = min(f_end - min_pau_frames, f_new)
        f_new   = max(f_bound, f_new)

        if f_new <= f_bound:
            continue

        result[i - 1][1] = f_new
        result[i][0]     = f_new

    return [tuple(iv) for iv in result]


# ---------------------------------------------------------------------------
# 外部ラベルの再アライメント
# ---------------------------------------------------------------------------

_VOWELS = frozenset({'a', 'i', 'u', 'e', 'o', 'N', 'A', 'I', 'U', 'E', 'O'})

SILENCE_PHONES: frozenset = frozenset({'pau', 'sil', 'cl', 'br'})


def _phoneme_class(ph: str, silence_phones: frozenset) -> str:
    """音素を 'silence' / 'vowel' / 'consonant' の3クラスに分類する。"""
    if ph in silence_phones:
        return 'silence'
    if ph in _VOWELS:
        return 'vowel'
    return 'consonant'


def realign_external_labels(
    model: HSMMModel,
    streams: List[np.ndarray],
    intervals: List[Tuple[float, float, str]],
    phonemap: dict,
    anchor_vowel_min_sec: float = 0.5,
    anchor_pau_min_sec: float = 0.5,
    anchor_pau_offset_sec: float = 0.25,
    max_seg_sec: float = 30.0,
    phoneme_map: Optional[Dict[str, str]] = None,
    nodur_phonemes: Optional[set] = None,
    hmm_cap: Optional[int] = None,
    ignore_transitions: Optional[FrozenSet[Tuple[str, str]]] = None,
    silence_phones: frozenset = SILENCE_PHONES,
    unknown_phoneme_map: Optional[Dict[str, str]] = None,
) -> List[Tuple[int, int, str]]:
    """
    外部ラベル（他の基準でアノテーションされた .lab / TextGrid）を
    このモデルの作風に変換する。

    外部ラベル中の「長い母音」や「長い pau」の時刻が信頼できるアンカーとして
    使われる。各アンカー間の区間を forced_align_2pass で再アライメントし、
    子音の閉鎖期・摩擦区間の扱い等をモデルの学習スタイルに揃える。

    Parameters
    ----------
    model              : HSMMModel
    streams            : MFCC 特徴量リスト (T, D)
    intervals          : 外部ラベルの音素区間リスト [(start_sec, end_sec, phoneme), ...]
                         read_lab() / read_textgrid() の出力（秒単位）
    phonemap           : phonemap.json の phone_map dict
    anchor_vowel_min_sec : この長さ（秒）以上の母音をアンカーに使う（デフォルト 1.0s）
    anchor_pau_min_sec   : この長さ（秒）以上の pau をアンカーに使う（デフォルト 1.0s）
    anchor_pau_offset_sec: pau 内のアンカー位置（pau 先頭からの秒数、デフォルト 0.5s）
    max_seg_sec          : アンカー間がこの秒数を超えたらスキップ（デフォルト 30.0s）
    phoneme_map          : 外部ラベルの音素名 → pyshiro 音素名の変換テーブル
                           例: {"q": "cl", "nn": "N"} — None のときは変換なし
    nodur_phonemes       : 継続時間モデルを無効化する音素集合（例: {"pau", "br", "cl"}）
    hmm_cap              : HMM 1パス目の状態あたり最大フレーム数
    ignore_transitions   : この移行タイプの音素境界は元の外部ラベルのまま保持する。
                           frozenset of (from_class, to_class) タプル。
                           各クラスは 'silence' / 'vowel' / 'consonant'。
                           例: frozenset({('consonant','vowel'), ('vowel','vowel')})
                           None（デフォルト）のとき全境界を更新（現行と同等）。
                           実装: 指定された境界を追加アンカーとして扱うことで、
                           そのセグメント境界が固定され境界の逆転も起きない。
    silence_phones       : 'silence' クラスとして扱う音素セット。
                           デフォルト: SILENCE_PHONES = {'pau','sil','cl','br'}
    unknown_phoneme_map  : phonemap に存在しない未知音素の代替マッピング。
                           アライメント時だけ指定音素に置き換え、出力ラベルは元の音素名を保持。
                           例: {"vy": "v", "fy": "f", "a'": "a"}
                           マップに登録されていない未知音素はデフォルトでアンカー扱い
                           （両端境界を固定し、隣接セグメントへの影響を防ぐ）。

    Returns
    -------
    List[Tuple[int, int, str]]
        フレーム単位の音素区間リスト（write_lab / write_textgrid に渡せる形式）

    Notes
    -----
    アンカー範囲外（先頭〜最初のアンカー、最後のアンカー〜末尾）も
    max_seg_sec 以内なら同様に処理する。超える区間は元の外部ラベルを維持する。
    """
    from pyshiro.labels import segments_to_phoneme_intervals

    T = streams[0].shape[0]

    # ── 音素名変換 ────────────────────────────────────────────────────────────
    def _map_ph(ph: str) -> str:
        if phoneme_map:
            return phoneme_map.get(ph, ph)
        return ph

    # ── 秒 → フレーム変換（外部ラベル） ──────────────────────────────────────
    intervals_f: List[Tuple[int, int, str]] = [
        (round(s * FPS), round(e * FPS), ph)
        for s, e, ph in intervals
    ]

    # ── アンカー時刻を検出 ────────────────────────────────────────────────────
    # 各アンカーはセグメント境界となる「時刻（フレーム）」。
    # 長い母音       : 中心時刻を1点
    # 長い無音(pau等): 両端から offset の2点（中間の純粋無音域を1音素区間にする）
    #                  無音クラス（silence_phones: pau / sil / cl / br）を対象とする
    anchor_frames: List[int] = []
    offset_f = round(anchor_pau_offset_sec * FPS)
    for s_f, e_f, ph in intervals_f:
        dur_sec = (e_f - s_f) / FPS
        mapped  = _map_ph(ph)
        if mapped in _VOWELS and dur_sec >= anchor_vowel_min_sec:
            anchor_frames.append((s_f + e_f) // 2)
        elif mapped in silence_phones and dur_sec >= anchor_pau_min_sec:
            a0 = s_f + offset_f
            a1 = e_f - offset_f
            anchor_frames.append(a0)
            if a1 > a0:
                anchor_frames.append(a1)

    # phonemap に存在しない未知音素の両端をアンカーに追加
    # （unknown_phoneme_map に登録されていないものに限る）
    # → 未知音素が孤立したセグメントになり、隣接の既知音素セグメントへの影響を防ぐ
    _unknown_set: set = set()
    for s_f, e_f, ph in intervals_f:
        mapped = _map_ph(ph)
        if mapped not in phonemap:
            if unknown_phoneme_map is None or mapped not in unknown_phoneme_map:
                anchor_frames.append(s_f)
                anchor_frames.append(e_f)
                _unknown_set.add(mapped)
    if _unknown_set:
        warnings.warn(
            f"realign_external_labels: unknown phonemes {sorted(_unknown_set)}"
            f" — treating as anchors (boundaries fixed).",
            stacklevel=2,
        )

    # 無視する移行タイプの境界もアンカーとして追加
    # → その境界が固定のセグメント分割点になり、アライメント結果で変更されない
    if ignore_transitions:
        for k in range(len(intervals_f) - 1):
            ph_k  = _map_ph(intervals_f[k][2])
            ph_k1 = _map_ph(intervals_f[k + 1][2])
            # 未知音素（phonemap 未登録）を含む境界は子音扱いにしない
            # → ignore_transitions のマッチ対象から除外し、別途アンカー処理に委ねる
            if ph_k not in phonemap or ph_k1 not in phonemap:
                continue
            cls_k  = _phoneme_class(ph_k,  silence_phones)
            cls_k1 = _phoneme_class(ph_k1, silence_phones)
            if (cls_k, cls_k1) in ignore_transitions:
                anchor_frames.append(intervals_f[k][1])  # 元ラベルの境界フレーム

    anchor_frames = sorted(set(anchor_frames))
    seg_bounds = [0] + anchor_frames + [T]

    # ── 結果配列を外部ラベルで初期化 ─────────────────────────────────────────
    result: List[Tuple[int, int, str]] = list(intervals_f)

    # ── アライメント用音素名変換（未知音素の代替） ────────────────────────────
    def _align_ph(ph: str) -> str:
        mapped = _map_ph(ph)
        if mapped not in phonemap and unknown_phoneme_map:
            return unknown_phoneme_map.get(mapped, mapped)
        return mapped

    # ── 1セグメントのアライメントを試みるヘルパー ─────────────────────────────
    def _try_seg(t_start: int, t_end: int) -> bool:
        """
        [t_start, t_end) を再アライメントして result を更新する。
        成功すれば True、失敗すれば False を返す（result は変更しない）。
        """
        if t_end <= t_start:
            return True

        overlap = [(i, s_f, e_f, ph)
                   for i, (s_f, e_f, ph) in enumerate(intervals_f)
                   if s_f < t_end and e_f > t_start]
        if not overlap:
            return True

        sub_phonemes  = [_align_ph(ph)  for _, _, _, ph in overlap]
        orig_phonemes = [_map_ph(ph)    for _, _, _, ph in overlap]
        sub_streams_  = [s[t_start:t_end] for s in streams]
        T_sub         = t_end - t_start

        try:
            state_seq = build_state_sequence(sub_phonemes, phonemap, T_sub)
            segs, _ = forced_align_2pass(
                model, sub_streams_, state_seq,
                nodur_phonemes=nodur_phonemes,
                hmm_cap=hmm_cap,
            )
        except Exception:
            return False

        new_ivs = segments_to_phoneme_intervals(sub_phonemes, segs)

        for k, (ph_idx, orig_s, orig_e, _) in enumerate(overlap):
            ns_rel, ne_rel, _ = new_ivs[k]
            ns  = ns_rel + t_start
            ne  = ne_rel + t_start
            nph = orig_phonemes[k]

            sl = orig_s < t_start
            sr = orig_e > t_end
            if sl and sr:
                pass
            elif sl:
                result[ph_idx] = (result[ph_idx][0], ne, nph)
            elif sr:
                result[ph_idx] = (ns, result[ph_idx][1], nph)
            else:
                result[ph_idx] = (ns, ne, nph)

        return True

    # ── セグメントごとに再アライメント ────────────────────────────────────────
    for ts, te in zip(seg_bounds[:-1], seg_bounds[1:]):
        # このセグメントに重なる音素のうち、境界がセグメント内側に入るもの
        # （= 実際に再アライメントで動く可能性のある音素）を数える。
        # 全音素が両端を跨ぐだけ（典型例: 長い pau の両端アンカー間）なら
        # 動かす境界が無いので警告せずスキップする。
        actionable = any(
            not (s_f < ts and e_f > te)            # 両端跨ぎでない
            for s_f, e_f, _ in intervals_f
            if s_f < te and e_f > ts               # セグメントに重なる
        )
        if not actionable:
            continue

        seg_sec = (te - ts) / FPS
        if seg_sec > max_seg_sec:
            warnings.warn(
                f"realign_external_labels: segment [{ts/FPS:.2f}s, {te/FPS:.2f}s]"
                f" is {seg_sec:.1f}s > max_seg_sec={max_seg_sec}s"
                f" — keeping original labels for this range.",
                stacklevel=2,
            )
            continue
        if not _try_seg(ts, te):
            warnings.warn(
                f"realign_external_labels: alignment failed for segment"
                f" [{ts/FPS:.2f}s, {te/FPS:.2f}s] — keeping original labels.",
                stacklevel=2,
            )

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    from pyshiro.features import extract_mfcc_from_file
    from pyshiro.model    import load_hsmm
    from pyshiro.phonemes import convert_lyric_file
    from pyshiro.labels   import segments_to_phoneme_intervals, write_lab

    parser = argparse.ArgumentParser(
        description="HSMM forced alignment for Japanese singing voice (pyshiro-fast)")
    parser.add_argument("wav",        type=Path,
                        help="input WAV file (16 kHz mono; no resampling is performed)")
    parser.add_argument("lyrics",     type=Path,
                        help="lyrics .txt file (hiragana, one phrase per line)")
    parser.add_argument("--model",    type=Path, default=None,
                        help="path to a .hsmm model (default: bundled jp-v2)")
    parser.add_argument("--phonemap", type=Path, default=None,
                        help="path to a phonemap .json (default: bundled jp-v2)")
    parser.add_argument("--out",      type=Path, default=None,
                        help="output path (defaults to the WAV name)")
    parser.add_argument("--format",   choices=["lab", "textgrid", "audacity"],
                        default="lab", help="output format (default: lab)")
    args = parser.parse_args()

    if args.model is None or args.phonemap is None:
        from pyshiro.bundled import bundled_model_path
        default_model, default_phonemap = bundled_model_path("jp-v2")
        args.model    = args.model    or default_model
        args.phonemap = args.phonemap or default_phonemap

    out_path = args.out or args.wav.with_suffix(
        ".lab" if args.format == "lab" else
        ".TextGrid" if args.format == "textgrid" else ".txt"
    )

    print(f"Loading model: {args.model}", flush=True)
    model    = load_hsmm(args.model)
    phonemap = load_phonemap(args.phonemap)

    print(f"Extracting features: {args.wav}", flush=True)
    streams  = extract_mfcc_from_file(args.wav)
    T        = streams[0].shape[0]

    print(f"Converting lyrics: {args.lyrics}", flush=True)
    phonemes = convert_lyric_file(args.lyrics)
    print(f"  phonemes ({len(phonemes)}): {' '.join(phonemes)}", flush=True)

    print(f"Aligning... (T={T} frames, {len(phonemes)} phonemes)", flush=True)
    state_seq = build_state_sequence(phonemes, phonemap, T)
    segments, loglik = forced_align_2pass(model, streams, state_seq,
                                          nodur_phonemes={"pau", "br"})
    print(f"  log-likelihood: {loglik / T:.4f} nats/frame", flush=True)

    intervals = segments_to_phoneme_intervals(phonemes, segments)

    if args.format == "lab":
        write_lab(intervals, out_path)
    elif args.format == "textgrid":
        from pyshiro.labels import write_textgrid
        write_textgrid(intervals, out_path)
    else:
        from pyshiro.labels import write_audacity
        write_audacity(intervals, out_path)

    print(f"Wrote: {out_path}", flush=True)


if __name__ == '__main__':
    main()
