"""
Basic behaviour tests.

The model path can be overridden with PYSHIRO_MODEL / PYSHIRO_PHONEMAP;
by default the bundled jp-v2 model is used.
"""

import os
import pytest
import numpy as np
from pathlib import Path

import pyshiro


_env_model    = os.environ.get("PYSHIRO_MODEL", "")
_env_phonemap = os.environ.get("PYSHIRO_PHONEMAP", "")

if _env_model and _env_phonemap:
    HSMM_PATH, PHONEMAP_PATH = Path(_env_model), Path(_env_phonemap)
else:
    from pyshiro.bundled import bundled_model_path
    HSMM_PATH, PHONEMAP_PATH = bundled_model_path("jp-v2")


def test_mfcc_shape():
    audio = np.zeros(16000, dtype=np.float32)
    streams = pyshiro.extract_mfcc(audio)
    assert len(streams) == 3
    for s in streams:
        assert s.shape[1] == 12


def test_phonemes_conversion():
    table = pyshiro.load_table()
    phonemes = pyshiro.text_to_phonemes("きっと\nとべば", table)
    assert phonemes[0] == "pau"
    assert phonemes[-1] == "pau"
    assert "k" in phonemes
    assert "pau" in phonemes


@pytest.mark.skipif(not HSMM_PATH.is_file() or not PHONEMAP_PATH.is_file(),
                    reason="model files not available")
def test_forced_align():
    model    = pyshiro.load_hsmm(HSMM_PATH)
    phonemap = pyshiro.load_phonemap(PHONEMAP_PATH)
    table    = pyshiro.load_table()

    audio    = np.zeros(16000, dtype=np.float32)
    streams  = pyshiro.extract_mfcc(audio)
    T        = streams[0].shape[0]

    phonemes  = pyshiro.text_to_phonemes("きっと", table)
    state_seq = pyshiro.build_state_sequence(phonemes, phonemap, T)
    segments, _ = pyshiro.forced_align(model, streams, state_seq,
                                       use_duration=False)
    assert len(segments) == len(state_seq)
