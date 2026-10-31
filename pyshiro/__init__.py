"""
pyshiro-fast — fast HSMM forced alignment for Japanese singing voice.

A drop-in fork of `pyshiro <https://github.com/wavtechyukky/pyshiro>`_ with a
rewritten Viterbi decoder: O(N·T) decoding in the duration-free pass,
compact backpointers, exact window pruning and Hirschberg divide-and-conquer
backtracking. Alignment output is bit-identical to upstream; long audio is
100x-700x faster.

Provides:
  - HSMM model loading/saving (.hsmm format)
  - MFCC feature extraction (SHIRO/ciglet compatible)
  - Viterbi forced alignment (2-pass: HMM bootstrap + HSMM refinement)
  - Label output (HTK .lab, Praat TextGrid, Audacity)
  - Kana-to-phoneme conversion (Japanese)
  - HSMM model training from corpus
  - Two bundled pre-trained Japanese models
"""

from pyshiro.model    import load_hsmm, save_hsmm, HSMMModel
from pyshiro.features import extract_mfcc, extract_mfcc_from_file
from pyshiro.align    import (load_phonemap, build_state_sequence,
                               forced_align, forced_align_2pass,
                               refine_pau_boundaries,
                               realign_external_labels,
                               SILENCE_PHONES)
from pyshiro.labels   import (write_lab, write_lab_sec, write_textgrid,
                               write_audacity, read_lab, read_textgrid,
                               read_audacity, segments_to_phoneme_intervals)
from pyshiro.phonemes import load_table, convert_lyric_file, text_to_phonemes
from pyshiro.bundled  import (load_bundled_model, bundled_model_path,
                               BUNDLED_MODELS)

__version__ = "1.0.0"

__all__ = [
    "load_hsmm", "save_hsmm", "HSMMModel",
    "extract_mfcc", "extract_mfcc_from_file",
    "load_phonemap", "build_state_sequence",
    "forced_align", "forced_align_2pass",
    "refine_pau_boundaries", "realign_external_labels",
    "SILENCE_PHONES",
    "write_lab", "write_lab_sec", "write_textgrid", "write_audacity",
    "read_lab", "read_textgrid", "read_audacity",
    "segments_to_phoneme_intervals",
    "load_table", "convert_lyric_file", "text_to_phonemes",
    "load_bundled_model", "bundled_model_path", "BUNDLED_MODELS",
    "__version__",
]
