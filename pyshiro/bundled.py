"""Bundled pre-trained model access.

The wheel ships two Japanese singing-voice models so that a fresh
``pip install pyshiro-fast`` can align audio without any extra download:

==========  =========  =====================================================
name        file       notes
==========  =========  =====================================================
``jp-v2``   .hsmm +    trained with ``--nmix 4``; recommended default
            phonemap
``jp-v1``   .hsmm +    19.4 h multi-database model; supports ``br`` (breath)
            phonemap
==========  =========  =====================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

from pyshiro.align import load_phonemap
from pyshiro.model import HSMMModel, load_hsmm

__all__ = ["BUNDLED_MODELS", "bundled_model_path", "load_bundled_model"]

_DATA_DIR = Path(__file__).resolve().parent / "data" / "models"

BUNDLED_MODELS = ("jp-v2", "jp-v1")


def bundled_model_path(name: str = "jp-v2") -> Tuple[Path, Path]:
    """Return ``(hsmm_path, phonemap_path)`` for a bundled model.

    Parameters
    ----------
    name:
        One of :data:`BUNDLED_MODELS` (``"jp-v2"`` or ``"jp-v1"``).
        ``"v2"`` / ``"v1"`` are accepted as aliases.

    Raises
    ------
    ValueError
        If the name is unknown.
    FileNotFoundError
        If the package data is missing (e.g. a source checkout without
        the model files).
    """
    key = name.strip().lower()
    if key in ("v2", "jp2"):
        key = "jp-v2"
    elif key in ("v1", "jp1"):
        key = "jp-v1"
    if key not in BUNDLED_MODELS:
        raise ValueError(
            f"unknown model {name!r}; available: {', '.join(BUNDLED_MODELS)}"
        )

    hsmm = _DATA_DIR / f"pyshiro-{key}.hsmm"
    phonemap = _DATA_DIR / f"pyshiro-{key}_phonemap.json"
    for path in (hsmm, phonemap):
        if not path.is_file():
            raise FileNotFoundError(
                f"bundled model file not found: {path}\n"
                "If you are running from a source checkout, run "
                "`git lfs pull` or download the model from the releases page."
            )
    return hsmm, phonemap


def load_bundled_model(name: str = "jp-v2") -> Tuple[HSMMModel, dict]:
    """Load a bundled model and its phonemap.

    Returns
    -------
    (model, phonemap)
        ``model`` is a :class:`pyshiro.model.HSMMModel`, ``phonemap`` is the
        ``phone_map`` dict expected by :func:`pyshiro.build_state_sequence`.

    Examples
    --------
    >>> import pyshiro
    >>> model, phonemap = pyshiro.load_bundled_model()      # jp-v2
    """
    hsmm, phonemap_path = bundled_model_path(name)
    return load_hsmm(hsmm), load_phonemap(phonemap_path)
