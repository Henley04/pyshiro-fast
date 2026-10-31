"""Alias package for :mod:`pyshiro`.

The distribution is called ``pyshiro-fast`` but the import name is ``pyshiro``
so that existing scripts keep working unchanged.  This module exists only for
people who prefer to import the distribution name directly::

    import pyshiro_fast as pyshiro

Both names expose the same objects and submodules.
"""

import sys as _sys

import pyshiro as _impl
from pyshiro import *  # noqa: F401,F403
from pyshiro import __version__  # noqa: F401
from pyshiro import (  # noqa: F401
    align,
    bundled,
    features,
    labels,
    model,
    phonemes,
    train,
)

for _name in ("align", "bundled", "features", "labels", "model", "phonemes", "train"):
    _sys.modules[__name__ + "." + _name] = getattr(_impl, _name)

__all__ = _impl.__all__
