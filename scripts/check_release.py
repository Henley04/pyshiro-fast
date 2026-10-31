#!/usr/bin/env python3
"""Pre-release sanity checks.

Run before `python -m build`. Exits non-zero on the first category of problem found.

    python scripts/check_release.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
problems: list[str] = []
notes: list[str] = []


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --- metadata --------------------------------------------------------------

pyproject = read(ROOT / "pyproject.toml")
if 'name = "pyshiro-fast"' not in pyproject:
    problems.append("pyproject.toml: project name is not pyshiro-fast")
if "dynamic" not in pyproject or 'attr = "pyshiro.__version__"' not in pyproject:
    problems.append("pyproject.toml: version is not dynamic from pyshiro.__version__")
if 'license = "GPL-3.0-only"' not in pyproject:
    problems.append("pyproject.toml: license is not GPL-3.0-only")
if "license-files" not in pyproject:
    problems.append("pyproject.toml: license-files missing (LICENSE / NOTICE)")

init = read(ROOT / "pyshiro" / "__init__.py")
m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
if not m:
    problems.append("pyshiro/__init__.py: __version__ not found")
    version = "?"
else:
    version = m.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        problems.append(f"__version__ {version!r} is not MAJOR.MINOR.PATCH")

changelog = read(ROOT / "CHANGELOG.md")
if f"## [{version}]" not in changelog:
    problems.append(f"CHANGELOG.md: no section for version {version}")

# --- files -----------------------------------------------------------------

required = [
    "README.md", "LICENSE", "NOTICE", "CHANGELOG.md", "CONTRIBUTING.md",
    "CITATION.cff", "MANIFEST.in", "pyproject.toml",
    "pyshiro/data/kana2phonemes.table",
    "pyshiro/data/models/pyshiro-jp-v2.hsmm",
    "pyshiro/data/models/pyshiro-jp-v2_phonemap.json",
    "pyshiro/data/models/pyshiro-jp-v1.hsmm",
    "pyshiro/data/models/pyshiro-jp-v1_phonemap.json",
]
for rel in required:
    if not (ROOT / rel).is_file():
        problems.append(f"missing file: {rel}")

# --- docs up to date -------------------------------------------------------

docs = ROOT / "docs"
for md in sorted(docs.glob("*.md")):
    html = md.with_suffix(".html")
    if not html.is_file():
        problems.append(f"docs/{html.name} missing — run scripts/build_docs.py")
    elif html.stat().st_mtime + 1 < md.stat().st_mtime:
        problems.append(f"docs/{html.name} older than {md.name} — rebuild the site")

# --- leftovers -------------------------------------------------------------

caches = sorted((ROOT / "pyshiro").rglob("__pycache__"))
if caches:
    notes.append(f"{len(caches)} __pycache__ director{'y' if len(caches) == 1 else 'ies'} "
                 "present (excluded from the wheel, but remove before committing)")

# --- report ----------------------------------------------------------------

print(f"version: {version}")

if notes:
    print("\nnotes:")
    for n in notes:
        print(f"  - {n}")

if problems:
    print("\nPROBLEMS:")
    for p in problems:
        print(f"  - {p}")
    sys.exit(1)

print("\nall checks passed")
