#!/usr/bin/env python3
"""Build the static documentation site in docs/.

Every docs/*.md is rendered to docs/*.html with one shared template and
docs/style.css. Deliberately dependency-light: Python-Markdown plus a
hand-written stylesheet, no JS, no theme.

Usage:
    python scripts/build_docs.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import markdown
except ImportError:  # pragma: no cover
    sys.exit("python-markdown is required: pip install markdown")

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/henley04/pyshiro-fast"

PAGES = [
    ("index", "Home"),
    ("installation", "Install"),
    ("quickstart", "Quick start"),
    ("cli", "CLI"),
    ("api", "API"),
    ("performance", "Performance"),
    ("migration", "Migration"),
    ("training", "Training"),
    ("models", "Models"),
    ("development", "Development"),
    ("faq", "FAQ"),
]

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="Fast HSMM forced alignment for Japanese singing voice. A drop-in pyshiro fork, 100-700x faster on long audio, identical output.">
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="site-header">
  <div class="wrap">
    <a class="site-name" href="index.html">pyshiro-fast</a>
    <nav>
{nav}
    </nav>
  </div>
</header>
<main class="wrap">
{body}
</main>
<footer class="site-footer">
  <div class="wrap">
    <p>pyshiro-fast · GPL-3.0 · <a href="{repo}">source</a> ·
       <a href="https://pypi.org/project/pyshiro-fast/">PyPI</a></p>
    <p class="muted">Fork of pyshiro by wavtechyukky. Bundled models belong to their
       original authors.</p>
  </div>
</footer>
</body>
</html>
"""


def build_nav(current: str) -> str:
    out = []
    for slug, label in PAGES:
        if slug == "index":
            continue          # the site name links home; no separate nav item
        active = ' aria-current="page"' if slug == current else ""
        out.append(f'      <a href="{slug}.html"{active}>{label}</a>')
    return "\n".join(out)


def rewrite_links(text: str) -> str:
    return re.sub(r'href="([^"#]+)\.md(#[^"]*)?"', r'href="\1.html\2"', text)


def wrap_tables(text: str) -> str:
    """Let wide tables scroll instead of widening the page on small screens."""
    return re.sub(r"<table>.*?</table>",
                  lambda m: f'<div class="table-wrap">{m.group(0)}</div>',
                  text, flags=re.S)


def main() -> None:
    md = markdown.Markdown(
        extensions=["tables", "fenced_code", "sane_lists"],
        output_format="html5",
    )
    built = 0
    for path in sorted(DOCS.glob("*.md")):
        slug = path.stem
        source = path.read_text(encoding="utf-8")

        first = next((ln for ln in source.splitlines() if ln.startswith("# ")), None)
        title = first[2:].strip() if first else slug
        title = f"{title} · pyshiro-fast" if slug != "index" else title

        md.reset()
        body = md.convert(source)
        body = wrap_tables(rewrite_links(body))

        html = TEMPLATE.format(
            title=title, nav=build_nav(slug), body=body, repo=REPO_URL)
        (DOCS / f"{slug}.html").write_text(html, encoding="utf-8")
        built += 1
        print(f"  docs/{slug}.html")

    print(f"built {built} pages")


if __name__ == "__main__":
    main()
