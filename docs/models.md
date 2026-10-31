# Bundled models

The wheel contains two Japanese singing-voice models plus their phonemaps, so a fresh
install can align without downloading anything.

| Name | File | Mixtures | Size | Notes |
|---|---|---|---:|---|
| `jp-v2` (default) | `pyshiro-jp-v2.hsmm` | 4 | 487 KB | better overall boundary accuracy |
| `jp-v1` | `pyshiro-jp-v1.hsmm` | 1 | 127 KB | more stable on `br` (breath) |

```python
import pyshiro
model, phonemap = pyshiro.load_bundled_model()          # jp-v2
model, phonemap = pyshiro.load_bundled_model("jp-v1")   # breath-heavy material
hsmm, pm = pyshiro.bundled_model_path("jp-v1")          # just the paths
```

## Accuracy

Measured on manually verified phoneme boundaries from singers that were not in the
training set:

| Model | Mean boundary error |
|---|---:|
| `jp-v1` | 3.27 frames (16 ms) |
| `jp-v2` | 2.05 frames (10 ms) |

### Known weakness of `jp-v2`: breaths

`jp-v2` under-estimates `br` regions. On 1 015 breath segments from one singing database,
the mean difference from the reference labels was 8.4 frames for `jp-v1` and 16.4 frames
for `jp-v2`; segments off by more than 200 ms were 13 for `jp-v1` and 133 for `jp-v2`.
Adjusting the duration floor does not close the gap.

For material with many breaths, either align with `jp-v1` or keep `br` out of the
duration model:

```python
segments, loglik = pyshiro.forced_align_2pass(
    model, streams, state_seq, nodur_phonemes={"pau", "br"})
```

The CLI already does this.

## Phoneme set

Both models use the same 49-entry Japanese phoneme set, including `pau` (pause), `br`
(breath), `cl` (consonant length) and `N` (moraic nasal). The phonemap defines, per
phoneme, its topology (`type-a` / `type-b` / `type-c` / `skip-boundary`), its duration
distribution index, and whether it is skippable (`pskip`).

## Kana-to-phoneme table

`pyshiro/data/kana2phonemes.table` is derived from ENUNU's `kana2phonemes_etk_001.table`.

## Licensing and credits

Both models were trained by wavtechyukky and are redistributed unmodified from the
upstream repository under GPL-3.0.

`pyshiro-jp-v1.hsmm` was trained on the following singing-voice databases. Credit belongs
to their creators and rights holders:

- **Onikuru Kuruumi Singing Voice Database** — Onikuru Kuruumi
  ([onikuru.info](https://onikuru.info))
- **Oftn-P Singing Voice Database** — database by Oftn-P
  ([sites.google.com/view/oftn-utagoedb](https://sites.google.com/view/oftn-utagoedb))
- **Namine Ritsu** — Canon
  ([canon-voice.com](https://www.canon-voice.com))
- **Tohoku Kiritan Singing Database** — © SSS
  ([zunko.jp/kiridev](https://zunko.jp/kiridev/login.php))

If you plan to use these models commercially, review the terms of those databases first.
The GPL covers the model files as distributed here; it cannot grant rights the original
data providers did not grant.

## Training your own model

See [Training](training.md). Any `.hsmm` produced there is compatible with the same
API — pass it explicitly instead of using `load_bundled_model`.
