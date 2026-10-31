# CLI reference

Three entry points are installed or available via `python -m`:

| Command | Module | Purpose |
|---|---|---|
| `pyshiro-align` | `pyshiro.align` | align one WAV + lyrics pair |
| `pyshiro-train` | `pyshiro.train` | train an `.hsmm` model from a corpus |
| `python -m pyshiro.untie` | `pyshiro.untie` | expand a monophone model to triphones |

---

## `pyshiro-align`

```
pyshiro-align WAV LYRICS [--model PATH] [--phonemap PATH] [--out PATH]
                        [--format {lab,textgrid,audacity}]
```

| Argument | Required | Description |
|---|---|---|
| `WAV` | yes | 16 kHz mono WAV |
| `LYRICS` | yes | UTF-8 lyrics, hiragana, one phrase per line |
| `--model` | no | `.hsmm` file; default: bundled `jp-v2` |
| `--phonemap` | no | `phonemap.json`; default: bundled `jp-v2` |
| `--out` | no | output path; default: the WAV name with the format's extension |
| `--format` | no | `lab` (default), `textgrid`, `audacity` |

Examples:

```bash
# bundled model, song.lab
pyshiro-align song.wav song.txt

# custom model, Praat TextGrid
pyshiro-align song.wav song.txt \
    --model my.hsmm --phonemap my_phonemap.json \
    --format textgrid --out song.TextGrid

# batch: every WAV in a folder that has a matching .txt
for w in wav/*.wav; do
  t="${w%.wav}.txt"
  [ -f "$t" ] && pyshiro-align "$w" "$t"
done
```

The CLI always runs the two-pass pipeline with `nodur_phonemes={"pau","br"}`. For other
settings, call the Python API directly.

**Output formats.** `lab` is HTK 100 ns integers (ENUNU / NNSVS / vLabeler compatible);
`textgrid` is a Praat TextGrid with one `phoneme` tier; `audacity` is tab-separated
seconds, one label per line.

**Exit status.** `0` on success. A `RuntimeError` (`alignment failed: no valid path
found`) means no state path covers the audio — usually lyrics longer than the audio.

---

## `pyshiro-train`

```
pyshiro-train --wav_dir DIR --lab_dir DIR [options]
```

| Option | Default | Description |
|---|---|---|
| `--wav_dir` | required | directory of 16 kHz mono WAVs |
| `--lab_dir` | required | directory of initial `.lab` files (same stems) |
| `--phonemap` | bundled `jp-v2` phonemap | phoneme set definition |
| `--out` | `my_model.hsmm` | output model |
| `--iters` | `5` | total EM iterations |
| `--hmm_iters` | `0` | duration-free bootstrap iterations (like `shiro-rest -g`) |
| `--daem` | off | temperature-annealed training (like `shiro-rest -D`) |
| `--nmix` | `1` | maximum GMM mixtures; splits happen during training |
| `--init_model` | none | start from this model instead of flat start |
| `--start_iter` | `0` | resume at this iteration (with `--init_model`) |
| `--jobs` | CPU count | parallel feature-extraction workers |
| `--cap_relax_iter` | none | from this iteration, relax `hmm_cap` from 200 to 1000 |
| `--test_wav_dir` | none | held-out WAVs for per-iteration test log-likelihood |
| `--test_lab_dir` | none | held-out labels, used with `--test_wav_dir` |

Minimal run:

```bash
pyshiro-train --wav_dir corpus/wav --lab_dir corpus/lab \
    --phonemap my_phonemap.json --out model.hsmm --iters 8
```

Recommended settings for a singing-voice corpus (the configuration used for the bundled
`jp-v2` model):

```bash
pyshiro-train --wav_dir corpus/wav --lab_dir corpus/lab \
    --phonemap my_phonemap.json --out model.hsmm \
    --hmm_iters 4 --daem --nmix 4 --iters 16 --cap_relax_iter 6
```

Checkpoints `model.iter001.hsmm`, `model.iter002.hsmm`, … are written after every
iteration, so training can be resumed:

```bash
pyshiro-train ... --init_model model.iter006.hsmm --start_iter 6 --iters 16
```

With `--nmix 4` the GMM split happens at roughly one third and two thirds of `--iters`,
which is why `--iters 16` is suggested rather than 8. See [Training](training.md).

---

## `python -m pyshiro.untie`

Expands a monophone model into context-dependent triphones, using the triphone contexts
observed in a labelled corpus.

```bash
python -m pyshiro.untie \
    --phonemap mono_phonemap.json --model mono.hsmm \
    --lab_dir corpus/lab \
    --out_phonemap tri_phonemap.json --out_model tri.hsmm \
    --sil pau sil br
```

| Option | Description |
|---|---|
| `--phonemap` | input monophone phonemap |
| `--model` | input monophone `.hsmm` |
| `--lab_dir` | corpus labels, used to collect contexts |
| `--out_phonemap` | output triphone phonemap |
| `--out_model` | output triphone model |
| `--sil` | phonemes treated as context boundaries (default `pau sil br`) |

Triphone models are larger and slower to align with; use them only when monophone
accuracy is insufficient.
