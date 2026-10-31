# Contributing

Thanks for taking the time to contribute. This project is small, so keeping the rules
short is enough.

## Before you start

- Open an issue for anything larger than a bug fix or a documentation correction, so the
  approach can be agreed on first.
- For alignment-quality questions, include the model, the lyrics, and the audio length —
  a minimal reproduction is worth more than a description.

## Development setup

```bash
git clone https://github.com/henley04/pyshiro-fast
cd pyshiro-fast
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## The one hard rule

**Alignment output must not change.** `tests/test_viterbi_exact.py` compares the
optimised kernel against the original `_viterbi_dp` and requires identical segments and
identical log-likelihood. If your change alters either, it is a behaviour change, not an
optimisation, and needs to be discussed in an issue first.

Performance changes should come with before/after numbers from
`bench/bench_align_speed.py`, on at least two audio lengths.

## Pull requests

1. One topic per pull request.
2. Tests for new behaviour; a benchmark for performance work.
3. If you change a `docs/*.md`, rebuild the site and commit the HTML:

   ```bash
   python scripts/build_docs.py
   ```

4. Update `CHANGELOG.md` under an "Unreleased" heading.
5. Describe *why* in the pull request, not only what changed.

## Style

- Python 3.10+, no new runtime dependencies without discussion.
- Match the surrounding code; there is no enforced formatter.
- User-facing strings and new docstrings in English. Inherited Japanese comments and
  docstrings may stay.
- Keep the DP kernel readable: it is the part most likely to be reviewed by someone
  checking correctness.

## Reporting bugs

Include:

- `pyshiro.__version__`, Python version, OS, numba version
- the command or script you ran
- the full traceback
- if the problem is alignment quality: audio duration, number of phonemes, model name,
  and the log-likelihood printed by the CLI

Do not attach copyrighted audio or lyrics. A synthetic reproduction is usually enough.

## License

By contributing you agree that your contribution is licensed under GPL-3.0, the license
of this project.
