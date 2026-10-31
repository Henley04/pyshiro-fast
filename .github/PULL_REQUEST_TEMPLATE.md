## What this changes

<!-- One or two sentences. Link the issue if there is one. -->

## Why

<!-- The problem, not the patch. For performance work, include before/after
     numbers from bench/bench_align_speed.py. -->

## Checklist

- [ ] `pytest` passes
- [ ] `tests/test_viterbi_exact.py` still passes (alignment output unchanged)
- [ ] Benchmark before/after, if this touches the decoder
- [ ] `python scripts/build_docs.py` run and `docs/*.html` committed, if docs changed
- [ ] `CHANGELOG.md` updated under "Unreleased"
- [ ] No new runtime dependency

## Notes for the reviewer

<!-- Anything non-obvious: trade-offs, things you tried and rejected. -->
