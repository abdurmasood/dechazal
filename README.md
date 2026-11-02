<div align="center">

# dechazal

Heartbeat classification on MIT-BIH, tested only on patients the model has never seen.

[![CI](https://github.com/abdurmasood/dechazal/actions/workflows/ci.yml/badge.svg)](https://github.com/abdurmasood/dechazal/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Keras 3 on JAX](https://img.shields.io/badge/keras-3%20on%20jax-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-blue)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/macro-f1-dark.svg">
  <img alt="Macro-F1 on 22 unseen patients: cnn1d_rr 39.3, cnn1d 36.4, alexnet_wide 30.4, alexnet 24.4" src="docs/macro-f1.svg" width="640">
</picture>

</div>

1. **Pictures of beats score lower than the signal and are less stable.** On the same beats,
   a 1D CNN scores 36.4 macro-F1. The two 2019 image CNNs score 30.4 (alexnet_wide) and
   24.4 (alexnet), and alexnet_wide's three seeds range from 19.9 to 39.4.
2. **S beats need rhythm.** Adding RR intervals raises S sensitivity from 6% to 36%.
3. **F beats don't transfer.** Nearly all of them come from one training patient and one
   test patient.

## Why it's useful

Most MIT-BIH tutorials split beats at random, so the same patient ends up in both train and
test, and accuracy looks far better than it would on a new patient. This repo uses the
standard inter-patient split from de Chazal et al. (2004) and gives you a leak-free baseline
to compare against, along with the pipeline that produced it.

| You get | Where |
|---|---|
| MIT-BIH download and preprocessing | `arrhythmia download`, `arrhythmia prepare` |
| DS1/DS2 patient split with an overlap guard | `src/arrhythmia/data/splits.py` |
| Four models with seeded training | `arrhythmia train` |
| Per-class sensitivity, PPV and F1 report | `arrhythmia report` |
| 60 tests, about 10 s, no download needed | `uv run pytest` |

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is pinned in `.python-version`.

```bash
uv sync
uv run arrhythmia download                      # ~100 MB from PhysioNet into data/raw/mitdb
uv run arrhythmia prepare                       # filter, window, cache; prints class counts
uv run arrhythmia train --model cnn1d    --seeds 0 1 2 3 4
uv run arrhythmia train --model cnn1d_rr --seeds 0 1 2 3 4
uv run arrhythmia train --model alexnet  --seeds 0 1 2   # ~7 s/epoch on an RTX 4090
uv run arrhythmia train --model alexnet_wide --seeds 0 1 2   # ~8 s/epoch on an RTX 4090
uv run arrhythmia report                        # prints the results table
```

Tests:

```bash
uv run pytest                                   # synthetic WFDB fixtures, no network
MITDB_DIR=data/raw/mitdb uv run pytest -m mitdb # real data: class counts must match de Chazal
```

## Per-class results

Test set: 22 DS2 patients, 49,685 beats. Each cell is sensitivity / PPV, mean over seeds, in %.

| Model | N | S | V | F |
|---|---:|---:|---:|---:|
| cnn1d_rr | 71.0 / 94.4 | 36.3 / 5.3 | 69.8 / 73.5 | 2.5 / 0.2 |
| cnn1d | 80.3 / 93.6 | 5.7 / 1.4 | 74.9 / 57.4 | 0.4 / 0.1 |
| alexnet_wide | 65.8 / 94.8 | 23.8 / 9.7 | 56.2 / 58.5 | 18.0 / 0.7 |
| alexnet | 42.9 / 92.6 | 38.0 / 2.7 | 70.7 / 35.2 | 0.0 / 0.0 |

N normal · S supraventricular ectopic · V ventricular ectopic · F fusion. A high sensitivity
with a tiny PPV means the model over-predicts that class: alexnet_wide's F score comes from one
seed labelling 21,446 normal beats as F. Standard deviations are in
[docs/results.md](docs/results.md#full-results).

> [!WARNING]
> Read before citing. F is one patient per side (records 208 and 213). The ± ranges show
> seed variance, not patient variance. Only the MLII lead is used. RR features look one beat
> ahead, so cnn1d_rr is an offline classifier. Full list: [Caveats](docs/results.md#caveats).

## More

- [Full results and how to read them](docs/results.md#full-results)
- [Method: split, filtering, windows, training](docs/results.md#method)
- [What changed since the 2019 version](docs/results.md#what-changed-since-2019)
- [Code layout](docs/results.md#code-layout)
- [References](docs/results.md#references)

## License

Code: [MIT](LICENSE). The MIT-BIH Arrhythmia Database is distributed by PhysioNet under its own
terms; it is downloaded at run time, not redistributed here.
