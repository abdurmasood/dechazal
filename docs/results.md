# Results and method

[← Back to README](../README.md)

## Full results

Numbers are for the DS2 test patients (22 records, 49,685 beats). Each cell is mean ± sample
std across seeds, in %. Se is sensitivity and PPV is positive predictive value.

| Model | Input | Seeds | Macro-F1 | N Se | N PPV | S Se | S PPV | V Se | V PPV | F Se | F PPV |
|---|---|---|---|---|---|---|---|---|---|---|---|
| cnn1d | filtered beat window | 5 | 36.4 ± 3.8 | 80.3 ± 12.7 | 93.6 ± 1.0 | 5.7 ± 6.4 | 1.4 ± 1.1 | 74.9 ± 20.8 | 57.4 ± 28.3 | 0.4 ± 0.8 | 0.1 ± 0.1 |
| cnn1d_rr | window + RR features | 5 | 39.3 ± 1.5 | 71.0 ± 20.6 | 94.4 ± 2.1 | 36.3 ± 35.1 | 5.3 ± 4.1 | 69.8 ± 18.2 | 73.5 ± 16.7 | 2.5 ± 5.2 | 0.2 ± 0.3 |
| alexnet | 224×224 beat image | 3 | 24.4 ± 6.2 | 42.9 ± 35.0 | 92.6 ± 2.8 | 38.0 ± 46.7 | 2.7 ± 1.8 | 70.7 ± 35.5 | 35.2 ± 16.9 | 0.0 ± 0.0 | 0.0 ± 0.0 |

Q excluded: 8 (DS1), 7 (DS2).

`uv run arrhythmia report` prints this table, minus the Input column.

### How to read this

- **The 2019 image representation is the weakest input.** On the same windows, split and
  training settings, AlexNet on plotted beats reaches 24.4 macro-F1, compared with 36.4 for
  cnn1d on the signal. Per-beat autoscaling discards amplitude, and the 28M-parameter model
  swings between seeds: N Se ±35 and S Se ±47 mean some seeds label many normal beats as S.
- **S beats need rhythm information.** From beat shape alone, cnn1d detects almost no S beats
  (Se 5.7%). Adding RR intervals raises the mean to 36%, but the spread across seeds (±35) shows
  the model is unstable on S.
- **F does not generalise across patients.** Almost all F beats come from one patient on each
  side of the split (see [Caveats](#caveats)). In practice F performance measures whether
  patient 208 predicts patient 213.
- **Class weighting trades N accuracy for minority recall.** Balanced weights give F a weight
  of 25.3 and N 0.28, roughly 90:1. The model then over-predicts the rare classes, which is why
  N Se is low and S PPV is tiny.
- These numbers are lower than the beat-wise random-split results common online. Those splits
  put the same patient in train and test, and inflate accuracy. Inter-patient results in the
  literature are also modest, especially for S and F.

Nothing above was tuned on DS2. All settings were fixed before the first test-set evaluation.

## Method

```
MIT-BIH record ──► MLII lead ──► baseline removal ──► 35 Hz low-pass ──► window at R peak
 (wfdb, 360 Hz)    (by name)     (200+600 ms median)   (Butterworth,     [-0.25 s, +0.45 s]
                                                        zero-phase)       252 samples
                                                                              │
              ┌───────────────────────────────────────────────────────────────┤
              ▼                                   ▼                           ▼
       cnn1d (73k params)               cnn1d_rr (+4 RR features)     224×224 line image
                                                                      ──► alexnet (28M)
              └────────────────────────── N / S / V / F ──────────────────────┘
```

| Step | Choice | Why |
|---|---|---|
| Split | DS1 for training (22 records), DS2 for testing (22). Paced 102/104/107/217 excluded. | Standard inter-patient protocol (de Chazal 2004). |
| Validation | DS1 records 114, 124, 201, 205, 215 | Picked by patient. 208 (most F) and 209 (most S) stay in training. |
| Labels | AAMI N/S/V/F. Q counted and dropped. | Comparable with the literature. Q has 15 beats outside the paced records. |
| Lead | MLII, selected by signal name | Channel 0 is not MLII in records 102, 104 and 114. |
| Filtering | Cascaded median baseline removal, then a 4th-order 35 Hz Butterworth (`sosfiltfilt`) | Follows de Chazal. The IIR replaces their FIR with zero phase. |
| Window | 90 samples before and 162 after the annotated R peak | Covers P-wave onset and T-wave end at normal heart rates. |
| 1D input | One global scalar standardisation fitted on training patients | Keeps relative amplitude, which helps identify V beats. |
| 2D input | Same windows drawn as a 224×224 line plot, min-max scaled per beat | Reproduces the 2019 representation, so input representation is the only difference. |
| Imbalance | Balanced class weights, applied to both training and validation loss | No resampling. Weights are fixed, not tuned. |
| Training | Adam 1e-3, at most 60 epochs, early stopping on weighted val loss (patience 8) | Validation has only 23 F beats, so a macro-F1 stopping rule would be noisy. |
| Seeds | 5 seeds for the 1D models, 3 for the 2D model | Each seed covers weight initialisation, shuffling and dropout. |

Each run writes `runs/<model>/seed<k>/` containing `metrics.json`, `confusion.csv`,
`config.json` and `model.keras`. `novelnet` can also be selected but was not trained for the
table.

## What changed since 2019

This repo started in 2019 as a 2D CNN trained on images of plotted heartbeats. In 2026 it was
rebuilt with a sound evaluation, keeping the 2019 image model as a baseline. The 2019 code could
not produce valid results:

1. **Every class got the same label.** `trainAndTestSplit` aliased the global class list and
   called `.remove()` on it while assigning labels with `.index()`. With folders listed as
   L, N, V, A, R, all five classes got label 0. A characterization test reproduces this:
   `tests/legacy/test_cnn_model_label_bug.py`.
2. **Patient leakage.** The split was random by beat, so the same patients appeared in train and
   test. The test set was also used as validation data and to pick the best checkpoint.
3. **Wrong lead for record 114.** Reading channel 0 gives V5 there, not MLII.
4. **The 2019 Novelnet couldn't be built.** Commit `aff2afe` added a 3×3 conv on a 2×2 feature
   map. `alexnet_wide` is that model, renamed for what it is: AlexNet with a 13×13 first kernel,
   wider conv4/conv5 (434 and 500 filters) and heavier dropout. It uses the layer configuration
   from before `aff2afe` and is trained with the same settings as every other model.
5. **No filtering.** Windows ran from one annotation to the next, so their length varied, and
   non-beat annotations (`+`, `~`, `|`) were treated as beats.

## Caveats

- **F is essentially one patient per side.** Record 208 holds 373 of DS1's 415 F beats. Record
  213 holds 362 of DS2's 388.
- **S is concentrated too.** Record 232 holds 1,382 of DS2's 1,837 S beats, so S results lean
  heavily on one test patient.
- **Seed std is not patient variance.** It shows how sensitive training is to its random choices,
  not how results would move with different patients.
- **MLII only.** de Chazal used both leads.
- **Plain Se/PPV.** ANSI/AAMI EC57 exempts F beats classified as V from V's false positives.
  That convention is not applied here.
- **Hardware.** cnn1d and cnn1d_rr were trained on an Apple-silicon CPU, and alexnet on an
  NVIDIA RTX 4090 (JAX with CUDA 12). Seeds fix initialisation, shuffling and dropout, but
  floating-point results can still differ slightly across hardware, so expect small deviations
  from the table on other machines.
- **RR features look one beat ahead.** cnn1d_rr uses the post-RR interval, so it is an offline
  classifier.

## Code layout

```
src/arrhythmia/
├── cli.py              arrhythmia download | prepare | train | report
├── pipeline.py         glue: build splits, train one seed, collect runs
├── config.py           fixed filter / window / image settings
├── data/
│   ├── records.py      WFDB loading (MLII by name), PhysioNet download
│   ├── labels.py       MIT-BIH symbol → AAMI class
│   ├── splits.py       DS1/DS2/validation records, patient-overlap guard
│   ├── preprocessing.py baseline removal, low-pass, R-centred windows
│   ├── rr.py           RR-interval features
│   ├── dataset.py      labelled, patient-tagged beat sets with an npz cache
│   └── images.py       beat → 224×224 image, memmap cache
├── models/
│   ├── blocks.py       shared conv/dense blocks for the 2019 image CNNs
│   ├── image_cnns.py   alexnet, alexnet_wide layer specs
│   └── cnn1d.py        cnn1d, cnn1d_rr
└── training/
    ├── fit.py          class weights, seeded fit, batch feeder
    └── evaluation.py   per-class Se/PPV/FPR/F1, macro-F1, seed aggregation
tests/              unit tests, CLI smoke test, real-data integration tests
scripts/            frozen 2019 code, kept for the label-bug characterization test
docs/               this page and the README chart
```

## References

- P. de Chazal, M. O'Dwyer, R. B. Reilly. *Automatic classification of heartbeats using ECG
  morphology and heartbeat interval features.* IEEE TBME 51(7):1196–1206, 2004.
- ANSI/AAMI EC57: *Testing and reporting performance results of cardiac rhythm and ST segment
  measurement algorithms.*
- G. B. Moody, R. G. Mark. *The impact of the MIT-BIH Arrhythmia Database.* IEEE EMB Magazine
  20(3):45–50, 2001.
- A. L. Goldberger et al. *PhysioBank, PhysioToolkit, and PhysioNet.* Circulation
  101(23):e215–e220, 2000.
