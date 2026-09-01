# Data provenance

## Source

**Server Machine Dataset (SMD)**, published by the NetMan Lab at Tsinghua University alongside
OmniAnomaly (Su et al., KDD 2019).

- Repository: <https://github.com/NetManAIOps/OmniAnomaly>
- Path within the repository: `ServerMachineDataset/`
- Licence: **MIT**, `Copyright (c) 2021 NetManAIOps-SMD` (`ServerMachineDataset/LICENSE`)
- Retrieved: 2026-08-31, via `git clone --filter=blob:none --sparse`
- Size on disk: 466 MB
- Integrity: a SHA-256 digest over every `train/machine-*.txt` file is written to
  `artifacts/provenance.json` by `scripts/run_experiment.py --stage provenance`, so a run can
  be tied to the exact bytes it read.

## Shape

| | |
|---|---|
| Machines | 28 (`machine-1-1` … `machine-3-11`, in three groups of 8 / 9 / 11) |
| Features | 38 per machine |
| Cadence | 1 minute; absolute timestamps are anonymised |
| Span | ~5 weeks per machine |
| `train` rows | 23,688 – 28,700 per machine |
| `test` rows | same order; 4.16% of rows labelled anomalous overall |
| `interpretation_label` | `start-end:dim,dim,...`, 1-indexed dimensions |

## Two properties that shaped the design

**1. The values are already min-max scaled.** The published files contain floats in [0, 1],
normalised per machine by the dataset authors over the whole series. Absolute units are gone,
so no statistic here may depend on raw scale, and every injected magnitude is expressed in
multiples of the pre-onset per-feature standard deviation rather than in raw units.

This global normalisation is itself a mild leak — it was computed using the whole series,
including the part we treat as future — but it affects a monotone per-column rescaling only,
and every transform we fit ourselves is fitted on the reference window alone.

**2. `train` is assumed clean, not verified clean.** SMD ships labels for `test` only. We use
`train` as the drift-free substrate on the dataset's own convention. If it contains unlabelled
anomalies, our drift-free baseline is slightly contaminated, which would inflate the measured
false-alarm rate rather than flatter it. Recorded as a limitation, not silently assumed away.

Several columns are constant for an entire machine. A shift of *k* sigma is undefined when
sigma is zero, so `constant_feature_mask` excludes them from injection explicitly instead of
letting a NaN reach a metric.

## Known criticism of this dataset

Wu & Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the
Illusion of Progress* (arXiv:2009.13807) analyse SMD among other benchmarks and argue that
popular anomaly benchmarks suffer from triviality, unrealistic anomaly density, and
mislabelling.

**This project is not affected by that criticism in its main results, and the reason is
specific.** We do not compete on SMD's anomaly labels, and we make no claim about anomaly
detection performance. SMD is used as a *substrate*: its marginals, cross-feature
correlations, autocorrelation and machine-to-machine heterogeneity are real telemetry rather
than a Gaussian simulator, and those are the properties the injection framework needs. The
labels appear only in the negative control (§ README), where the question is whether a drift
detector is merely re-finding labelled point anomalies — a use that does not depend on the
labels being complete or perfectly placed.

## Datasets considered and rejected

| Dataset | Decision | Reason |
|---|---|---|
| **USP DS Repository** (Insects streams), CC BY 4.0, <https://sites.google.com/view/uspdsrepository> | kept for external validation | real drift with ground truth from a controlled temperature schedule; the strongest available answer to "you injected it yourself" |
| **Backblaze Drive Stats**, free with attribution, no resale | rejected | drift is real but confounded: the fleet composition changes over time, so a detected shift is often a change in *which drives exist* rather than in drive behaviour |
| **ELEC2 / Electricity** | rejected | Žliobaitė, arXiv:1301.3524 — labels are strongly temporally dependent; a "predict the previous label" classifier reaches ~85% where independence would give ~51%, and accuracy and Kappa both fail to expose it |
| **NAB / SMAP / MSL** | rejected | same family of criticism as above but more severe, and they are anomaly benchmarks rather than drift benchmarks |

## Attribution

If you use SMD, cite the OmniAnomaly paper:

> Su, Y., Zhao, Y., Niu, C., Liu, R., Sun, W., Pei, D. *Robust Anomaly Detection for
> Multivariate Time Series through Stochastic Recurrent Neural Network.* KDD 2019.
