# Validation design

Written before the evaluation code, and not revised after seeing results. Where the design
did change during development, the change and its reason are recorded in § Revisions.

## What is being measured

For each stream the system emits alarms; the evaluation compares those alarms against
injected ground truth `(onset, drift type, affected features, magnitude)`. There is no target
column and no supervised signal anywhere in the pipeline.

## Units

| Level | Definition | Why |
|---|---|---|
| Row | one `(machine, minute)` with 38 values | the raw record |
| Window | 2500 consecutive rows of one machine | drift is a property of a distribution; no single row can be "drifted" |
| **Stream** | one machine × one scenario × one magnitude × one seed | **the unit of evaluation and of bootstrap resampling** |

Everything aggregates over streams. Windows within a stream overlap by 2000 rows and are
strongly dependent, so averaging over them — or bootstrapping over them — would manufacture
precision that does not exist.

## Splits

Two independent splits, doing two different jobs. Conflating them is the mistake this section
exists to prevent.

**Split 1 — machines, for design decisions.**

| Split | Machines | Used for |
|---|---|---|
| calibration | 11 | reserved; not used for reporting |
| development | 11 | design, iteration, error analysis, every number quoted while building |
| held-out | 6 | touched once, after the pipeline is frozen |

Split by machine, never by row: the 38 signals of one machine resemble each other across time
far more than they resemble another machine's, so a row-level split would leak a machine's
behavioural profile. Each split spans all three SMD groups, because the groups differ
systematically and a split confined to one would measure the group rather than the method.

**Split 2 — time within each machine, for thresholds.**

Thresholds are calibrated **per machine**, on the first 35% of that machine's clean history.
Evaluation streams are drawn only from the remaining 65%. No row is ever in both.

Per machine, not global, because the measurement forced it: the calibrated threshold for
`pca_recon` spans **1.04 to 30,675 across the 28 machines**, a factor of ~29,500. A single
global cut-off would be set by whichever machine happens to be noisiest and would say nothing
about any of them. Production monitoring baselines per entity for exactly this reason.

Calibrating on the same machine is legitimate *only* because of the temporal disjointness,
which is enforced in `data.smd.segment_bounds` and tested in
`tests/test_data.py::test_segments_are_drawn_from_the_requested_half_only`.

## Why nominal significance levels are not used

Every two-sample test here assumes i.i.d. draws. SMD violates that badly: mean absolute
autocorrelation is still 0.29–0.57 at lag 100, and decay is not monotone — it dips near lag
300 and rises again near lag 600, which is a daily cycle. Measured per machine, the lag at
which it first falls below 0.3 is 73–191 rows; below 0.1 it is 231–1124.

Consequences, both handled explicitly:

1. **Thresholds are empirical.** A KS p-value on 2500 autocorrelated rows behaves like a test
   on ~15–35 independent ones. Thresholding on a nominal α would give a false-alarm rate with
   no relation to the one requested.
2. **Window sizes are set by the autocorrelation, not by convenience.** A 500-row window holds
   roughly three effectively independent samples, and no two-sample test can work there. This
   is the binding constraint on latency: nothing here can be expected to notice a change in
   under ~40 hours of telemetry.

`*_thinned` detector variants subsample by the estimated autocorrelation time before testing,
attacking the cause rather than the symptom.

## Operating point

All detectors are compared at a **matched false-alarm budget** of 0.5 alarms per drift-free
stream. Comparing detectors at their own arbitrary scales is meaningless; a matched budget is
both the only fair comparison and the one an operator actually faces.

## Alarm accounting

Windows are classified relative to the onset:

- **clean** — `window.end <= onset`; the only windows that can produce a false alarm
- **straddle** — overlapping the onset; excluded from both false alarms and detections, and
  the count is reported so the exclusion is visible
- **post** — `window.start >= onset`; the only windows that can produce a detection

Straddling windows contain both regimes. Counting them as false alarms would punish a detector
for being right slightly early; counting them as detections would reward it for a partial view.

**False alarms are measured on separate drift-free streams (`scenario = none`)**, never on the
pre-onset region of a drifted stream. Using the pre-onset region would make the false-alarm
rate depend on where the onset happens to sit, which is a property of the injection rather than
of the detector.

## Metrics

- **Detection delay** — windows between the first fully post-onset window and the first alarm.
  Reported as a distribution (median, IQR), never as a mean: it is right-skewed and censored by
  definition.
- **Recall within tolerance** — detected within 4 windows. A later detection is a miss.
- **False alarms per stream** — on drift-free streams.
- **Detectability curve** — recall as a function of injected magnitude. The most informative
  artefact in the project: it answers "how big must a change be before this notices".
- **Attribution precision** — top-*k* attributed features against ground truth, with *k* equal
  to the true number of affected features. This is generous (an operator does not know *k*), so
  the numbers are an upper bound.
- **Bootstrap 95% CI** over streams for every headline number.

**No superiority claim is made across overlapping confidence intervals.**
`scripts/analyse.py::superiority_report` emits `inconclusive` rather than a winner, and the
README reports it that way.

## Leakage checklist

| Risk | Mitigation | Enforced by |
|---|---|---|
| Temporal leakage | transforms fitted on the reference window only | `ReferenceTracker`; `tests/test_windows_and_timeseries.py` |
| Entity leakage | machine-level split | config; `tests/test_data.py` |
| Calibration/evaluation overlap | disjoint time halves per machine | `segment_bounds`; `tests/test_data.py` |
| Threshold tuned on evaluation data | thresholds frozen in a separate stage before evaluation runs | separate CLI stages |
| Circular evaluation | injection spec written without knowledge of any detector statistic | `tests/test_detectors.py` |
| Anomaly/drift conflation | negative control against SMD anomaly labels | `--stage negative_control` |
| Multiple testing | 38 tests per window; Bonferroni variant reported alongside | `KSBonferroniDetector` |
| Seed cherry-picking | fixed seeds in config; distributions reported | `configs/default.yaml` |

## Revisions

Three changes were made after the first pilot run, all before any result was recorded, and all
forced by measurement rather than by preference:

1. **Window 500 → 2500 rows, reference 1000 → 4000.** The pilot showed every detector nearly
   saturated on drift-free data. Diagnosis: with τ ≈ 150, a 500-row window holds ~3 independent
   samples.
2. **Global thresholds → per-machine thresholds**, with a temporal split inside each machine.
   The pilot showed the drift-free null differing across machines by up to ~29,500×.
3. **Stream length capped at 14,000 rows.** The smallest machine has 23,688 clean rows; after
   reserving 35% for calibration, 15,397 remain.

Nothing in this list was chosen after looking at a recall number.
