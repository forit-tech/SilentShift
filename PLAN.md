# SilentShift — PLAN

Adaptive drift & behavioral change detection in multivariate system telemetry.

**Status: plan only. No code written, no experiment run, no number in this document is a result.**
Every metric named below is a metric we intend to compute, not one we have.

---

## 1. Problem definition

A system emits multivariate telemetry continuously. Over time its behaviour changes: a new
operating mode appears, a subpopulation of machines starts behaving differently, the correlation
between two signals breaks, a degradation creeps in over days. Nobody labels the moment it starts.

The question SilentShift answers is not *"is this point anomalous"* but:

> **Has the process that generates this data changed, when did it start, and which signals carry
> the change?**

Three sub-problems, in order of difficulty:

1. **Detect** that the generating distribution changed, without a supervised signal.
2. **Localise in time** — estimate the onset, not just raise a flag eventually.
3. **Attribute** — name the features responsible, so the alert is actionable.

### Why this is not anomaly detection

This distinction is load-bearing for the whole project and is the first thing an interviewer
should ask about.

| | Point/segment anomaly | Distributional drift |
|---|---|---|
| Unit of judgement | a timestamp | a window / population |
| Ground truth | this point is weird | the process changed at time *t* |
| Correct response | investigate the event | recalibrate, retrain, or accept a new normal |

A dataset labelled for anomalies is **not** a drift benchmark. Treating anomaly labels as change
points is a category error, and one this project explicitly refuses to make (see §3).

---

## 2. Candidate datasets

Five candidates were examined. For each: what it is, who published it, licence, and the known
criticism in the literature.

### 2.1 SMD — Server Machine Dataset

- **Source**: NetMan Lab, Tsinghua University, released with OmniAnomaly (KDD 2019),
  <https://github.com/NetManAIOps/OmniAnomaly>
- **Licence**: MIT (repository licence covers the bundled `ServerMachineDataset` folder).
- **Shape**: 28 server machines, 38 monitored variables each, 1-minute interval, 5 weeks.
  ~708k train rows / ~708k test rows, ~4.16% of test points labelled anomalous. Timestamps
  anonymised. Ships `interpretation_label` naming which dimensions contribute to each anomaly.
- **Known criticism**: Wu & Keogh, *Current Time Series Anomaly Detection Benchmarks are Flawed*
  (arXiv:2009.13807) analyse SMD among others and argue popular benchmarks suffer from triviality,
  unrealistic anomaly density and mislabelling.

### 2.2 USP DS Repository — Insects streams

- **Source**: Souza et al., <https://sites.google.com/view/uspdsrepository>
- **Shape**: 33 optical-sensor features, 6 insect species, streams ordered so that a controlled
  air-temperature schedule induces **abrupt / incremental / gradual / incremental-reoccurring /
  incremental-abrupt-reoccurring** drift. Temperature is the concept and is *not* a feature.
- **Why it matters here**: this is one of the few real-world streams where drift ground truth is
  known by construction rather than asserted after the fact.
- **Licence**: to be confirmed against the repository page before use. If it cannot be confirmed,
  this dataset is dropped rather than used on assumption.

### 2.3 Backblaze Drive Stats

- **Source**: <https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data>
- **Licence**: free to use with three conditions — cite Backblaze, accept sole responsibility for
  use, do not resell the data itself.
- **Shape**: daily SMART snapshots per drive, 388M+ records, growing ~240k/day.
- **Why rejected as primary**: the drift is real but *confounded*. Fleet composition changes over
  time — new drive models arrive, old ones are retired — so a detected distribution shift is often
  a change in *which drives exist*, not a change in drive behaviour. Disentangling that is a
  different project. Kept as a possible stretch experiment, clearly labelled.

### 2.4 ELEC2 / Electricity — **rejected**

The classic concept-drift benchmark, and the wrong choice. Žliobaitė, *How good is the Electricity
benchmark for evaluating concept drift adaptation* (arXiv:1301.3524): the labels are strongly
temporally dependent — a naive "predict the previous label" classifier reaches ~85% accuracy where
independence would give ~51%. Accuracy and Kappa both fail to expose this, and change detectors
evaluated on it can look good for reasons unrelated to drift. Using it would be a signal that the
methodology was not read.

### 2.5 NAB / SMAP / MSL — **rejected**

Same family of criticism as §2.1 but more severe, and they are anomaly benchmarks rather than drift
benchmarks. Wu & Keogh introduced the UCR Anomaly Archive specifically as a replacement.

---

## 3. Dataset selection and the honesty problem

**Primary substrate: SMD. External validation: USP Insects (licence permitting).**

The obvious objection has to be answered before any code exists:

> If you inject the drift yourself and then detect it, you have proven nothing.

That objection is correct as stated, and the design answers it in three ways.

**(a) SMD is used as a substrate, not as a benchmark.** We are not claiming state of the art on
SMD's anomaly labels — which is exactly what Wu & Keogh criticise. We use SMD because its
*marginals, cross-feature correlations, autocorrelation and per-machine heterogeneity* are real
telemetry rather than a Gaussian simulator. The realism we need is in the noise, not in the labels.

**(b) Injection is parameterised independently of detection.** The injection framework is written
against a spec of *what changed in the data-generating process*, with no knowledge of which
statistic any detector computes. Concretely, the drift catalogue deliberately includes changes that
some detectors cannot see by construction:

| Injected drift | Ground truth we control | Detectors expected to fail |
|---|---|---|
| Sudden covariate shift | onset *t*, affected features, magnitude | — |
| Gradual (mixture ramp over window *w*) | onset, end, mixing schedule | window-based tests with a short window |
| Incremental (slow mean/scale creep) | onset, rate | tests with a sliding reference that absorbs the creep |
| New behavioural subgroup (fraction *p* of entities) | onset, which entities, subgroup profile | population-level tests when *p* is small |
| **Correlation-structure change with marginals held fixed** | onset, which feature pair | **PSI, KS, per-feature Wasserstein — blind by construction** |
| Recurring drift (return to a previous regime) | onset of each regime | detectors that reset their reference on alarm |

The last two rows are the point. A framework that only injects mean shifts and only tests mean
shifts is theatre. Reporting that per-feature divergence tests score ~0 recall on a pure copula
change is a *result*, not a failure — it is the argument for multivariate detection.

**(c) SMD's own anomaly labels are held out as a negative control, not as drift labels.** They tell
us where point anomalies live so we can check that a drift detector is not merely re-detecting
them. If our "drift" alarms line up with SMD anomaly segments, the detector is doing anomaly
detection with extra steps and we say so.

**Deliverable**: `docs/DATA.md` with source URL, download timestamp, file checksums, licence text,
and this reasoning.

---

## 4. Observational unit

Ambiguity here is where drift projects quietly go wrong, so it is fixed up front.

- **Row**: one `(machine_id, timestamp)` with 38 telemetry values.
- **Detection unit**: a **window** of consecutive rows for one machine — drift is a property of a
  distribution, so no single row can be "drifted".
- **Evaluation unit**: one **stream** = one machine × one injected drift scenario × one seed.
  Metrics aggregate over streams, never over windows, because windows within a stream are
  massively dependent and averaging over them would fabricate precision.
- **Fleet-level variant**: a window across *all* machines at time *t*, used only for the
  new-subgroup scenario where the change is in the population mix.

---

## 5. Objective

Unsupervised. There is no target column.

For each stream, the system emits a sequence of alarms `(t_alarm, score, attributed_features)`.
Evaluation compares that sequence against the injected ground truth `(t_onset, drift_type,
affected_features, magnitude)`.

---

## 6. Validation design

Written before any training code — this becomes `docs/VALIDATION_DESIGN.md`.

### 6.1 Splits

Three disjoint machine groups, split **by machine**, not by row:

| Split | Purpose | Sees ground truth? |
|---|---|---|
| **Calibration** (≈40% of machines) | set detector thresholds on *drift-free* segments | no drift injected |
| **Development** (≈40%) | design, tune, iterate, error analysis | yes |
| **Held-out** (≈20%) | final numbers, touched once | yes |

Machine-level splitting because the 38 signals of one machine are far more similar to each other
across time than to another machine's. A row-level split would leak a machine's behavioural profile
into calibration and inflate everything.

### 6.2 Threshold calibration — the part that usually goes wrong

KS, PSI and friends come with nominal p-values that assume i.i.d. samples. Telemetry is heavily
autocorrelated, so those p-values are simply wrong, and a detector tuned to α = 0.01 will not fire
at 1%. Therefore:

- Thresholds are calibrated **empirically** on drift-free calibration-split segments, targeting a
  chosen false-alarm rate per stream-hour.
- Nominal α is never used as a threshold. If we report one, it is to show how far off it is.
- Per-feature tests across 38 features are a multiple-testing problem; Benjamini–Hochberg is
  applied and the uncorrected variant is reported alongside to show the size of the effect.

### 6.3 Reference-window policy

Every detector must declare how its reference window behaves: fixed, sliding, or reset-on-alarm.
This is not an implementation detail — a sliding reference silently absorbs incremental drift and
will report a near-zero detection rate on exactly the drift type it is supposed to catch. All three
policies are evaluated, and the interaction between policy and drift type is a headline table.

### 6.4 Metrics

Defined precisely, because "detection delay" has at least four incompatible definitions in the
literature:

- **Detection delay** — `t_first_alarm − t_onset`, counted only for alarms inside a tolerance
  window `[t_onset, t_onset + H]`. Alarms before `t_onset` are false alarms, never early detections.
  Reported as a distribution (median + IQR), not a mean; the distribution is right-skewed and
  censored by definition.
- **Missed detection rate** — fraction of streams with no alarm inside the tolerance window.
- **False alarm rate** — alarms per stream-hour, measured on **separate drift-free streams**, never
  on the pre-onset portion of a drifted stream.
- **Precision / recall** at the alarm level, with the tolerance window stated in every table.
- **Detectability curve** — recall as a function of injected drift magnitude. This is the most
  informative plot in the project: it answers "how big does a change have to be before this thing
  notices", which is the question an operator actually has.
- **Attribution accuracy** — precision/recall over the set of affected features, against the
  injected ground truth.
- **Bootstrap CIs over streams** for every headline number. No claim that method A beats method B
  if the intervals overlap.

---

## 7. Leakage and statistical risks

| Risk | How it would happen here | Mitigation |
|---|---|---|
| Temporal leakage | scaler/PCA fitted on the whole stream, including post-drift data | all transforms fitted on the reference window only, refitted per stream |
| Entity leakage | same machine in calibration and held-out | split by machine |
| Circular evaluation | injections designed around the detector's statistic | injection spec written first, drift types included that specific detectors provably cannot see |
| Anomaly/drift conflation | SMD anomaly segments re-detected and called drift | anomaly labels used as negative control; overlap reported |
| Autocorrelation → wrong α | nominal p-values on dependent samples | empirical calibration, FAR measured on drift-free streams |
| Multiple testing | 38 per-feature tests per window | BH correction, both variants reported |
| Threshold tuned on evaluation data | picking the threshold that maximises the reported score | thresholds frozen on calibration split before held-out is touched |
| Seed cherry-picking | one lucky injection seed | ≥ N seeds per scenario, distributions reported, seeds fixed in config |

---

## 8. Baselines

In increasing order, and the cheap ones are not decoration — the honest possibility is that a
per-feature KS test with a properly calibrated threshold is hard to beat on most drift types, and
if that is what we find, that is what the README will say.

1. **Always-alarm / never-alarm** — degenerate bounds that make FAR and recall interpretable.
2. **Persistence-style control** — alarm at random with the same rate as the best detector; any
   method that fails to beat this is not detecting anything.
3. **Per-feature two-sample tests** — KS, Wasserstein, Jensen–Shannon on binned marginals; PSI as
   the industry-standard variant.
4. **Univariate change-point detection** — offline (PELT-style cost segmentation) and an online
   sequential variant, for onset localisation rather than just alarming.
5. **Multivariate density-ratio / classifier two-sample test** — train a classifier to separate
   reference from current window; its AUC is a drift statistic. This is the baseline that should
   catch the correlation-structure change the marginal tests miss.
6. **Isolation Forest** on windowed feature aggregates — included because it is what people reach
   for, and to demonstrate that a point-anomaly scorer aggregated over a window is a mediocre drift
   detector.
7. **Representation-based** — PCA reconstruction error against a reference-fitted basis; optionally
   a small autoencoder **only if** it beats PCA by more than the bootstrap interval. If it does not,
   it gets deleted and the README says it was tried and did not help.

UMAP/HDBSCAN are for the exploratory notebook and the subgroup-discovery scenario only. Neither is
used to produce a headline metric, because neither is stable enough under reseeding to support one.

---

## 9. Attribution — "why did it fire?"

An alarm with no explanation is not usable. Three mechanisms, evaluated against injected ground truth:

1. **Per-feature divergence ranking** — rank features by their individual test statistic.
2. **Classifier-based importance** — permutation importance on the reference-vs-current classifier.
   Expected to beat (1) on correlation-structure drift, where no single marginal moved.
3. **Contribution to reconstruction error** — per-feature share of the PCA residual.

Plus a before/after visualisation for the top-ranked features. Attribution is scored, not just
displayed — precision/recall against `affected_features`.

---

## 10. Repository architecture

```
silentshift/
├── src/silentshift/
│   ├── data/          loading, checksum verification, machine-level splits
│   ├── injection/     drift catalogue + injection engine (writes ground truth alongside)
│   ├── windows/       windowing, reference-window policies
│   ├── detectors/     one module per detector family, common protocol
│   ├── attribution/   feature-contribution methods
│   ├── evaluation/    delay/FAR/detectability, bootstrap CIs
│   └── reporting/     plots and tables
├── tests/
├── notebooks/         exploration only, imports the package
├── configs/           YAML: seeds, scenarios, windows, thresholds
├── scripts/           make targets call these
├── docs/              DATA.md, VALIDATION_DESIGN.md, METHODOLOGY.md, REVIEW.md
├── Makefile
├── pyproject.toml
└── README.md
```

One deliberate omission: no Docker, no MLflow, no Airflow. Nothing in SilentShift needs them, and
adding them would be resume-driven design. DecisionForge is where that infrastructure earns its
place.

---

## 11. Expected failure modes

Written down now so they cannot be quietly rediscovered and reframed as successes later.

- Correlation-structure drift is invisible to every marginal test — expected, and the reason the
  classifier two-sample test is in the baseline set.
- Small-subgroup drift (p ≈ 5% of entities) is likely undetectable at the fleet level at any
  reasonable FAR. If so, that is a finding about aggregation, not a bug.
- Gradual drift with a long ramp may never trigger a sliding-reference detector at all.
- The autoencoder probably does not beat PCA on 38 dimensions with this much data. Prediction
  recorded in advance so the outcome cannot be rationalised afterwards.
- Detection delay and false-alarm rate trade off directly; any single-number comparison between
  detectors will be misleading, so the primary artefact is a delay-vs-FAR curve, not a leaderboard.

---

## 12. Definition of Done

- [ ] dataset provenance, licence and checksums in `docs/DATA.md`
- [ ] `docs/VALIDATION_DESIGN.md` written **before** the training code
- [ ] injection framework produces machine-readable ground truth for every stream
- [ ] all drift types in the §3 catalogue implemented, including the ones expected to defeat baselines
- [ ] thresholds calibrated on drift-free calibration split; held-out touched once
- [ ] every reported number produced by an actually-executed run
- [ ] bootstrap CIs on headline comparisons; no superiority claim across overlapping intervals
- [ ] negative control against SMD anomaly labels reported
- [ ] attribution scored against ground truth, not just plotted
- [ ] detectability curve + delay-vs-FAR curve produced
- [ ] `pytest`, `ruff`, `mypy` actually run, output pasted into REVIEW.md
- [ ] adversarial review round 1 → fixes → round 2
- [ ] README matches the code; limitations section names what failed
- [ ] reproduction tested from a clean clone
- [ ] no secrets, licence checked
- [ ] **nothing published until explicitly approved**

---

## 13. Open questions for you

1. **USP Insects licence** — I could not confirm it from the repository page in this pass. If it
   turns out to be unclear, do we drop external validation and rely on injection alone (weaker but
   honest), or spend the time finding a substitute real-drift stream?
2. **Scope of the autoencoder arm.** It adds real time and will probably lose to PCA. Include it as
   a documented negative result, or cut it and keep the project tighter?
3. **Backblaze stretch experiment** — worth it for the "real, messy, confounded" story, or scope creep?
