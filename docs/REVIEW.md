# Adversarial review

Two passes over the repository, reading it as if someone else had written it and the job were
to find the reason not to trust the results.

Round 1 findings are below with their resolution. Round 2 follows.

---

## Round 1

### R1-1 — CRITICAL — `hash()` is not stable across processes, so runs do not reproduce

`experiment.build_stream` seeded the per-stream RNG with
`abs(hash((machine.name, scenario, magnitude, seed, part)))`. Python randomises `hash()` for
`str` on every interpreter start unless `PYTHONHASHSEED` is pinned. The seed therefore changed
between processes, which changed *which slice of clean history* each stream was drawn from.

Impact: the README's reproducibility claim was false, and two runs of `make all` would silently
evaluate on different data. Nothing about the conclusions is invalidated — segment choice is
not confounded with detector — but the claim was wrong.

**Resolution:** replaced with `stream_seed`, a BLAKE2b digest of the stream identity. Fixed, and
the full pipeline was re-run from scratch so every published number comes from the stable-seed
code. Regression test added: `tests/test_experiment.py::test_stream_seed_is_stable_across_processes`.

### R1-2 — MAJOR — `detectability_table` reported a false-alarm rate it should not have

The column `false_alarms_per_stream` was computed from the clean region of *drifted* streams,
while `docs/VALIDATION_DESIGN.md` states that false alarms are only ever measured on separate
drift-free streams. Both statements cannot be true.

The pre-onset region is a bad estimator here: how many clean windows exist depends on where the
onset was placed, which is a property of the injection, not of the detector.

**Resolution:** column removed from `detectability_table`. The false-alarm rate is reported only
from `scenario == "none"` streams, in `false_alarm_rate_*.csv` and `calibration_transfer.csv`.

### R1-3 — MAJOR — `window_auc` pools negatives across machines

Positives for a cell come from whichever machines produced those streams; negatives were pooled
over all drift-free windows from all machines. Because the null level differs between machines
by orders of magnitude (§ threshold spread), part of the measured separation could be *machine
identity* rather than drift.

**Resolution:** AUC is now computed **per machine and then averaged over machines**, so a
detector can no longer score by telling machines apart.

The change was not cosmetic. Most numbers moved by 0.02–0.08, and one conclusion changed:
`wasserstein_max` on `correlation_break` went from 0.561 with a CI containing 0.5 to **0.620
[0.531, 0.700]**, which excludes it. See R2-7 — that turned out to expose a claim in the README
that was too strong.

### R1-4 — MODERATE — `attribution_scores` returned precision and recall that are equal by construction

With top-*k* selection where *k* is the true number of affected features, precision and recall
are the same number. Reporting both invites a reader to treat them as two pieces of evidence.

**Resolution:** one metric, named `precision_at_k`, with the identity stated in the docstring
and the README.

### R1-5 — MODERATE — the bootstrap silently capped itself

`window_auc` accepted `n_bootstrap` and then used `min(n_bootstrap, 500)`. A caller asking for
2000 draws got 500 and was not told.

**Resolution:** the cap is gone; the caller's value is used and the actual number of draws is
recorded in the output table.

### R1-6 — MODERATE — the null sample for the oracle threshold used 5 of 16 available windows

`matched_far_recall` estimated the per-stream false-positive rate from the first
`tolerance + 1` windows of each drift-free stream, discarding two thirds of the available null
data and making the threshold noisier than it needed to be.

**Resolution:** the rate is now estimated over every length-`horizon` block of each drift-free
stream, which uses all of the null data at the same horizon.

### R1-7 — MINOR — a third of all windows are discarded and the README did not say so

With onset at 60% and 2500-row windows on a 500-row stride, 5 of 16 windows straddle the onset
and are excluded from both false alarms and detections. The exclusion is correct and deliberate,
but the reader should be told its size.

**Resolution:** stated in the README and in `VALIDATION_DESIGN.md`.

### R1-8 — MINOR — the negative control reported a correlation without an alarm rate

A weak correlation between score and labelled-anomaly density is consistent with "the detector
ignores point anomalies" *and* with "the detector alarms on everything". The two are
distinguishable only if the alarm rate is reported alongside.

**Resolution:** `negative_control_report` now reports the alarm rate on the test half at each
machine's calibrated threshold, next to the correlation.

### R1-9 — NOT FIXED, documented — streams from one machine share a substrate

The bootstrap resamples streams, and streams from the same machine are drawn from overlapping
regions of that machine's history. The intervals are therefore mildly optimistic. Resampling
machines instead would be correct but leaves 11 units, which is too few to be informative.

Documented in `docs/METHODOLOGY.md` § 9 rather than papered over.

### R1-10 — NOT FIXED, scope — held-out machines are still untouched

Correct discipline, but it means every number is from the development split. Stated in the
README's limitations rather than presented as a final result.

---

## Round 2

Re-read after the round 1 fixes, looking specifically for damage done by them.

### R2-1 — checked: does per-machine AUC change any conclusion?

The separation between joint and marginal detectors on `correlation_break` survives and widens
(`pca_recon` 0.845 → 0.891). But one number crossed a line — see R2-7.

### R2-7 — MAJOR — "blind by construction" was too strong, and the data said so

`wasserstein_max` scores 0.620 [0.531, 0.700] on `correlation_break`. The interval excludes 0.5,
so it is detecting *something*, which contradicts the claim that a marginal statistic cannot see
this change at all.

Investigated rather than explained away. The injection preserves the marginal exactly over the
**whole post-onset region** — that is what the test asserts, and the test is correct. But a
scored window is a 2500-row *subsample* of that region, and block permutation moves values
between windows, so a window's empirical marginal is not the one it would otherwise have had.
Measured directly: the per-window KS statistic between permuted and original is 0.010–0.022.
Small, and enough for a statistic sensitive to whole-distribution mass movement to pick up.

**Resolution:** the claim in the README is now the precise one — marginal statistics have no
access to the dependence change itself, and what they detect is a finite-window artefact. The
test was not weakened; a second measurement was added to explain the gap between what the test
guarantees and what the detector sees.

### R2-8 — MAJOR — a configuration that never did anything

`reset_on_alarm` produced output **bit-identical** to `fixed`. `ReferenceTracker.notify_alarm` is
implemented and unit-tested, but `score_stream` never calls it: scoring happens before thresholds
are applied, so nothing in the pipeline knows an alarm occurred.

Found by noticing that two supposedly different configurations agreed to the last digit — which
is the kind of agreement that should always be suspicious.

**Not fixed in this iteration.** Wiring it needs an online threshold, which is a design change
rather than a patch. Recorded in the README's error analysis and limitations, and the policy
table reports `fixed` and `sliding` only rather than presenting a third row that means nothing.

### R2-9 — checked: two pre-registered predictions failed

`iforest_mean` was predicted mediocre and is the strongest detector; a sliding reference was
predicted to lose most of its power on `incremental_shift` and is the one place it wins. Both are
reported as failures in the README with the reasoning that was wrong, not deleted.

Four of six pre-registered predictions held. That ratio is worth stating: a pre-registration
where everything is confirmed usually means the predictions were written to be safe.

### R2-2 — checked: is the control still a control?

`random` measures AUC ≈ 0.46 and a calibration-transfer inflation of 1.23×. Both are what a
data-independent scorer must produce. This is the property that makes every other number
interpretable, and it is now asserted by a test rather than only observed.

### R2-3 — checked: any remaining path from ground truth into a detector?

Traced `DriftSpec` through the call graph. It reaches `inject`, the stream metadata, and the
evaluation. No detector constructor or `score` call receives it. `attribution_for_stream` reads
`spec.affected` only *after* the detector has produced its vector.

### R2-4 — checked: does anything still tune on evaluation data?

The oracle operating point does, deliberately and with the label attached in every table and in
the README. The deployed thresholds do not: they come from a stage that runs before evaluation
and reads a disjoint time range.

### R2-5 — accepted risk: the autocorrelation threshold of 0.3 is a judgement call

It changes the thinned variants materially. Both the 0.1 and 0.3 operating points are reported
in `METHODOLOGY.md` with the resulting thinning steps, so a reader can see the trade rather than
take the constant on faith. A sweep over the threshold would be better and is listed as future
work rather than claimed.

### R2-6 — style pass

Removed an empty `attribution/` package that existed only because the plan listed it as a
module: attribution is a method on the detector protocol and a scoring function in
`evaluation`, and a directory containing nothing but `__init__.py` is a promise the code does
not keep.

Checked for the usual signs of generated-looking code and did not find them: no `utils.py`, no
class with a single method, no factory wrapping a constructor, no `try`/`except` swallowing an
error to keep a run alive. Total: 3,415 lines of Python across source, scripts and tests, of
which tests are roughly a third.
