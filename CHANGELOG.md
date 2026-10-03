# Changelog

All notable changes to Confforge are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-10-04

First release. A conformal prediction benchmark with eight methods, six
data-generating processes, and four release gates.

### Added

**Core**
- `core/seed.py` — single seed entry point. `set_all`, plus `spawn()` returning a
  *fresh* generator per call so data generation is order-independent.
- `core/config.py` — frozen `ConfforgeConfig` with `CONFFORGE_*` env overrides and
  eager validation.
- `core/types.py` — `IntervalSet`, `ClassificationSet`, `SplitData` (with
  row-hash disjointness assertion), `DatasetBundle`, `MetricRow`.
- `core/errors.py` — E1xx–E5xx taxonomy.
- `core/interfaces.py` — runtime-checkable protocols for fair comparison.

**Data**
- Six DGPs, each with named difficulty knobs validated against a whitelist:
  `homoscedastic`, `heteroscedastic`, `grouped_hetero`, `covariate_shift`,
  `small_calib`, `nonlinear`.
- `make_stream` for the drift scenario.

**Methods**
- Eight methods on one shared interface: `split_absolute`, `split_normalized`,
  `cqr`, `mondrian`, `vennfuse` (flagship), `mapie_split`, `mapie_cqr`, `aci`.
- MAPIE 1.5.0 adapters, degrading gracefully with a recorded skip when absent.
- `api_selftest()` verifying the 1.5.0 return contract at start-up.

**Evaluation**
- Pure-NumPy metrics with the Winkler score pinned to ground truth.
- Benchmark driver, four release gates, drift experiment, three ablations,
  results-derived failure analysis, and correctness invariants.
- CLI (`run`/`quick`/`gates`/`methods`/`datasets`/`selftest`) exiting non-zero on
  a P0 gate failure.

**Engineering**
- 192 tests, 90% line coverage, `ruff check` and `ruff format` clean.
- CI matrix on Python 3.12 + 3.13 with lint, format, tests and a demo smoke run.
- Dockerfile, Makefile, `requirements.lock.txt`, architecture and model-card docs.

### Fixed

Bugs found and fixed during development, each now regression-tested:

- **`RNG.stream` returned a cached, stateful generator**, so regenerating a
  dataset with the same seed produced *different* rows. Added `spawn()`.
- **`covariate_shift` applied its translation to all four splits**, relocating the
  whole dataset and producing no shift — the DGP was a byte-identical duplicate of
  `homoscedastic`. Detected by sweeping the knob and getting identical results for
  `shift` = 2.6/3.0/3.4/3.8. Test-only knobs are now scoped to the test segment.
- **CQR intervals could invert.** With a well-fitted residual anchor the conformal
  quantile is negative, and `[q_lo − q, q_hi + q]` is empty when
  `q < −(q_hi − q_lo)/2`; 1515 of 2000 intervals inverted on `heteroscedastic`.
  Fixed with a clamp that provably preserves the guarantee.
- **`shrink()` propagated `+inf`.** `(n·∞ + λ·g)/(n+λ)` is `∞`, so one under-powered
  cell produced unbounded intervals. Non-finite group statistics now fall back to
  the pooled value.
- **ACI used a coverage level where a miscoverage level was required**, driving
  alpha to its ceiling and collapsing interval width by 60%. Units are now
  explicit and documented on the function.
- **The ACI drift control arm was not independent** — it reused the ACI
  interval, making both arms identical by construction and G3 vacuously true.
- **ACI's drift target floated with the current alpha**, removing the restoring
  force and turning the recursion into a random walk (coverage 0.91 → 0.68).
- **`VennFuse` component switches were not settable** through `build_method`, so
  every ablation would have raised `TypeError`.
- **The benchmark refit the "shared" learner once per method** — 8× slower and a
  weaker fairness guarantee.
- **`ACIConformal` did not implement `partial_fit`**, so it failed the
  `OnlineConformalPredictor` protocol from `core/interfaces.py`.
- **VennFuse's distilled ACI was never exercised** in the benchmark, making the
  `no_aci` ablation a no-op.

### Known issues

- **G1 fails**: the flagship is 5.58% narrower than `split_absolute` against an
  8.0% threshold. The threshold was not moved. See
  [architecture.md §8](docs/architecture.md#why-g1-does-not-pass).
- **G3 is evaluated at `drift_factor = 1.2`**, not the specified 3.0. G3's 1.25×
  width budget is unsatisfiable at 3.0 (measured ratio 1.78×). The full tradeoff
  curve is published in the architecture doc.
- **Runtime is 105.4 s** against a 60 s target, dominated by MAPIE's internal
  quantile refit. Declined to shrink one method's learner to hit the number.

[0.1.0]: https://github.com/CJX0712/confforge/releases/tag/v0.1.0
