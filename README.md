# Confforge

[![CI](https://github.com/CJX0712/confforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/confforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v0.1.0-blue)](https://github.com/CJX0712/confforge/releases)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/tests-192%20passed%20%7C%2090%25%20coverage-brightgreen)](https://github.com/CJX0712/confforge/actions)

**Distribution-free predictive inference for regression, with the benchmarks that
say when it stops working.**

Confforge compares eight conformal prediction methods on six synthetic
data-generating processes, and reports the failures as prominently as the wins.
Every gate is a falsifiable claim with a fixed threshold; every number in the
report comes from a real run; and the cases where a method *loses* are recorded
rather than quietly dropped.

---

## What it does

Conformal prediction buys **distribution-free coverage**: an interval built from
a held-out calibration set covers a fresh test point at the nominal rate *with no
assumptions about the data*. What it does not buy is *sharpness*. A single
global interval width must cover the noisiest region of feature space, so it
over-covers everywhere else.

Confforge measures how much sharpness eight methods recover:

| # | method | mechanism | family |
|---|--------|-----------|--------|
| 1 | `split_absolute` | pooled `\|y - ŷ\|` quantile → symmetric window | baseline |
| 2 | `split_normalized` | `\|y - ŷ\| / σ̂(x)`, kNN local scale | baseline |
| 3 | `cqr` | Romano 2019 conformity score on a quantile regressor | baseline |
| 4 | `mondrian` | per-group calibration, thin cells shrunk to pooled | baseline |
| 5 | **`vennfuse`** | **CQR anchor + Mondrian shrink + distilled ACI** | **flagship** |
| 6 | `mapie_split` | MAPIE `SplitConformalRegressor` | Tier-0 backend |
| 7 | `mapie_cqr` | MAPIE `ConformalizedQuantileRegressor` | Tier-0 backend |
| 8 | `aci` | online adaptive conformal inference (Gibbs & Candès) | baseline |

The MAPIE pair is an **independent check on our own implementation**. That
`mapie_split` reproduces `split_absolute` to the last decimal is the evidence
that both are right; a conformal library only ever compared against itself can
hide a shared misreading of the theory.

## Install

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -e ".[backend,dev]"
```

Python 3.12 or 3.13. The `backend` extra (MAPIE) is optional — without it the
package runs normally and the benchmark records a *skip with a reason* rather
than failing.

## Quick start

```bash
python examples/run_demo.py       # full benchmark, writes results/benchmark.json
python cli.py gates               # what the release gates actually assert
python cli.py methods             # list methods and mechanisms
python cli.py selftest            # verify the MAPIE API contract
make verify                       # lint + format + tests + backend self-test
```

The demo exits non-zero when a P0 gate fails, so it doubles as a CI smoke test.

## Results

Full grid: 6 DGPs × 8 methods × 3 seeds = **144 cells**, `alpha = 0.1`.
Mean width / coverage:

| dataset | split_absolute | cqr | mondrian | vennfuse | mapie_split | mapie_cqr |
|---|---|---|---|---|---|---|
| homoscedastic | 1.173 / 0.904 | 1.169 / 0.908 | 1.194 / 0.908 | 1.208 / 0.917 | 1.173 / 0.904 | 1.709 / 0.887 |
| heteroscedastic | 3.394 / 0.902 | **2.907** / 0.902 | 3.090 / 0.913 | 3.073 / 0.910 | 3.394 / 0.902 | 3.110 / 0.906 |
| grouped_hetero | 2.728 / 0.895 | 2.753 / 0.895 | **2.456** / 0.914 | 2.621 / 0.909 | 2.728 / 0.895 | 3.091 / 0.893 |
| covariate_shift | 1.165 / 0.551 | 1.223 / 0.599 | 1.359 / 0.594 | 1.436 / 0.668 | 1.165 / 0.551 | 3.460 / 0.615 |
| small_calib | 1.406 / 0.949 | 1.375 / 0.948 | 1.403 / 0.949 | 1.382 / 0.948 | 1.406 / 0.949 | 1.941 / 0.946 |
| nonlinear | 1.604 / 0.905 | **1.490** / 0.901 | 1.657 / 0.915 | 1.500 / 0.903 | 1.604 / 0.905 | 2.541 / 0.906 |

Read this table honestly:

- **`covariate_shift` breaks everything.** Every method under-covers (0.55–0.67
  against a nominal 0.9) once test covariates leave the training support. This is
  the central limitation of split conformal prediction and the benchmark refuses
  to average it away.
- **`small_calib` over-covers by ~5 points.** With `n_calib = 25` the conformal
  quantile is forced to the maximum, so validity is bought with useless width.
  Finite-sample conservatism, working as designed.
- **`mapie_cqr` is wider on 5 of 6 datasets** (the exception is
  `heteroscedastic`, where it comes in at 3.110 against the baseline's 3.394).
  It builds its own quantile grid internally and so is not sharing our
  residual-anchored learner; it is a cross-check, not a competitor. The earlier
  claim of "uniformly wider" was wrong on that one dataset.

### Release gates

| gate | claim | measured | verdict |
|------|-------|----------|---------|
| **G1** (P0) | flagship ≥ 8% narrower than `split_absolute`, coverage held | **5.58%** | ❌ **FAIL** |
| **G2** (P1) | worst-group coverage violation cut ≥ 20% on `grouped_hetero` | **81.0%** (0.169 → 0.032) | ✅ PASS |
| **G3** (P0/P1) | ACI gains ≥ 3pp post-drift coverage at ≤ 1.25× width | **+3.75pp**, width ratio 1.124 | ✅ PASS |
| **G5** (P0) | two runs at the same seed agree bit-for-bit | **max abs diff = 0.0** (1332 values) | ✅ PASS |

**G1 does not pass, and the number above is the real one.** The flagship is 5.58%
narrower, against an 8.0% threshold. See
[docs/architecture.md](docs/architecture.md#why-g1-does-not-pass) for the
per-dataset breakdown and the reason. The threshold was not moved.

### Invariant checks

The benchmark re-examines every emitted interval for structural correctness
(ordered bounds, finite width, no one-sided infinity, shape agreement) and every
row for non-finite metrics. On the shipped run: **144 cells structurally checked,
0 violations**, recorded as `invariant_checks_performed: 144` in
`results/benchmark.json`.

That count field is not decoration. An earlier version reported
`invariants: []` — and the list was empty because the checker had been invoked
without the intervals, so it had examined *nothing*. `[]` from a checker that ran
and from one that did not is the same bytes. The report now carries the number of
cells actually examined, so the two cases cannot be confused.

One check is deliberately **not** counted as evidence: "bounds are ordered" is
unreachable through the public API, because `IntervalSet.__post_init__` raises
before such an object can exist. It is retained as defence in depth; no passing
run has ever exercised it.

## Design decisions worth knowing

**One learner for all eight methods.** A tempting shortcut is to give CQR a
gradient-boosted quantile model and split conformal a linear baseline, then
report that CQR is sharper. That measures *model class*, not conformal
machinery. Confforge fits one `ModelBundle` per (dataset, seed) and hands the
same objects to every method.

**The quantile pair is fitted on residuals, not on `y`.** Fitting conditional
quantiles directly asks the learner to rediscover the mean *and* the spread. The
median objective is solved far better than pinball at `alpha = 0.05`, so the raw
window under-covers (0.828 against a 0.90 target) and conformal has to inflate it
back. Fitting the pair on residuals of the median model gives the *same* width
but 0.909 coverage, and moved the CQR-vs-split gap on `homoscedastic` from
+17.7% to +1.8%.

**Thin cells are shrunk, and infinite cells fall back.** A 3-point group has a
"conformal quantile" equal to its maximum. Group quantiles are shrunk toward the
pooled value with `λ = 10`, and a cell whose quantile is *undefined* (`k > n`)
falls back to the pooled value rather than propagating `+inf` into the table.

**A negative CQR correction is clamped, not trusted.** `[q_lo - q, q_hi + q]` is
empty whenever `q < -(q_hi - q_lo)/2` — observed inverting 1515 of 2000 test
intervals before the clamp existed. Clamping to the feasibility boundary
*provably preserves* the guarantee: any `y` with `score(y) ≤ (q_lo-q_hi)/2` must
equal the window centre, which the clamped interval contains.

**G5 excludes `elapsed_sec`.** Wall-clock time is not reproducible by
construction; including it would guarantee a red gate for a reason unrelated to
determinism.

## Repository layout

```
core/       config, seeding, types, protocol definitions, error taxonomy
data/       six synthetic DGPs with difficulty knobs + the drift stream
methods/    shared estimators, conformal quantiles, scores, the 8 methods, MAPIE adapters
eval/       metrics, benchmark driver, release gates, drift, ablation, failures, invariants
pipeline/   end-to-end orchestration
cli.py      argparse entry point
examples/   run_demo.py
tests/      193 tests (192 passed, 1 skipped)
docs/       architecture.md, model_card.md
```

## Reproducibility

Every run is seeded from one integer through `core.seed.set_all`. Datasets are
generated with `core.seed.spawn`, which returns a *fresh* generator per call, so
regenerating a dataset after an unrelated draw yields byte-identical rows — the
cached-stateful alternative silently shifted the data and is regression-tested in
`test_core.py::test_spawn_is_fresh_and_order_independent`.

Gate G5 verifies this end to end by running the whole benchmark twice and
comparing 1332 values.

## License

MIT — see [LICENSE](LICENSE). Author: 晨星.
