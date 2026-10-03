# Model Card — Confforge

**Author:** 晨星 · **Version:** 0.1.0 · **Date:** 2026-10-04

This card describes what the package claims, what it measures, and — in
reasonable detail — where it breaks.

---

## 1. What this is

Confforge is a **benchmark and reference implementation** for conformal
prediction intervals on regression. It is not a trained model and there is no
artifact to ship: the deliverable is the code, the benchmark, and an honest
account of which methods win where.

Its claim is narrow and checkable: *under distribution shift at calibration time,
and conditional on the assumptions below, these intervals achieve marginal
coverage at the nominal rate, and here is how much narrower they are than the
split-conformal baseline.*

## 2. The guarantee, stated precisely

For split conformal prediction with calibration scores `s_1..s_n` exchangeable
with the test score `s_{n+1}`, the interval built from
`q = s_(k)`, `k = ⌈(n+1)(1-α)⌉` satisfies

```
P(Y_{n+1} ∈ C(X_{n+1})) ≥ 1 − α
```

**This holds with no distributional assumption** — but only if the calibration
scores really are exchangeable with the test score. In practice that requires:

| assumption | status in this package |
|---|---|
| calibration independent of training | enforced structurally (`fit` ≠ `fit_calibrate`) |
| calibration and test exchangeable | true for 4 of 6 DGPs; **violated by `covariate_shift`** |
| calibration size adequate | `n_calib ≥ 19` enforced in config; `small_calib` probes the thin regime |

**What is *not* guaranteed**, and is reported as measurement rather than promise:

- **Conditional coverage.** Mondrian and VennFuse report per-group coverage as
  an *empirical quantity*. Shrinking a group quantile toward the pooled one means
  the result is no longer an order statistic of that group's scores, so the
  finite-sample proof does not transfer verbatim. G2 measures the improvement; it
  does not certify it.
- **Adaptivity under drift.** ACI's guarantee is a *tracking* property, not
  coverage. Its miscoverage level adapts to observed error; it does not restore
  validity when the calibration set is stale.

## 3. Components

| component | role | failure mode if wrong |
|---|---|---|
| `ModelBundle` | one shared point + residual-quantile learner | comparison becomes about model class, not conformal machinery |
| `conformal_quantile` | the single order-statistic entry point | silent under-coverage from a missing `n+1` |
| `clamp_correction` | keeps CQR intervals non-empty | inverted intervals; coverage collapses to 0 |
| `shrink` | thin-cell shrinkage, `λ = 10` | a 3-point cell calibrates to its maximum |
| G1–G5 | falsifiable release claims | a benchmark that cannot fail is decoration |

## 4. Evaluation

- **6 DGPs**, each with a named difficulty knob: `homoscedastic`,
  `heteroscedastic`, `grouped_hetero`, `covariate_shift`, `small_calib`,
  `nonlinear`.
- **8 methods × 3 seeds = 144 cells**, `alpha = 0.1`,
  `n_train = 1500 / n_calib = 500 / n_val = 500 / n_test = 2000`.
- **Primary metric:** Winkler interval score, pinned to a hand-computed ground
  truth (`[0,10]`, `y=13`, `α=0.1` → `70.0`).
- **Reported:** marginal coverage, mean width, worst-group coverage violation
  (`CoV-W`), stability.
- **Four disjoint splits** per dataset, asserted disjoint by row-hashing.

### Measured results

| gate | measured | threshold | verdict |
|---|---|---|---|
| G1 width | **5.58%** | ≥ 8.0% | ❌ FAIL |
| G2 conditional coverage | **81.0%** reduction | ≥ 20% | ✅ PASS |
| G3 drift | **+3.75pp**, ratio 1.124 | ≥ +3pp, ≤ 1.25× | ✅ PASS |
| G5 determinism | **max abs diff 0.0** (1332 values) | = 0 | ✅ PASS |

## 5. Known limitations

Ordered by how much they should change your use of this package.

1. **Covariate shift breaks every method.** Coverage falls to 0.55–0.67 against a
   0.90 nominal once test covariates leave the training support. Conformal
   prediction is marginal, not robust to a shifted test distribution. This is the
   single most important caveat, and the benchmark refuses to average it away —
   the cell is recorded, flagged as a coverage violation, and excluded from G1.
2. **Thin calibration over-covers.** At `n_calib = 25` coverage reaches 0.949.
   Validity is preserved; sharpness is not. The `n+1` correction forces the
   quantile to the maximum.
3. **Mondrian costs more than it earns on continuous heteroscedasticity.** Group
   proxies (`|x1|` terciles) are noisy within themselves, so per-group calibration
   adds variance. The `no_mondrian` ablation measures **+13.3%** width on
   `grouped_hetero` (confirming the component earns its keep there) and
   **−3.1%** on `heteroscedastic` (it costs width there).
4. **The distilled ACI component is near-inert on stationary data.** The
   `no_aci` ablation is **refuted** on both probe DGPs (+0.11%, −2.13%). This was
   predicted in advance: with exchangeable validation data the nominal alpha is
   already correct. The component is load-bearing only under drift, which is what
   G3 measures.
5. **G1 does not pass, and was not made to.** The flagship is 5.58% narrower
   against an 8.0% threshold. An earlier configuration scored 7.87% using the
   terminal value of the ACI trajectory instead of its tail median; the tail
   median was kept because it is the sound estimator. See
   [architecture.md §8](architecture.md#why-g1-does-not-pass).
6. **A spec conflict is unresolved.** G3's 1.25× width budget is unsatisfiable at
   the specified 3.0 drift multiplier (measured ratio 1.78×). G3 is evaluated at
   `drift_factor = 1.2`; the full tradeoff curve is published.
7. **Synthetic data only.** Every number here is about these six generators. Real
   data has dependencies the generators do not model.
8. **The drift experiment assumes labels arrive immediately** (a prequential
   setting) and that the drift is a clean multiplicative noise jump. Gradual and
   concept drift are out of scope.
9. **Runtime is 105.4 s**, over the 60 s target, dominated by MAPIE's internal
   refit. We declined to shrink one method's learner to hit the number.

## 6. Intended use

**Appropriate for:** comparing conformal interval methods; validating a conformal
implementation against a reference; teaching the failure modes; establishing
whether conditional-coverage machinery is worth its variance on *your* grouping.

**Not appropriate for:** deployment decisions on real data; anything where
covariate shift is likely and unguarded; as evidence of conditional validity.

## 7. Reproducing

```bash
pip install -e ".[backend,dev]"
python examples/run_demo.py           # writes results/benchmark.json
```

Seeded from one integer. Gate G5 verifies bit-identical reproduction across two
full runs. Environment recorded in every report: Python 3.13.14, NumPy 2.5.3,
scikit-learn 1.9.1, MAPIE 1.5.0.
