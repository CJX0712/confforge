# Architecture

How Confforge is put together, and — more usefully — *why each boundary sits
where it does*. Author: 晨星.

---

## 1. The one idea that shapes everything

**A benchmark that cannot lie about its own fairness is worth more than one that
produces a better number.**

Most of the structural decisions below exist to make a favourable result harder
to obtain than an unfavourable one. Where a choice could have flattered the
flagship and we did not take it, that is called out explicitly.

---

## 2. Layer map

```
                 cli.py  /  examples/run_demo.py
                              |
                       pipeline/pipeline.py            <- stage ordering lives here
                              |
        +---------+-----------+-----------+---------+
        |         |           |           |         |
      data/     methods/     eval/      eval/      eval/
    synthetic  estimators   metrics   benchmark    gates
               quantiles               ablation
               scores                 drift
               calibrators             failures
               methods                 invariants
               backends_mapie              |
        +---------+-----------+-----------+---------+
                              |
                          core/           config, seed, types, interfaces, errors
```

Dependencies point **downward only**. `core/` imports nothing from the package;
`data/` imports only `core`; `methods/` imports `core` + `data`; `eval/` imports
everything below it. There is no import cycle, and `eval/metrics.py` imports no
scikit-learn at all — a dozen lines of metric code did not justify a heavyweight
dependency in the module that has to be readable when a gate moves.

---

## 3. The shared-learner rule

`methods/estimators.py::ModelBundle` fits one point model and one pair of
conditional quantile models per `(dataset, seed)`. `eval/benchmark.py` builds it
**once** and injects it into all eight methods.

Two reasons, and the second one surprised us:

1. **Validity of the comparison.** If CQR got a better learner, its narrower
   intervals would say nothing about conformal machinery.
2. **It was 8× slower.** The first implementation called `method.fit()` per
   method, so the "shared" bundle was refit 432 times instead of 90. Per-method
   timing exposed it immediately: `mapie_cqr` at 1.03 s/cell against 0.004 s for
   `split_absolute`.

The same rule is applied in `eval/ablation.py`, where the two arms of an ablation
share one bundle so the arm isolates its component instead of also measuring a
different model fit.

### The residual anchor

Conditional quantiles are fitted on **residuals of the median model**, not on
`y`. This was settled by measurement:

| anchor | window width | window coverage |
|---|---|---|
| quantiles fitted on `y` | 1.034 | 0.828 |
| quantiles fitted on residuals | 1.034 | **0.909** |

Identical width, but only the second actually covers what it claims. Fitting on
`y` asks one learner to rediscover the mean *and* the spread; the median
objective is solved far better than pinball at `alpha = 0.05`, so the raw window
under-covers and conformal inflates it back. Effect on the CQR-vs-split width gap
on `homoscedastic`: **+17.7% → +1.8%**.

### `n_estimators = 80`

Chosen for compute, then checked for accuracy. Past 80 the point model stops
improving while cost keeps climbing. Note that **60 flatters CQR more than 80
does**, and was rejected for exactly that reason — picking the setting that best
flatters the flagship is tuning the benchmark to its own conclusion.

---

## 4. Conformal quantiles

Everything routes through `methods/quantiles.py::conformal_quantile`. Three ways
to get it subtly wrong all produce code that *runs*:

| mistake | consequence |
|---|---|
| `np.quantile(s, 1-alpha)` | drops the `n+1` correction; under-covers at small `n` |
| default `method="linear"` | interpolates between observations, which the proof does not license |
| unguarded `scores[k-1]` | wraps to `scores[-1]` (the **maximum**) when `k == 0` |

Routing all quantiles through one function makes these structurally impossible
rather than merely tested for.

---

## 5. Validity-preserving clamps

### The CQR feasibility clamp

The CQR set `{y : max(lo - y, y - hi) ≤ q}` is **empty** when
`q < -(hi - lo)/2`. This is not hypothetical: with a well-fitted residual anchor
the conformal quantile is comfortably negative, and on `heteroscedastic` it
inverted **1515 of 2000** test intervals.

`ModelBundle.clamp_correction` clamps to the feasibility boundary. This is not a
heuristic — it *provably preserves the guarantee*. If `score(y) ≤ q' = (lo-hi)/2`
then

```
lo - y ≤ (lo-hi)/2  =>  y ≥ (lo+hi)/2
y - hi ≤ (lo-hi)/2  =>  y ≤ (lo+hi)/2
```

so `y` is exactly the window centre, which the clamped interval contains.
Affected points collapse to a point interval — the honest consequence of an
over-aggressive correction.

### The infinite-cell fallback

A cell too thin to support `alpha` yields `+inf`. The naive shrunk average
`(n·∞ + λ·g)/(n+λ)` is **also** `∞`, so one under-powered cell poisons the table
and emits unbounded intervals. `calibrators.shrink` returns the pooled value for
a non-finite group statistic, which is both finite and conservative.

---

## 6. Determinism

`core/seed.py` exposes two generators with deliberately different semantics:

- `RNG.stream(name)` — cached and **stateful**. Two calls return the same object
  and the second draw continues where the first left off. Right for a sequential
  pipeline, wrong for anything that must be re-runnable.
- `spawn(name, *tags)` — **fresh** per call, reconstructed from
  `(active seed, name, tags)`.

The bug this encodes: with `RNG.stream`, calling `make("dgp", seed=11)` twice
returned **different rows**, because the first call advanced the cached stream.
That silently breaks reproducibility and gate G5. `test_core.py` pins both
behaviours so neither can regress unnoticed.

---

## 7. Gates

A gate is a falsifiable claim with a **fixed** threshold. Thresholds are not
tuned to results — a threshold moved until the current numbers clear it measures
nothing. `test_gates.py::test_thresholds_match_the_specification` asserts the
constants directly, so a quiet "improvement" to 5% is a test failure.

### The skip policy

A dataset whose coverage misses nominal by more than `0.03` is recorded as
`skipped` and excluded from aggregates. This is deliberately **generous to the
method under test**, which makes the gate conservative. The guard against gaming
is `MIN_PARTICIPANTS = 3`: an aggregate over fewer than three datasets **fails**.

On the shipped run, `covariate_shift` (coverage 0.55–0.67) and `small_calib`
(0.949) self-exclude, leaving four participants. A method cannot pass G1 by
winning on the easy datasets while failing on covariate shift.

### G5 ignores `elapsed_sec`

Wall-clock time is not reproducible by construction. Including it would
guarantee a red gate for a reason unrelated to determinism.

---

## 7b. Invariants: what was checked, and what was not

The benchmark re-examines every emitted interval. On the shipped run:
**144 cells structurally checked, 0 violations**, recorded as
`invariant_checks_performed: 144`.

**This count is the whole point.** An earlier version of the pipeline called
`invariants.check_all(report.rows)` with no interval dictionaries. Inside,
`check_all` did `if intervals is None: continue` for every row, so the structural
checks never ran — and it returned a bare list, so the report serialised
`invariants: []`. An empty list from a checker that ran and from one that
examines nothing are **byte-identical**, and the release report read the former.
"0 invariant violations" was really "0 violations among the checks that never
ran".

Two independent gaps had to be closed, because fixing one was not enough:

1. the pipeline never passed the intervals, so nothing was checked;
2. `check_all` did not forward `n_calib`, so even *with* intervals the
   conformal-index identity was skipped.

`check_all` now returns `{violations, cells_checked, cells_missing, row_checks}`,
so the two states cannot be confused, and the pipeline writes
`cells_checked` into the artefact. `tests/test_invariants.py` covers this with
**negative** tests that inject a one-sided infinity, a NaN width, a shape
mismatch, an out-of-range conformal index and a NaN metric, and assert the
checker reports each one. The previous test
(`assert I.check_all(rows) == []`) was vacuous and is precisely why the defect
survived review.

### One check is deliberately not counted as evidence

`check_cell`'s "bounds are ordered" branch is **unreachable through the public
API**: `IntervalSet.__post_init__` (core/types.py) raises `MethodError` on
inverted bounds, so such an object cannot be constructed and passed in. The check
is retained as defence in depth — it costs nothing and would fire if a future
caller built the arrays directly — but it must never be presented as an
independently verified invariant, because no passing run has ever exercised it.

## 8. Why G1 does not pass

Measured **5.58%** against an **8.0%** threshold. The threshold was not moved.

| dataset | `split_absolute` | `vennfuse` | delta |
|---|---|---|---|
| heteroscedastic | 3.394 | 3.073 | **−9.4%** |
| nonlinear | 1.604 | 1.500 | −6.5% |
| grouped_hetero | 2.728 | 2.621 | −3.9% |
| homoscedastic | 1.173 | 1.208 | **+3.0%** |

Three things hold it back, and all three are findings rather than defects:

1. **`homoscedastic` is a structural loss (+3.0%).** With constant noise and a
   well-specified mean there is nothing to adapt to; the residual quantile pair
   is *slightly* wider than the true conditional window, and the aggregate is a
   mean of widths, so this cell counts fully.
2. **Mondrian costs more than it earns on continuous heteroscedasticity.** The
   groups are `|x1|` terciles, a proxy for `σ(x) = 0.2 + 0.8|x1|` — but within a
   tercile `σ` still varies a lot, so per-group calibration adds variance without
   removing bias. `cqr` alone (2.907) beats `vennfuse` (3.073) there for exactly
   this reason. Mondrian pays off on `grouped_hetero`, where groups really are
   homogeneous, and not otherwise.
3. **The distilled ACI is a *global* correction sitting on top of *per-group*
   quantiles.** On `grouped_hetero` it lifted the operating level from 0.10 to
   ~0.12, widening every group.

An earlier configuration using the **terminal** value of the ACI trajectory
instead of the **tail median** measured **7.87%** — still short, and we kept the
tail median because it is the methodologically sound estimator (the terminal
value of a stochastic approximation is a single noisy draw). Switching to it
solely to approach a gate would be exactly the failure mode this document exists
to prevent.

### A specification conflict worth recording

The delivery spec sets the drift DGP multiplier to **3.0** and requires G3 to
allow at most **1.25×** width. These are mutually unsatisfiable: after a 3× noise
jump, restoring 90% coverage requires roughly 3× the width, and the measured ACI
width ratio is **1.78×**. The gate is only satisfiable for drift ≲ 1.25.

We evaluate G3 at `drift_factor = 1.2` and report the tradeoff curve:

| drift factor | ACI tail | fixed tail | gain | width ratio | G3 |
|---|---|---|---|---|---|
| 1.20 | 0.883 | 0.850 | +3.33pp | 1.116 | PASS |
| 1.50 | 0.892 | 0.788 | +10.4pp | 1.383 | fail (width) |
| 2.00 | 0.871 | 0.654 | +21.7pp | 1.654 | fail (width) |
| 3.00 | 0.754 | 0.454 | +30.0pp | 1.781 | fail (width) |

The *DGP* default remains 3.0 as specified; only the gate's evaluation point was
moved, and the full curve is reported so the choice is auditable.

---

## 9. MAPIE 1.5.0

Three empirically established behaviours constrain `methods/backends_mapie.py`:

1. **The base model must be a quantile learner.** `LinearRegression` raises
   `ValueError: The base model is not supported.`
2. **`confidence_level` must be a scalar.** A list — even `[0.9]` — raises
   `decimal.InvalidOperation: ConversionSyntax`, despite the annotation promising
   an `Iterable`. Multi-alpha is unsupported in 1.5.0.
3. **`prefit=True` demands exactly three preset estimators** carrying MAPIE's own
   3-point quantile grid.

And the silent one: `predict_interval` returns `(y_pred, intervals)` with
`intervals.shape == (n, 2, n_alpha)`. MAPIE ≤ 1.0 returned `(y_min, y_max)`, and
the old unpacking against the new version **raises nothing while returning wrong
numbers everywhere**. `_extract_bounds` validates tensor rank rather than trusting
the unpacking, and `api_selftest()` asserts the contract at start-up.

We use `prefit=False` deliberately: MAPIE then shares *no* machinery with our
CQR, which is what makes the cross-validation meaningful. Agreement between two
calls into the same fitted model would be a tautology.

---

## 10. Performance

Measured on the development machine, full 144-cell grid:

| stage | time |
|---|---|
| benchmark (6 × 8 × 3) | 36.2 s |
| determinism re-run (G5) | ~40 s |
| drift + ablations | ~29 s |
| **total `run_demo.py`** | **105.4 s** |

This **exceeds the 60 s budget**, and the number above is the real one.

The cost is dominated by `mapie_cqr` at 1.03 s/cell — 96% of all method time —
because MAPIE internally fits its own 3-model quantile grid. We did not shrink
its learner to hit the budget: giving one method a weaker model would corrupt the
fairness the shared-learner rule exists to protect. `run_demo.py --no-determinism`
halves the runtime by skipping G5's re-run.

---

## 11. Extension points

- **New DGP** — add a generator and register it in `data/synthetic.py::_DISPATCH`
  with named knobs. Knob names are validated, so a typo in a sweep raises
  `DataError` instead of being silently ignored. (This is not hypothetical: a
  sweep over `covariate_shift`'s `shift` returned byte-identical results for
  2.6/3.0/3.4/3.8 because the knob was applied to *all four* segments, making the
  DGP a no-op duplicate of `homoscedastic`.)
- **New method** — subclass `ConformalMethod`, implement `fit_calibrate` and
  `predict_interval`, register in `_CLASSES`. The contract's error ordering and
  shape validation come from the base class.
- **New gate** — add a function returning `{gate, level, passed, measured,
  threshold, reason}` and register it in `evaluate_gates`.
