"""Typed configuration objects with environment-variable overrides.

Every knob can be set through an ``CONFFORGE_*`` environment variable, e.g.
``CONFFORGE_ALPHA=0.05``. Values are validated eagerly so misconfiguration
surfaces as a :class:`~core.errors.ConfigError` instead of a silently wrong
benchmark.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from typing import Any

from core.errors import ConfigError

__all__ = ["ENV_PREFIX", "ConfforgeConfig", "load_config"]

ENV_PREFIX = "CONFFORGE_"


def _coerce(raw: str, target: Any) -> Any:
    """Convert an environment string to the annotated type."""
    if target is bool:
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"cannot parse boolean from {raw!r}")
    if target is int:
        try:
            return int(raw)
        except ValueError as exc:
            raise ConfigError(f"cannot parse int from {raw!r}") from exc
    if target is float:
        try:
            return float(raw)
        except ValueError as exc:
            raise ConfigError(f"cannot parse float from {raw!r}") from exc
    if target is str:
        return raw
    return raw


@dataclass(frozen=True)
class ConfforgeConfig:
    """Immutable run configuration.

    Attributes:
        seed: Master seed forwarded to :func:`core.seed.set_all`.
        alpha: Miscoverage level, i.e. the target coverage is ``1 - alpha``.
        n_train: Number of training samples per synthetic dataset.
        n_calib: Number of calibration samples (the conformal calibration set).
        n_val: Number of validation samples used for model/hyper-parameter choice.
        n_test: Number of hold-out test samples used for every reported metric.
        seeds: Seeds used for the multi-seed benchmark.
        n_jobs: Parallelism handed to scikit-learn; ``1`` keeps results
            bit-identical across platforms and is the default.
        coverage_grid: Nominal coverage levels for the efficiency diagram.
        drift_steps: Length of the online stream used by the drift scenario.
        drift_gamma: Step size of the ACI (adaptive conformal inference) update.
        use_backend: Whether the optional MAPIE backend may be used.
    """

    seed: int = 20261004
    alpha: float = 0.1
    n_train: int = 1500
    n_calib: int = 500
    n_val: int = 500
    n_test: int = 2000
    seeds: tuple[int, ...] = (11, 23, 37)
    n_jobs: int = 1
    coverage_grid: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    drift_steps: int = 400
    drift_gamma: float = 0.01
    use_backend: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ConfigError(f"alpha must lie in (0, 1), got {self.alpha}")
        for name in ("n_train", "n_calib", "n_val", "n_test"):
            value = getattr(self, name)
            if value <= 0:
                raise ConfigError(f"{name} must be positive, got {value}")
        if self.n_calib < 19:
            # The finite-sample correction uses ceil((n+1)(1-alpha)); with a
            # tiny calibration set the quantile degenerates to the maximum,
            # which is legal but makes every method identical.
            raise ConfigError("n_calib must be >= 19 for a meaningful conformal quantile")
        if len(self.seeds) < 3:
            raise ConfigError("at least 3 seeds are required for mean+/-std reporting")
        if len(set(self.seeds)) != len(self.seeds):
            raise ConfigError(f"seeds must be distinct, got {self.seeds}")
        if self.drift_gamma <= 0:
            raise ConfigError("drift_gamma must be positive")
        if self.drift_steps < 50:
            raise ConfigError("drift_steps must be >= 50 for the ACI update to settle")
        for level in self.coverage_grid:
            if not 0.0 < level < 1.0:
                raise ConfigError(f"coverage levels must lie in (0, 1), got {level}")
        if tuple(sorted(self.coverage_grid)) != self.coverage_grid:
            raise ConfigError("coverage_grid must be sorted ascending")
        if self.seed < 0:
            raise ConfigError("seed must be non-negative")

    @property
    def coverage(self) -> float:
        """Target coverage ``1 - alpha``."""
        return 1.0 - self.alpha

    def with_env(self, env: dict[str, str] | None = None) -> ConfforgeConfig:
        """Return a copy with ``CONFFORGE_*`` overrides applied.

        Raises:
            ConfigError: If an override cannot be parsed into the field type.
        """
        source = os.environ if env is None else env
        overrides: dict[str, Any] = {}
        for f in fields(self):
            key = ENV_PREFIX + f.name.upper()
            if key not in source:
                continue
            raw = source[key]
            target = f.type
            # ``from __future__ import annotations`` makes f.type a string.
            if isinstance(target, str):
                target = {"int": int, "float": float, "bool": bool, "str": str}.get(
                    target.replace(" ", ""), str
                )
            overrides[f.name] = _coerce(raw, target)
        if "seeds" in overrides:
            raw = overrides.pop("seeds")
            overrides["seeds"] = tuple(int(x) for x in raw.replace(" ", "").split(",") if x)
        if "coverage_grid" in overrides:
            raw = overrides.pop("coverage_grid")
            overrides["coverage_grid"] = tuple(
                float(x) for x in raw.replace(" ", "").split(",") if x
            )
        return replace(self, **overrides) if overrides else self


def replace(config: ConfforgeConfig, **changes: Any) -> ConfforgeConfig:
    """``dataclasses.replace`` re-exported so callers need not import dataclasses."""
    from dataclasses import replace as _replace

    return _replace(config, **changes)


def load_config(**overrides: Any) -> ConfforgeConfig:
    """Build a config from environment overrides plus explicit keyword overrides."""
    base = ConfforgeConfig()
    return base.with_env().with_env({}) if not overrides else _merge(base, overrides)


def _merge(base: ConfforgeConfig, overrides: dict[str, Any]) -> ConfforgeConfig:
    from dataclasses import replace as _replace

    known = {f.name for f in fields(base)}
    unknown = set(overrides) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    return _replace(base, **overrides)


# ``field`` is imported for dataclass consumers that build derived configs.
_ = field
