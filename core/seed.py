"""Global determinism helpers.

``set_all`` is the single seed entry point for the whole package: numpy's global
legacy state, Python's :mod:`random`, and (optionally) every third-party backend
that exposes a ``seed``/``random_state`` knob are configured from one integer.
"""

from __future__ import annotations

import os
import random
from typing import Any, ClassVar

import numpy as np

__all__ = ["RNG", "get_seed", "seed_everything", "set_all", "spawn"]

_ENV_VAR = "CONFFORGE_SEED"
_ACTIVE_SEED: int | None = None


class RNG:
    """Named, isolated numpy generators.

    Using distinct generators per component prevents a change in one part of the
    pipeline from shifting the random stream of another part, which is the usual
    cause of irreproducible benchmarks.

    .. warning::
       :meth:`stream` hands out a **stateful, shared** generator. Two calls to
       ``RNG.stream("a")`` return the *same* object, and the second draw
       continues where the first left off. That is what you want inside a
       sequential pipeline, and exactly what you do *not* want when a component
       must be re-runnable in isolation -- e.g. regenerating a dataset with the
       same seed, which would otherwise return *different* rows and silently
       break bit-for-bit reproducibility. Use :func:`spawn` for that case.
    """

    _streams: ClassVar[dict[str, np.random.Generator]] = {}

    @classmethod
    def stream(cls, name: str) -> np.random.Generator:
        """Return (creating on first use) the generator registered under ``name``."""
        if name not in cls._streams:
            base = get_seed()
            # SeedSequence spawns independent, statistically sound streams.
            seq = np.random.SeedSequence([base, _stable_hash(name)])
            cls._streams[name] = np.random.Generator(np.random.PCG64(seq))
        return cls._streams[name]

    @classmethod
    def reset(cls) -> None:
        """Drop all registered streams so the next call re-seeds from scratch."""
        cls._streams.clear()


def spawn(name: str, *tags: int | str) -> np.random.Generator:
    """Return a **fresh** generator for ``name``, safe to call repeatedly.

    Unlike :meth:`RNG.stream`, the returned generator is not shared and not
    cached: it is reconstructed from ``(active seed, name, tags)`` on every call,
    so ``spawn("dgp.test", 7)`` yields the identical sequence however many times
    it is called and in whatever order. This is the correct entry point for data
    generation, where "same inputs -> same outputs" is a contract rather than a
    convenience.

    Args:
        name: Component name, e.g. ``"data.heteroscedastic.test"``.
        *tags: Extra integer/string discriminators, e.g. the per-run seed. They
            are folded into the entropy so one component can produce independent
            streams per seed without name collisions.

    Returns:
        A new :class:`numpy.random.Generator`.
    """
    entropy: list[int] = [get_seed(), _stable_hash(name)]
    for tag in tags:
        if isinstance(tag, str):
            entropy.append(_stable_hash(tag))
        else:
            entropy.append(int(tag))
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def _stable_hash(text: str) -> int:
    """A process-independent, PYTHONHASHSEED-independent string hash (FNV-1a)."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
    return h


def set_all(seed: int) -> int:
    """Seed every random source used by Confforge.

    Args:
        seed: Non-negative integer seed.

    Returns:
        The seed that was applied.

    Raises:
        ConfigError: If ``seed`` is negative.
    """
    global _ACTIVE_SEED

    if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        from core.errors import ConfigError

        raise ConfigError(f"seed must be a non-negative integer, got {seed!r}")

    seed = int(seed)
    _ACTIVE_SEED = seed

    os.environ[_ENV_VAR] = str(seed)
    # Legacy global state: needed because scikit-learn internals still call
    # np.random.seed / np.random.* in a few code paths.
    np.random.seed(seed % (2**32))  # noqa: NPY002 - legacy global state, on purpose
    random.seed(seed)
    RNG.reset()
    _seed_optional_backends(seed)
    return seed


def _seed_optional_backends(seed: int) -> None:
    """Best-effort seeding of optional backends; never raises."""
    try:  # pragma: no cover - optional dependency
        import mapie

        if hasattr(mapie, "set_seed"):
            mapie.set_seed(seed)
    except Exception:
        pass


def seed_everything(seed: int | None = None) -> int:
    """Like :func:`set_all` but falls back to ``CONFFORGE_SEED`` then ``0``."""
    if seed is None:
        seed = int(os.environ.get(_ENV_VAR, "0"))
    return set_all(seed)


def get_seed() -> int:
    """Return the active seed, defaulting to ``0`` before :func:`set_all`."""
    if _ACTIVE_SEED is not None:
        return _ACTIVE_SEED
    return int(os.environ.get(_ENV_VAR, "0"))


def as_generator(seed: int | None = None, name: str = "default") -> Any:
    """Convenience: return a fresh ``np.random.Generator`` for ``name``."""
    if seed is not None:
        set_all(seed)
    return RNG.stream(name)
