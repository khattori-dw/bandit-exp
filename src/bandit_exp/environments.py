"""Benchmark problem settings (environments) for bandit experiments.

All environments here are *Bernoulli* bandits: pulling an arm yields a
0/1 reward. What differs between environments is *how the click
probability of each arm is determined* over time (stationary, drifting,
abrupt change, ...).

Design notes
------------
- A random generator (``random.Random``) is *injected* into every
  environment. This is the core of fair benchmarking: by sharing a
  seeded generator (or one generator per algorithm derived from the same
  seed), every algorithm can be evaluated against a reproducible reward
  stream. We use the standard-library ``random`` module so the
  environments have no third-party dependencies.
- Each environment exposes the *true* mean of every arm via ``mean`` and
  the best achievable mean via ``optimal_mean``. These are not needed to
  run an algorithm, but they let us compute regret later and make the
  environments easy to test.
"""

from __future__ import annotations

import abc
import math
import random


class Environment(abc.ABC):
    """Abstract base class for a Bernoulli bandit problem setting.

    Parameters
    ----------
    num:
        Number of arms.
    rng:
        Random generator used to draw rewards. Injecting it keeps the
        environment reproducible and lets several algorithms be compared
        on identical reward streams. Defaults to a fresh ``random.Random``.
    """

    def __init__(self, num: int, rng: random.Random | None = None):
        if num < 1:
            raise ValueError("num must be >= 1")
        self.num = num
        self.rng = rng if rng is not None else random.Random()

    @abc.abstractmethod
    def mean(self, i: int, t: int) -> float:
        """Return the true click probability of arm ``i`` at time ``t``."""
        raise NotImplementedError

    def pull(self, i: int, t: int) -> int:
        """Pull arm ``i`` at time ``t`` and return a 0/1 reward."""
        if not 0 <= i < self.num:
            raise IndexError(f"arm index {i} out of range [0, {self.num})")
        p = self.mean(i, t)
        return int(self.rng.random() < p)

    def optimal_mean(self, t: int) -> float:
        """Return the best achievable click probability at time ``t``."""
        return max(self.mean(i, t) for i in range(self.num))

    def optimal_arm(self, t: int) -> int:
        """Return the index of the best arm at time ``t`` (ties -> lowest)."""
        return max(range(self.num), key=lambda i: self.mean(i, t))


class StationaryBernoulli(Environment):
    """Each arm has a fixed click probability (the classic setting).

    This is the direct successor of the original ``Arms`` class.
    """

    def __init__(self, ps: list[float], rng: random.Random | None = None):
        super().__init__(len(ps), rng)
        for p in ps:
            if not 0.0 <= p <= 1.0:
                raise ValueError("every probability must be in [0, 1]")
        self.ps = list(ps)

    def mean(self, i: int, t: int) -> float:
        return self.ps[i]


class GapBernoulli(Environment):
    """Stationary environment built from an explicit difficulty gap.

    The optimal arm has probability ``best`` and every other arm has
    ``best - gap``. A small ``gap`` yields a hard problem (arms are hard
    to tell apart); a large ``gap`` yields an easy one. This makes it
    convenient to compare algorithms across controlled difficulty levels.
    """

    def __init__(
        self,
        num: int,
        best: float,
        gap: float,
        rng: random.Random | None = None,
    ):
        super().__init__(num, rng)
        if not 0.0 <= best <= 1.0:
            raise ValueError("best must be in [0, 1]")
        if gap < 0.0:
            raise ValueError("gap must be >= 0")
        suboptimal = best - gap
        if suboptimal < 0.0:
            raise ValueError("best - gap must be >= 0")
        self.best = best
        self.gap = gap
        # Arm 0 is the optimal one; the rest are sub-optimal.
        self.ps = [best] + [suboptimal] * (num - 1)

    def mean(self, i: int, t: int) -> float:
        return self.ps[i]


class DriftingBernoulli(Environment):
    """Non-stationary: every arm's probability drifts linearly over time.

    At time ``t`` arm ``i`` has probability
    ``clip(start[i] + slope[i] * t / horizon, 0, 1)``. The identity of the
    best arm can change during the run, which stresses algorithms that
    assume stationarity.
    """

    def __init__(
        self,
        start: list[float],
        end: list[float],
        horizon: int,
        rng: random.Random | None = None,
    ):
        if len(start) != len(end):
            raise ValueError("start and end must have the same length")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        super().__init__(len(start), rng)
        for p in (*start, *end):
            if not 0.0 <= p <= 1.0:
                raise ValueError("every probability must be in [0, 1]")
        self.start = list(start)
        self.end = list(end)
        self.horizon = horizon

    def mean(self, i: int, t: int) -> float:
        frac = min(max(t / self.horizon, 0.0), 1.0)
        p = self.start[i] + (self.end[i] - self.start[i]) * frac
        return min(max(p, 0.0), 1.0)


class AbruptChangeBernoulli(Environment):
    """Non-stationary: probabilities switch abruptly at a change point.

    Before ``change_at`` arm ``i`` uses ``before[i]``; from ``change_at``
    onward it uses ``after[i]``. Models a sudden regime change (e.g. a new
    trend appearing).
    """

    def __init__(
        self,
        before: list[float],
        after: list[float],
        change_at: int,
        rng: random.Random | None = None,
    ):
        if len(before) != len(after):
            raise ValueError("before and after must have the same length")
        if change_at < 0:
            raise ValueError("change_at must be >= 0")
        super().__init__(len(before), rng)
        for p in (*before, *after):
            if not 0.0 <= p <= 1.0:
                raise ValueError("every probability must be in [0, 1]")
        self.before = list(before)
        self.after = list(after)
        self.change_at = change_at

    def mean(self, i: int, t: int) -> float:
        return self.after[i] if t >= self.change_at else self.before[i]


class SinusoidalNoiseBernoulli(Environment):
    """Non-stationary: each arm oscillates around a fixed base probability.

    At time ``t`` arm ``i`` has probability

        clip(base[i] + amplitude[i] * sin(theta[i] * t), 0, 1)

    i.e. a deterministic (not random) periodic perturbation on top of a
    stationary base. With a small amplitude this models mild, structured
    fluctuation over time (e.g. daily/periodic effects on a page's CTR)
    while keeping the arms close together.
    """

    def __init__(
        self,
        base: list[float],
        amplitude: list[float] | float,
        theta: list[float],
        rng: random.Random | None = None,
    ):
        super().__init__(len(base), rng)
        if isinstance(amplitude, (int, float)):
            amplitude = [float(amplitude)] * len(base)
        if not (len(base) == len(amplitude) == len(theta)):
            raise ValueError("base, amplitude and theta must have the same length")
        for b in base:
            if not 0.0 <= b <= 1.0:
                raise ValueError("every base probability must be in [0, 1]")
        for a in amplitude:
            if a < 0.0:
                raise ValueError("amplitude must be >= 0")
        self.base = list(base)
        self.amplitude = list(amplitude)
        self.theta = list(theta)

    def mean(self, i: int, t: int) -> float:
        p = self.base[i] + self.amplitude[i] * math.sin(self.theta[i] * t)
        return min(max(p, 0.0), 1.0)
