"""Tests for the benchmark environments.

Two flavours of test:

1. *Contract tests* — properties that every Environment must satisfy
   (rewards are 0/1, indices are validated, mean/optimal_mean agree, ...).
2. *Reproducibility tests* — with a fixed seed the reward stream is
   deterministic, which is what makes cross-algorithm comparison fair.
"""

from __future__ import annotations

import random

import pytest

from environments import (
    AbruptChangeBernoulli,
    DriftingBernoulli,
    Environment,
    GapBernoulli,
    StationaryBernoulli,
)


def make_all(rng: random.Random | None = None) -> list[Environment]:
    """One instance of every concrete environment, for parametric tests."""
    return [
        StationaryBernoulli([0.1, 0.5, 0.9], rng=rng),
        GapBernoulli(num=4, best=0.6, gap=0.2, rng=rng),
        DriftingBernoulli(
            start=[0.1, 0.9], end=[0.9, 0.1], horizon=100, rng=rng
        ),
        AbruptChangeBernoulli(
            before=[0.2, 0.8], after=[0.8, 0.2], change_at=50, rng=rng
        ),
    ]


# --------------------------------------------------------------------------
# Contract tests (apply to every environment)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("env", make_all(random.Random(0)))
def test_pull_returns_binary_reward(env: Environment):
    for t in range(200):
        for i in range(env.num):
            r = env.pull(i, t)
            assert r in (0, 1)
            assert isinstance(r, int)


@pytest.mark.parametrize("env", make_all())
def test_mean_in_unit_interval(env: Environment):
    for t in (0, 25, 50, 75, 100, 500):
        for i in range(env.num):
            assert 0.0 <= env.mean(i, t) <= 1.0


@pytest.mark.parametrize("env", make_all())
def test_optimal_mean_is_max_of_means(env: Environment):
    for t in (0, 10, 50, 99, 100, 200):
        means = [env.mean(i, t) for i in range(env.num)]
        assert env.optimal_mean(t) == pytest.approx(max(means))
        assert env.mean(env.optimal_arm(t), t) == pytest.approx(max(means))


@pytest.mark.parametrize("env", make_all())
def test_pull_rejects_out_of_range_arm(env: Environment):
    with pytest.raises(IndexError):
        env.pull(-1, 0)
    with pytest.raises(IndexError):
        env.pull(env.num, 0)


def test_empirical_ctr_converges_to_true_mean():
    """Pulling an arm many times gives a frequency near its true mean."""
    env = StationaryBernoulli([0.3], rng=random.Random(1))
    n = 50_000
    total = sum(env.pull(0, t) for t in range(n))
    assert total / n == pytest.approx(0.3, abs=0.01)


# --------------------------------------------------------------------------
# Reproducibility tests (the core of fair benchmarking)
# --------------------------------------------------------------------------


def test_same_seed_gives_same_reward_stream():
    a = StationaryBernoulli([0.1, 0.5, 0.9], rng=random.Random(42))
    b = StationaryBernoulli([0.1, 0.5, 0.9], rng=random.Random(42))
    stream_a = [a.pull(t % 3, t) for t in range(300)]
    stream_b = [b.pull(t % 3, t) for t in range(300)]
    assert stream_a == stream_b


def test_different_seed_gives_different_stream():
    a = StationaryBernoulli([0.5], rng=random.Random(0))
    b = StationaryBernoulli([0.5], rng=random.Random(1))
    stream_a = [a.pull(0, t) for t in range(300)]
    stream_b = [b.pull(0, t) for t in range(300)]
    assert stream_a != stream_b


# --------------------------------------------------------------------------
# Per-environment behaviour
# --------------------------------------------------------------------------


def test_gap_bernoulli_optimal_arm_and_gap():
    env = GapBernoulli(num=5, best=0.7, gap=0.25)
    assert env.optimal_arm(0) == 0
    assert env.mean(0, 0) == pytest.approx(0.7)
    for i in range(1, 5):
        assert env.mean(i, 0) == pytest.approx(0.45)
    assert env.optimal_mean(0) == pytest.approx(0.7)


def test_drifting_best_arm_switches_over_time():
    env = DriftingBernoulli(start=[0.1, 0.9], end=[0.9, 0.1], horizon=100)
    assert env.optimal_arm(0) == 1  # arm 1 starts best
    assert env.optimal_arm(100) == 0  # arm 0 ends best
    # midpoint: both equal
    assert env.mean(0, 50) == pytest.approx(env.mean(1, 50))


def test_drifting_clamps_time_beyond_horizon():
    env = DriftingBernoulli(start=[0.2], end=[0.8], horizon=10)
    assert env.mean(0, 10) == pytest.approx(0.8)
    assert env.mean(0, 1000) == pytest.approx(0.8)  # clamped at end


def test_abrupt_change_switches_at_change_point():
    env = AbruptChangeBernoulli(before=[0.2, 0.8], after=[0.8, 0.2], change_at=50)
    assert env.mean(0, 49) == pytest.approx(0.2)
    assert env.mean(0, 50) == pytest.approx(0.8)  # inclusive of change point
    assert env.optimal_arm(49) == 1
    assert env.optimal_arm(50) == 0


# --------------------------------------------------------------------------
# Validation tests
# --------------------------------------------------------------------------


def test_stationary_rejects_probability_out_of_range():
    with pytest.raises(ValueError):
        StationaryBernoulli([0.5, 1.5])
    with pytest.raises(ValueError):
        StationaryBernoulli([-0.1])


def test_gap_rejects_negative_resulting_probability():
    with pytest.raises(ValueError):
        GapBernoulli(num=3, best=0.1, gap=0.5)  # best - gap < 0


def test_drifting_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        DriftingBernoulli(start=[0.1, 0.2], end=[0.3], horizon=10)


def test_environment_rejects_zero_arms():
    with pytest.raises(ValueError):
        StationaryBernoulli([])
