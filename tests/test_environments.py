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

from bandit_exp.environments import (
    AbruptChangeBernoulli,
    DriftingBernoulli,
    Environment,
    GapBernoulli,
    SinusoidalNoiseBernoulli,
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
        SinusoidalNoiseBernoulli(
            base=[0.3, 0.5], amplitude=0.05, theta=[0.01, 0.02], rng=rng
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


def test_sinusoidal_oscillates_around_base():
    import math

    base = [0.3, 0.5]
    amp = 0.05
    theta = [0.01, 0.02]
    env = SinusoidalNoiseBernoulli(base=base, amplitude=amp, theta=theta)
    # At t=0, sin(0)=0, so mean equals the base.
    assert env.mean(0, 0) == pytest.approx(0.3)
    assert env.mean(1, 0) == pytest.approx(0.5)
    # At a later time, mean = base + amp*sin(theta*t), and stays within +/-amp.
    for t in (1, 7, 50, 123, 999):
        for i in range(2):
            expected = base[i] + amp * math.sin(theta[i] * t)
            assert env.mean(i, t) == pytest.approx(expected)
            assert base[i] - amp - 1e-9 <= env.mean(i, t) <= base[i] + amp + 1e-9


def test_sinusoidal_clips_to_unit_interval():
    # Large amplitude would push below 0 / above 1; result must be clipped.
    env = SinusoidalNoiseBernoulli(base=[0.02, 0.99], amplitude=0.5, theta=[1.0, 1.0])
    for t in range(200):
        for i in range(2):
            assert 0.0 <= env.mean(i, t) <= 1.0


def test_sinusoidal_scalar_amplitude_broadcasts():
    env = SinusoidalNoiseBernoulli(base=[0.3, 0.4, 0.5], amplitude=0.01, theta=[0.1, 0.2, 0.3])
    assert env.amplitude == [0.01, 0.01, 0.01]


def test_sinusoidal_video_top_arms_stay_close():
    # The video_top_noisy configuration: four near-tied arms, small wobble.
    env = SinusoidalNoiseBernoulli(
        base=[0.1115, 0.1161, 0.1195, 0.1197],
        amplitude=0.009,
        theta=[0.011, 0.017, 0.023, 0.031],
    )
    for t in range(0, 2000, 50):
        means = [env.mean(i, t) for i in range(env.num)]
        assert max(means) - min(means) < 0.05  # arms remain a close race


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


def test_sinusoidal_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        SinusoidalNoiseBernoulli(base=[0.3, 0.5], amplitude=[0.01], theta=[0.1, 0.2])
    with pytest.raises(ValueError):
        SinusoidalNoiseBernoulli(base=[0.3, 0.5], amplitude=0.01, theta=[0.1])


def test_environment_rejects_zero_arms():
    with pytest.raises(ValueError):
        StationaryBernoulli([])
