"""Tests for the bandit algorithms.

Contract tests (properties every algorithm must satisfy), reproducibility
tests (fixed seed -> identical behaviour), and a couple of sanity checks
that a learning algorithm actually beats the random baseline on an easy
problem.
"""

from __future__ import annotations

import random

import pytest

from algorithms import (
    BanditAlgorithm,
    EGreedy,
    Random,
    ThomsonSampling,
    UCB1,
    UCB1Tuned,
)
from environments import GapBernoulli, StationaryBernoulli


def make_all(num: int, rng: random.Random | None = None) -> list[BanditAlgorithm]:
    return [
        Random(num, rng=rng),
        EGreedy(num, 0.1, rng=rng),
        ThomsonSampling(num, rng=rng),
        UCB1(num, rng=rng),
        UCB1Tuned(num, rng=rng),
    ]


# --------------------------------------------------------------------------
# Contract tests
# --------------------------------------------------------------------------


@pytest.mark.parametrize("alg", make_all(4, random.Random(0)))
def test_choose_returns_valid_arm(alg: BanditAlgorithm):
    env = StationaryBernoulli([0.2, 0.4, 0.6, 0.8], rng=random.Random(0))
    for t in range(500):
        i = alg.display()
        assert 0 <= i < alg.num
        alg.reward(i, bool(env.pull(i, t)))


@pytest.mark.parametrize("alg", make_all(3, random.Random(0)))
def test_stats_stay_consistent(alg: BanditAlgorithm):
    env = StationaryBernoulli([0.3, 0.5, 0.7], rng=random.Random(1))
    for t in range(300):
        i = alg.display()
        alg.reward(i, bool(env.pull(i, t)))
    assert sum(alg.displayed) == 300
    for i in range(alg.num):
        assert 0 <= alg.clicked[i] <= alg.displayed[i]
    assert 0.0 <= alg.ctr_overall() <= 1.0


def test_ctr_overall_zero_before_any_display():
    assert Random(3).ctr_overall() == 0.0


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda rng: Random(3, rng=rng),
        lambda rng: EGreedy(3, 0.1, rng=rng),
        lambda rng: ThomsonSampling(3, rng=rng),
        lambda rng: UCB1(3, rng=rng),
        lambda rng: UCB1Tuned(3, rng=rng),
    ],
)
def test_same_seed_gives_same_choices(build):
    def run(seed: int) -> list[int]:
        env = StationaryBernoulli([0.3, 0.5, 0.7], rng=random.Random(seed))
        alg = build(random.Random(seed + 1))
        choices = []
        for t in range(200):
            i = alg.display()
            alg.reward(i, bool(env.pull(i, t)))
            choices.append(i)
        return choices

    assert run(7) == run(7)


# --------------------------------------------------------------------------
# Sanity: learners beat the random baseline on an easy problem
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "build",
    [
        lambda rng: EGreedy(5, 0.05, rng=rng),
        lambda rng: ThomsonSampling(5, rng=rng),
        lambda rng: UCB1(5, rng=rng),
        lambda rng: UCB1Tuned(5, rng=rng),
    ],
)
def test_learner_beats_random_baseline(build):
    horizon = 3000

    def final_ctr(build_fn, seed: int) -> float:
        env = GapBernoulli(num=5, best=0.6, gap=0.3, rng=random.Random(seed))
        alg = build_fn(random.Random(seed + 1))
        for t in range(horizon):
            i = alg.display()
            alg.reward(i, bool(env.pull(i, t)))
        return alg.ctr_overall()

    seeds = range(5)
    learner = sum(final_ctr(build, s) for s in seeds) / len(seeds)
    baseline = sum(final_ctr(lambda rng: Random(5, rng=rng), s) for s in seeds) / len(
        seeds
    )
    assert learner > baseline


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def test_egreedy_rejects_bad_epsilon():
    with pytest.raises(ValueError):
        EGreedy(3, 1.5)


def test_thomson_rejects_nonpositive_params():
    with pytest.raises(ValueError):
        ThomsonSampling(3, alpha=0.0)


def test_algorithm_rejects_zero_arms():
    with pytest.raises(ValueError):
        Random(0)
