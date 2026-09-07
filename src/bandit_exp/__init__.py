"""bandit-exp: a harness for fairly comparing multi-armed bandit algorithms."""

from bandit_exp.algorithms import (
    BanditAlgorithm,
    EGreedy,
    Random,
    ThomsonSampling,
    UCB1,
    UCB1Tuned,
)
from bandit_exp.environments import (
    AbruptChangeBernoulli,
    DriftingBernoulli,
    Environment,
    GapBernoulli,
    StationaryBernoulli,
)

__all__ = [
    # algorithms
    "BanditAlgorithm",
    "EGreedy",
    "Random",
    "ThomsonSampling",
    "UCB1",
    "UCB1Tuned",
    # environments
    "Environment",
    "StationaryBernoulli",
    "GapBernoulli",
    "DriftingBernoulli",
    "AbruptChangeBernoulli",
]
