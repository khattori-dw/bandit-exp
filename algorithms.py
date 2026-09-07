"""Bandit algorithms, decoupled from any particular problem setting.

Every algorithm only ever sees which arm it chose and the 0/1 reward it
received; it never inspects the environment's true means. This keeps the
comparison fair: an algorithm cannot "cheat" by peeking at the problem.

Like the environments, algorithms take an injected ``random.Random`` so a
run can be made fully reproducible. No third-party dependencies are used
(``ThomsonSampling`` draws Beta samples via ``random.betavariate``).
"""

from __future__ import annotations

import abc
import math
import random


class BanditAlgorithm(abc.ABC):
    """Base class tracking per-arm display/click statistics."""

    def __init__(self, num: int, rng: random.Random | None = None):
        if num < 1:
            raise ValueError("num must be >= 1")
        self.num = num
        self.rng = rng if rng is not None else random.Random()
        self.displayed = [0] * num
        self.clicked = [0] * num
        self.ctr = [0.0] * num

    @abc.abstractmethod
    def choose(self) -> int:
        """Return the index of the arm to display next."""
        raise NotImplementedError

    def display(self) -> int:
        """Choose an arm, record that it was displayed, and return it."""
        i = self.choose()
        self.displayed[i] += 1
        self.ctr[i] = self.clicked[i] / self.displayed[i]
        return i

    def reward(self, i: int, clicked: bool) -> None:
        """Record the 0/1 reward received for arm ``i``."""
        if clicked:
            self.clicked[i] += 1
            self.ctr[i] = self.clicked[i] / self.displayed[i]

    def ctr_overall(self) -> float:
        sum_displayed = sum(self.displayed)
        if sum_displayed == 0:
            return 0.0
        return sum(self.clicked) / sum_displayed


class Random(BanditAlgorithm):
    """Uniformly random arm selection (a baseline / lower bound)."""

    def choose(self) -> int:
        return self.rng.randrange(self.num)


class EGreedy(BanditAlgorithm):
    """Epsilon-greedy: exploit the best empirical arm, explore otherwise."""

    def __init__(self, num: int, epsilon: float, rng: random.Random | None = None):
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1]")
        super().__init__(num, rng)
        self.epsilon = epsilon

    def choose(self) -> int:
        if self.rng.random() < self.epsilon:  # Explore
            return self.rng.randrange(self.num)
        # Exploit: pick (randomly among) the arms with the highest CTR.
        maxctr = max(self.ctr)
        maxarms = [i for i in range(self.num) if self.ctr[i] >= maxctr]
        return self.rng.choice(maxarms)


class ThomsonSampling(BanditAlgorithm):
    """Thompson sampling with a Beta prior (stdlib Beta sampling)."""

    def __init__(
        self,
        num: int,
        alpha: float = 1.0,
        beta: float = 1.0,
        rng: random.Random | None = None,
    ):
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be > 0")
        super().__init__(num, rng)
        self.alpha = alpha
        self.beta = beta

    def choose(self) -> int:
        best_theta = -1.0
        best_arm = 0
        for i in range(self.num):
            a = self.alpha + self.clicked[i]
            b = self.beta + self.displayed[i] - self.clicked[i]
            theta = self.rng.betavariate(a, b)
            if theta > best_theta:
                best_theta = theta
                best_arm = i
        return best_arm


class UCB1(BanditAlgorithm):
    """UCB1: optimism in the face of uncertainty."""

    def choose(self) -> int:
        # Pull every arm once before applying the UCB formula.
        for i in range(self.num):
            if self.displayed[i] == 0:
                return i
        n = sum(self.displayed)
        best_theta = -1.0
        best_arm = 0
        for i in range(self.num):
            bonus = math.sqrt(2.0 * math.log(n) / self.displayed[i])
            theta = self.clicked[i] / self.displayed[i] + bonus
            if theta > best_theta:
                best_theta = theta
                best_arm = i
        return best_arm


class UCB1Tuned(BanditAlgorithm):
    """UCB1-Tuned: variance-aware refinement of UCB1."""

    def choose(self) -> int:
        for i in range(self.num):
            if self.displayed[i] == 0:
                return i
        n = sum(self.displayed)
        best_theta = -1.0
        best_arm = 0
        for i in range(self.num):
            mean = self.clicked[i] / self.displayed[i]
            # Sample variance of a Bernoulli arm is mean * (1 - mean).
            var = mean * (1.0 - mean)
            v = var + math.sqrt(2.0 * math.log(n) / self.displayed[i])
            bonus = math.sqrt(math.log(n) / self.displayed[i] * min(0.25, v))
            theta = mean + bonus
            if theta > best_theta:
                best_theta = theta
                best_arm = i
        return best_arm
