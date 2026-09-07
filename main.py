"""Benchmark runner for bandit-exp.

Evaluates *every* algorithm against *every* benchmark environment and
writes the results to a JSON file. There is no UI: this is a pure,
reproducible batch job.

Fairness
--------
For each (environment, algorithm) pair and each trial we seed both the
environment and the algorithm deterministically from a base seed. Every
algorithm therefore faces the *same* sequence of reward realisations on a
given trial, so differences in outcome reflect the algorithm, not luck.

Metrics
-------
- ``final_ctr``       : overall click-through rate at the end of the run.
- ``cumulative_regret``: sum over time of (optimal mean - chosen arm mean),
  the standard way to score a bandit. Lower is better.
"""

from __future__ import annotations

import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

# Allow running as a plain script (``python main.py``) without installing
# the package: make ``src/`` importable.
sys.path.insert(0, str(Path(__file__).parent / "src"))

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
    SinusoidalNoiseBernoulli,
    StationaryBernoulli,
)

RESULTS_DIR = Path(__file__).parent / "results"


@dataclass(frozen=True)
class EnvSpec:
    name: str
    build: Callable[[random.Random], Environment]


@dataclass(frozen=True)
class AlgSpec:
    name: str
    build: Callable[[int, random.Random], BanditAlgorithm]


def make_env_specs() -> list[EnvSpec]:
    """The suite of benchmark problem settings."""
    return [
        EnvSpec(
            "stationary_easy",
            lambda rng: StationaryBernoulli([0.1, 0.2, 0.5], rng=rng),
        ),
        EnvSpec(
            "gap_hard",
            lambda rng: GapBernoulli(num=10, best=0.5, gap=0.02, rng=rng),
        ),
        EnvSpec(
            "gap_easy",
            lambda rng: GapBernoulli(num=10, best=0.5, gap=0.25, rng=rng),
        ),
        EnvSpec(
            "drifting",
            lambda rng: DriftingBernoulli(
                start=[0.1, 0.4, 0.7],
                end=[0.7, 0.4, 0.1],
                horizon=2000,
                rng=rng,
            ),
        ),
        EnvSpec(
            "abrupt_change",
            lambda rng: AbruptChangeBernoulli(
                before=[0.1, 0.2, 0.6],
                after=[0.6, 0.2, 0.1],
                change_at=1000,
                rng=rng,
            ),
        ),
        # Real CTRs of a video site's top page: four very close arms.
        EnvSpec(
            "video_top_stationary",
            lambda rng: StationaryBernoulli(
                [0.1115, 0.1161, 0.1195, 0.1197], rng=rng
            ),
        ),
        # Same base CTRs, plus a small deterministic sinusoidal wobble
        # (< 1%, so arms stay close). theta_i are arbitrary constants,
        # chosen distinct so the arms do not oscillate in phase.
        EnvSpec(
            "video_top_noisy",
            lambda rng: SinusoidalNoiseBernoulli(
                base=[0.1115, 0.1161, 0.1195, 0.1197],
                amplitude=0.009,  # 0.9% peak wobble (< 1%)
                theta=[0.011, 0.017, 0.023, 0.031],
                rng=rng,
            ),
        ),
    ]


def make_alg_specs() -> list[AlgSpec]:
    """Every algorithm to be compared."""
    return [
        AlgSpec("Random", lambda n, rng: Random(n, rng=rng)),
        AlgSpec("EGreedy(0.01)", lambda n, rng: EGreedy(n, 0.01, rng=rng)),
        AlgSpec("EGreedy(0.1)", lambda n, rng: EGreedy(n, 0.1, rng=rng)),
        AlgSpec("ThomsonSampling", lambda n, rng: ThomsonSampling(n, rng=rng)),
        AlgSpec("UCB1", lambda n, rng: UCB1(n, rng=rng)),
        AlgSpec("UCB1Tuned", lambda n, rng: UCB1Tuned(n, rng=rng)),
    ]


def run_trial(
    env: Environment, alg: BanditAlgorithm, horizon: int
) -> tuple[float, float]:
    """Run one algorithm on one environment for ``horizon`` steps.

    Returns ``(final_ctr, cumulative_regret)``.
    """
    cumulative_regret = 0.0
    for t in range(horizon):
        i = alg.display()
        clicked = bool(env.pull(i, t))
        alg.reward(i, clicked)
        cumulative_regret += env.optimal_mean(t) - env.mean(i, t)
    return alg.ctr_overall(), cumulative_regret


def benchmark(
    horizon: int = 2000, num_trials: int = 20, base_seed: int = 12345
) -> dict:
    env_specs = make_env_specs()
    alg_specs = make_alg_specs()

    results: dict = {
        "config": {
            "horizon": horizon,
            "num_trials": num_trials,
            "base_seed": base_seed,
        },
        "environments": {},
    }

    for env_spec in env_specs:
        env_result: dict = {}
        for alg_spec in alg_specs:
            final_ctrs: list[float] = []
            regrets: list[float] = []
            for trial in range(num_trials):
                # Same seed for env and alg on a given (env, alg, trial),
                # derived deterministically so runs are reproducible and
                # every algorithm sees a comparable reward stream.
                seed = base_seed + trial
                env = env_spec.build(random.Random(seed))
                alg = alg_spec.build(env.num, random.Random(seed + 1))
                ctr, regret = run_trial(env, alg, horizon)
                final_ctrs.append(ctr)
                regrets.append(regret)
            env_result[alg_spec.name] = {
                "mean_final_ctr": sum(final_ctrs) / num_trials,
                "mean_cumulative_regret": sum(regrets) / num_trials,
            }
        results["environments"][env_spec.name] = env_result

    return results


def main() -> None:
    started = time.time()
    results = benchmark()
    results["config"]["elapsed_seconds"] = round(time.time() - started, 3)

    RESULTS_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    out_path = RESULTS_DIR / f"benchmark-{timestamp}.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"Wrote results to {out_path}")
    # Also print a compact regret leaderboard per environment.
    for env_name, env_result in results["environments"].items():
        print(f"\n[{env_name}] mean cumulative regret (lower is better):")
        ranked = sorted(
            env_result.items(), key=lambda kv: kv[1]["mean_cumulative_regret"]
        )
        for alg_name, metrics in ranked:
            print(
                f"  {alg_name:20s} "
                f"regret={metrics['mean_cumulative_regret']:9.2f}  "
                f"ctr={metrics['mean_final_ctr']:.4f}"
            )


if __name__ == "__main__":
    main()
