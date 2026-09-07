# bandit-exp

![](https://img.shields.io/badge/python-3.13%2B-brightgreen?style=flat-square)

A small harness for **fairly comparing multi-armed bandit algorithms**
across a suite of benchmark problem settings. It has no third-party
runtime dependencies — only the Python standard library.

## Design

- **`environments.py`** — the benchmark *problem settings*. All are
  Bernoulli (0/1 reward) bandits; they differ in how each arm's click
  probability is determined over time:
  - `StationaryBernoulli` — fixed per-arm probabilities.
  - `GapBernoulli` — controlled difficulty via an explicit optimal/gap.
  - `DriftingBernoulli` — probabilities drift linearly (non-stationary).
  - `AbruptChangeBernoulli` — probabilities switch at a change point.

  Each environment exposes the true `mean(i, t)` and `optimal_mean(t)`,
  which are used to compute regret. A `random.Random` is injected so runs
  are reproducible and algorithms can be compared on identical reward
  streams.

- **`algorithms.py`** — the bandit algorithms (`Random`, `EGreedy`,
  `ThomsonSampling`, `UCB1`, `UCB1Tuned`). An algorithm only sees the arm
  it chose and the 0/1 reward — never the environment's true means.

- **`main.py`** — the benchmark runner. Evaluates **every algorithm
  against every environment** over many seeded trials and writes the
  aggregated metrics (mean final CTR and mean cumulative regret) to a
  timestamped JSON file under `results/`.

## Usage

```bash
uv run python main.py      # or: make run
```

Results are written to `results/benchmark-<timestamp>.json` (this
directory is gitignored). A regret leaderboard per environment is also
printed to stdout.

## Tests

```bash
uv run pytest -q           # or: make test
```
