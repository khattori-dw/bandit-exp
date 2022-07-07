import math
import random

import numpy.random
import streamlit

streamlit.title("Bandit-exp")
streamlit.markdown(
    "[khattori-dw/bandit-exp](https://github.com/khattori-dw/bandit-exp)"
)


class Arms:
    def __init__(self, ps: list[float]):
        self.ps = ps
        self.num = len(ps)

    def choose(self, i) -> bool:
        """Returns True if clicked"""
        assert 0 <= i < self.num
        p = self.ps[i]
        return random.random() < p


class BanditAlgorithm:

    num: int
    displayed: list[int]
    clicked: list[int]
    ctr: list[float]

    def __init__(self, num: int):
        self.num = num
        self.displayed = [0] * num
        self.clicked = [0] * num
        self.ctr = [0.0] * num

    def choose(self) -> int:
        raise NotImplementedError

    def display(self) -> int:
        """Choice an arm, and display it"""
        i = self.choose()
        self.displayed[i] += 1
        self.ctr[i] = self.clicked[i] / self.displayed[i]
        return i

    def reward(self, i: int, clicked: bool) -> None:
        """Given reward for arm #i"""
        if clicked:
            self.clicked[i] += 1
            self.ctr[i] = self.clicked[i] / self.displayed[i]

    def ctr_overall(self) -> float:
        sum_displayed = sum(self.displayed)
        sum_clicked = sum(self.clicked)
        if sum_displayed == 0:
            return 0.0
        return sum_clicked / sum_displayed


class EGreedy(BanditAlgorithm):
    def __init__(self, num: int, epsilon: float):
        super().__init__(num)
        self.epsilon = epsilon

    def choose(self) -> int:
        """Choice an arm (not displayed, not clicked)"""
        if self.epsilon < random.random():  # Explore
            return random.randrange(self.num)
        else:  # Exploit
            maxctr = max(self.ctr)
            maxarms = [i for i in range(self.num) if self.ctr[i] >= maxctr]
            return random.choice(maxarms)


class Random(BanditAlgorithm):
    def __init__(self, num: int):
        super().__init__(num)

    def choose(self) -> int:
        return random.randrange(self.num)


class ThomsonSampling(BanditAlgorithm):
    def __init__(self, num: int, alpha: float, beta: float):
        assert alpha > 0
        assert beta > 0
        super().__init__(num)
        self.alpha = alpha
        self.beta = beta

    def choose(self) -> int:
        max_theta = 0.0
        max_arm_index = 0
        for i in range(self.num):
            theta = numpy.random.beta(
                self.alpha + self.clicked[i],
                self.beta + self.displayed[i] - self.clicked[i],
            )
            if theta > max_theta:
                max_theta = theta
                max_arm_index = i
        return max_arm_index


class UCB1(BanditAlgorithm):
    def __init__(self, num: int):
        super().__init__(num)

    def choose(self) -> int:
        max_theta = 0.0
        max_arm_index = 0
        n = sum(self.displayed)
        for i in range(self.num):
            if self.displayed[i] == 0:
                return i
        for i in range(self.num):
            theta = self.clicked[i] / self.displayed[i] + math.sqrt(
                2.0 * math.log(n) / self.displayed[i]
            )
            if theta > max_theta:
                max_theta = theta
                max_arm_index = i
        return max_arm_index


class UCB1Tuned(BanditAlgorithm):
    def __init__(self, num: int):
        super().__init__(num)

    def choose(self) -> int:
        max_theta = 0.0
        max_arm_index = 0
        n = sum(self.displayed)
        for i in range(self.num):
            if self.displayed[i] == 0:
                return i
        for i in range(self.num):
            variance = self.clicked[i] * (self.displayed[i] - self.clicked[i]) / (
                max(1, self.displayed[i]) ** 2
            ) + math.sqrt(2.0 * math.log(n) / self.displayed[i])
            theta = self.clicked[i] / self.displayed[i] + min(
                math.sqrt(math.log(n) / self.displayed[i]), variance
            )
            if theta > max_theta:
                max_theta = theta
                max_arm_index = i
        return max_arm_index


streamlit.subheader("Arms")
num_arms = int(streamlit.number_input("#arms", min_value=3, max_value=100))
ps = []
for i in range(num_arms):
    p = (
        streamlit.number_input(
            f"CTR(%) for arm#{i}", value=3.0, min_value=0.0, max_value=100.0, step=0.5
        )
        / 100.0
    )
    ps.append(p)
arms = Arms(ps)


streamlit.subheader("Algorithms")
num_algs = int(streamlit.number_input("#algorithms", min_value=1, max_value=100))
algs: list[BanditAlgorithm] = []
for i in range(num_algs):
    streamlit.markdown(f"**Algorithm#{i}**")
    algname = streamlit.selectbox(
        "algorithm",
        ["EGreedy", "Random", "ThomsonSampling", "UCB1", "UCB1Tuned"],
        key=f"algname{i}",
    )

    if algname == "EGreedy":
        epsilon = streamlit.number_input(
            "epsilon",
            value=0.01,
            min_value=0.0,
            max_value=1.0,
            step=0.01,
            key=f"EGreedy{i}",
        )
        algs.append(EGreedy(num_arms, epsilon))

    elif algname == "Random":
        algs.append(Random(num_arms))

    elif algname == "ThomsonSampling":
        alpha = streamlit.number_input("alpha", value=1.0, min_value=0.1, step=0.1)
        beta = streamlit.number_input("beta", value=1.0, min_value=0.1, step=0.1)
        algs.append(ThomsonSampling(num_arms, alpha, beta))

    elif algname == "UCB1":
        algs.append(UCB1(num_arms))

    elif algname == "UCB1Tuned":
        algs.append(UCB1Tuned(num_arms))


streamlit.subheader("Run Algorithms")
maxtime = int(streamlit.number_input("#maxtime", value=1000, min_value=1, step=1))
num_tries = int(
    streamlit.number_input("#tries for each algorithms", value=3, min_value=1, step=1)
)
if streamlit.button("Run"):
    table = []
    names = []
    for alg_index, alg in enumerate(algs):
        tries = []
        for _ in range(num_tries):
            ctrs = []
            for t in range(maxtime):
                i = alg.display()
                clicked = arms.choose(i)
                alg.reward(i, clicked)
                ctrs.append(alg.ctr_overall())
            tries.append(ctrs)
        ctrs = [
            sum(tries[j][t] for j in range(num_tries)) / num_tries * 100.0
            for t in range(maxtime)
        ]
        table.append(ctrs)
        names.append(f"Alg#{alg_index} ({alg.__class__.__name__})")

    streamlit.markdown("#### cumulative CTR(%)")
    streamlit.line_chart({name: ctrs for name, ctrs in zip(names, table)})
