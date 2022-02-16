#!/usr/bin/env python
# coding: utf-8

from typing import Optional
from itertools import product
import timeit
import sqlite3

import numpy as np
import jax
from jax import numpy as jnp
from jax import scipy as jsp
import pandas as pd
from adopy.base import GridSpace, Task, Model, Engine

engine = None  # type: Optional[Engine]


def make_grid_design():
    # p_var & a_var for risky & ambiguous trials
    pval = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]
    aval = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75]

    # risky trials: a_var fixed to 0
    pa_risky = [[p, 0] for p in pval]
    # ambiguous trials: p_var fixed to 0.5
    pa_ambig = [[0.5, a] for a in aval]
    pr_am = np.array(pa_risky + pa_ambig)

    # r_var & r_fix while r_var > r_fix
    rval = [10, 15, 21, 31, 45, 66, 97, 141, 206, 300]
    rewards = []
    for r_var in rval:
        for r_fix in rval:
            if r_var > r_fix:
                rewards.append([r_var, r_fix])
    rewards = np.array(rewards)

    return GridSpace({("p_var", "a_var"): pr_am, ("r_var", "r_fix"): rewards})


def make_grid_param():
    global num_alpha, num_beta, num_loggamma
    return GridSpace(
        {
            "alpha": np.linspace(0, 3, num_alpha),
            "beta": np.linspace(-3, 3, num_beta),
            "loggamma": np.linspace(-4, 4, num_loggamma),
        }
    )


def make_grid_response():
    return GridSpace({"choice": [0, 1]})


class ModelLinear(Model):
    @staticmethod
    @jax.jit
    def compute(choice, p_var, a_var, r_var, r_fix, alpha, beta, loggamma):
        sv_var = jnp.power(r_var, alpha)
        sv_var = (p_var - beta * jnp.divide(a_var, 2)) * sv_var
        sv_fix = 0.5 * jnp.power(r_fix, alpha)
        p_obs = 1.0 / (1.0 + jnp.exp(-jnp.exp(loggamma) * (sv_var - sv_fix)))
        return jsp.stats.bernoulli.logpmf(choice, p_obs)


def init_engine():
    global engine, num_alpha, num_beta, num_loggamma

    grid_design = make_grid_design()
    grid_param = make_grid_param()
    grid_response = make_grid_response()

    task = Task(
        name="Choice under risk and ambiguity",
        designs=["p_var", "a_var", "r_var", "r_fix"],
        responses=["choice"],
        grid_design=grid_design,
        grid_response=grid_response,
    )

    model = ModelLinear(
        name="Linear model",
        task=task,
        params=["alpha", "beta", "loggamma"],
        grid_param=grid_param,
    )

    engine = Engine(task=task, model=model)


def run_iter():
    global engine

    if engine:
        d = engine.get_design()
        y = np.random.randint(0, 1)
        engine.update(d, {"choice": y})


if __name__ == "__main__":
    lists_setup = [np.arange(10, 55, 10) + 1, np.arange(10, 55, 10) + 1, [10]]

    for num_alpha, num_beta, num_loggamma in product(*lists_setup):
        times_iter = timeit.Timer(run_iter, init_engine).repeat(10, 10)

        df = pd.DataFrame(
            {
                "condition": "adopy_jax",
                "num_alpha": num_alpha,
                "num_beta": num_beta,
                "num_loggamma": num_loggamma,
                "time_iter": times_iter,
            },
            columns=["condition", "num_alpha", "num_beta", "num_loggamma", "time_iter"],
        )

        conn = sqlite3.connect("/Users/pluvian/src/adopy-benchmark.db")
        df.to_sql("benchmark", conn, if_exists="append", index=False)
        conn.close()
