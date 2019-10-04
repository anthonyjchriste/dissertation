import functools

import numpy as np


def sem(sigma: float, n: float) -> float:
    return sigma / np.sqrt(n)


def propagate_constant(delta: float, *c: float) -> float:
    _c = functools.reduce(lambda x, y: x * y, c)
    return delta * np.abs(_c)


def propagate_sum(*deltas: float) -> float:
    summed_squares = sum([delta**2 for delta in deltas])
    return np.sqrt(summed_squares)


def propagate_multiplication(mu: float, *mu_deltas: (float, float)) -> float:
    summed_squares = sum([(d / m)**2 for (m, d) in mu_deltas])
    return np.abs(mu) * np.sqrt(summed_squares)
