import numpy as np


def sem(sigma: float, n: float) -> float:
    return sigma / np.sqrt(n)
