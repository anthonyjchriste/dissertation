from functools import reduce
from typing import List, Set
import numpy as np


def sum_series(series: np.ndarray) -> np.ndarray:
    result: List[int] = []
    for i in range(1, len(series)):
        result.append(sum(series[:i]))

    return np.array(result)


if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 6, 7]
    a = np.array(a)
    print(sum_series(a))
