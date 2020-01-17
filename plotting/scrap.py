from typing import List

import numpy as np

def sum_series(series: np.ndarray) -> np.ndarray:
    result: List[int] = [series[0]]
    for i in range(1, len(series)):
        result.append(result[i - 1] + series[i])

    return np.array(result)


def main():
    a = np.array([1, 2, 3, 4, 5])
    print(sum_series(a))


if __name__ == "__main__":
    main()