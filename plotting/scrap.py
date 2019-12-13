from typing import List
import numpy as np

def correct_counts(counts: np.ndarray) -> np.ndarray:
    diffs = np.diff(counts)
    diffs[np.where(diffs < 0)] = 0

    corrected_counts: List[int] = []
    for i in range(len(diffs)):
        corrected_counts.append(diffs[0:i].sum())

    return np.array(corrected_counts)

if __name__ == "__main__":
    a: np.ndarray = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 0, 1, 0])
    b = correct_counts(a)
    print(b)
