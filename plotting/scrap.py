from functools import reduce
from typing import List, Set
import numpy as np


def intersect_lists(lists: List[List]) -> Set:
    sets: List[Set] = list(map(set, lists))
    return set.intersection(*sets)


if __name__ == "__main__":
    a = [1, 2, 3]
    b = [2, 3, 4]
    c = [3, 4, 5]

    print(np.array({1, 2, 3}))