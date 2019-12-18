import datetime
from typing import Callable, List, Tuple, Set

import numpy as np


# A = TypeVar("A")
# B = TypeVar("B")
# C = TypeVar("C")
# D = TypeVar("D")


def bin_dt_by_min(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0, 0, tzinfo=datetime.timezone.utc)


def align_data_by_min(series_a: List,
                      series_b: List,
                      dt_func_a: Callable,
                      dt_func_b: Callable,
                      v_func_a: Callable,
                      v_func_b: Callable) -> Tuple[np.ndarray,
                                                   np.ndarray,
                                                   np.ndarray,
                                                   np.ndarray]:
    dts_a = list(map(dt_func_a, series_a))
    dts_b: List[datetime.datetime] = list(map(dt_func_b, series_b))

    binned_dts_a: List[datetime.datetime] = list(map(bin_dt_by_min, dts_a))
    binned_dts_b: List[datetime.datetime] = list(map(bin_dt_by_min, dts_b))

    intersecting_dts: Set[datetime.datetime] = set(binned_dts_a).intersection(set(binned_dts_b))

    aligned_a_dts: List[datetime.datetime] = []
    aligned_a_vs: List = []
    aligned_b_dts: List[datetime.datetime] = []
    aligned_b_vs: List = []

    already_seen_a: Set[datetime.datetime] = set()
    already_seen_b: Set[datetime.datetime] = set()

    for i, dt in enumerate(binned_dts_a):
        if dt in intersecting_dts and dt not in already_seen_a:
            aligned_a_dts.append(dt)
            aligned_a_vs.append(v_func_a(series_a[i]))
            already_seen_a.add(dt)

    for i, dt in enumerate(binned_dts_b):
        if dt in intersecting_dts and dt not in already_seen_b:
            aligned_b_dts.append(dt)
            aligned_b_vs.append(v_func_b(series_b[i]))
            already_seen_b.add(dt)

    return np.array(aligned_a_dts), \
           np.array(aligned_a_vs), \
           np.array(aligned_b_dts), \
           np.array(aligned_b_vs)
