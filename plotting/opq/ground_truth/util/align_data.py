import datetime
from dataclasses import dataclass
import functools
from typing import Callable, List, Tuple, Set, Iterable

import numpy as np


@dataclass
class SeriesSpec:
    time: List
    values: List
    dt_func: Callable
    v_func: Callable


def bin_dt_by_min(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0, 0, tzinfo=datetime.timezone.utc)


def intersect_lists(lists: List[List[datetime.datetime]]) -> Set[datetime.datetime]:
    sets: List[Set] = list(map(set, lists))
    return set.intersection(*sets)


def align_data_multi(series: List[SeriesSpec]) -> None:
    all_dts: List[List[datetime.datetime]] = []
    all_binned_dts: List[List[datetime.datetime]] = []

    for serie in series:
        dts: Iterable[List[datetime.datetime]] = list(map(serie.dt_func, serie.time))
        binned_dts: Iterable[List[datetime.datetime]] = list(map(bin_dt_by_min, dts))
        all_dts.extend(dts)
        all_binned_dts.extend(binned_dts)

    intersecting_dts: Set[datetime.datetime] = intersect_lists(all_binned_dts)

    for serie_idx, serie in enumerate(series):
        pass




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
