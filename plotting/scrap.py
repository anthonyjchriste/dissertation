import datetime
from functools import reduce
from typing import List, Set, Callable, Any, Tuple
import numpy as np


def resample(from_series: List,
             to_series: List,
             from_series_dt_fn: Callable[[Any], datetime.datetime],
             to_series_dt_fn: Callable[[Any], datetime.datetime]) -> Tuple[List[datetime.datetime], List, List]:
    pass


if __name__ == "__main__":

