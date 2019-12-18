from dataclasses import dataclass
from typing import Dict, List, Optional, Union


@dataclass
class TrendMetric:
    min_v: float
    max_v: float
    average: float

    @staticmethod
    def from_doc(doc: Dict[str, Union[float, int]]) -> 'TrendMetric':
        return TrendMetric(doc["min"],
                           doc["max"],
                           doc["average"])


@dataclass
class Trend:
    box_id: str
    timestamp_ms: int
    thd: Optional[TrendMetric]
    voltage: Optional[TrendMetric]
    frequency: Optional[TrendMetric]

    @staticmethod
    def from_doc(doc: Dict) -> 'Trend':
        return Trend(doc["box_id"],
                     doc["timestamp_ms"],
                     TrendMetric.from_doc(doc["thd"]) if "thd" in doc else None,
                     TrendMetric.from_doc(doc["voltage"]) if "voltage" in doc else None,
                     TrendMetric.from_doc(doc["frequency"]) if "frequency" in doc else None)


class DataPoint:
    def __init__(self,
                 ts_s: int,
                 actual_v: float,
                 min_v: float,
                 max_v: float,
                 avg_v: float,
                 stddev_v: float) -> None:
        self.ts_s: int = ts_s
        self.actual_v: float = actual_v
        self.min_v: float = min_v
        self.max_v: float = max_v
        self.avg_v: float = avg_v
        self.stddev_v: float = stddev_v

    @staticmethod
    def from_line(line: str) -> 'DataPoint':
        split_line = line.split(" ")
        ts_s = int(split_line[0])
        actual_v = float(split_line[1])
        min_v = float(split_line[2])
        max_v = float(split_line[3])
        avg_v = float(split_line[4])
        stddev_v = float(split_line[5])

        return DataPoint(ts_s, actual_v, min_v, max_v, avg_v, stddev_v)

    def __str__(self) -> str:
        return f"{self.ts_s} {self.actual_v} {self.min_v} {self.max_v} {self.avg_v} {self.stddev_v}"


def parse_file(path: str) -> List[DataPoint]:
    with open(path, "r") as fin:
        lines: List[str] = list(map(lambda line: line.strip(), fin.readlines()))
        return list(map(DataPoint.from_line, lines))
