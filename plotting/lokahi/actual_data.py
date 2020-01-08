import collections
import datetime
import os
import os.path
from dataclasses import dataclass
from typing import Dict, Set, List
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database


@dataclass
class DailyMetric:
    binned_ts: str
    total_80hz_packets: int
    total_800hz_packets: int
    total_8000hz_packets: int
    total_packets: int
    total_data_bytes_80hz: int
    total_data_bytes_800hz: int
    total_data_bytes_8000hz: int
    total_data_bytes: int
    total_devices_80hz: int
    total_devices_800hz: int
    total_devices_8000hz: int
    total_devices: int

    @staticmethod
    def from_line(line: str) -> 'DailyMetric':
        split_line: List[str] = line.split(" ")
        binned_ts: str = split_line[0]
        typed_line: List[int] = list(map(int, split_line[1:]))
        return DailyMetric(binned_ts, *typed_line)

    def aml_size_bytes_80hz(self) -> int:
        return self.total_80hz_packets * 2471

    def aml_size_bytes_800hz(self) -> int:
        return self.total_800hz_packets * 2471

    def aml_size_bytes_8000hz(self) -> int:
        return self.total_8000hz_packets * 2471

    def aml_size_bytes(self) -> int:
        return self.aml_size_bytes_80hz() + self.aml_size_bytes_800hz() + self.aml_size_bytes_8000hz()

    def dt(self) -> datetime.datetime:
        split_date: List[str] = self.binned_ts.split("-")
        year: int = int(split_date[0])
        month: int = int(split_date[1])
        day: int = int(split_date[2])
        return datetime.datetime(year, month, day, 0, 0, 0)

    def ts(self) -> int:
        return int(self.dt().timestamp())


def load_daily_metrics(path: str) -> List[DailyMetric]:
    with open(path, "r") as fin:
        lines: List[str] = fin.readlines()
        filterd_lines: List[str] = list(map(lambda line: line.strip(), lines))
        return list(map(DailyMetric.from_line, filterd_lines))


def plot_active_sensors(daily_metrics: List[DailyMetric]):
    # Data
    dts: np.ndarray = np.array(list(map(DailyMetric.dt, daily_metrics)))
    total_devices_80hz: np.ndarray = np.array(list(map(lambda daily_metric: daily_metric.total_devices_80hz, daily_metrics)))
    total_devices_800hz: np.ndarray = np.array(list(map(lambda daily_metric: daily_metric.total_devices_800hz, daily_metrics)))
    total_devices_8000hz: np.ndarray = np.array(list(map(lambda daily_metric: daily_metric.total_devices_8000hz, daily_metrics)))
    total_devices: np.ndarray = np.array(list(map(lambda daily_metric: daily_metric.total_devices, daily_metrics)))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.plot(dts, total_devices_80hz, label="Total Devices 80Hz", color="blue")
    ax.plot(dts, [total_devices_80hz.mean() for _ in dts], label="Mean Devices 80Hz", color="blue", linestyle="--")

    ax.plot(dts, total_devices_800hz, label="Total Devices 800Hz", color="green")
    ax.plot(dts, [total_devices_800hz.mean() for _ in dts], label="Mean Devices 800Hz", color="green", linestyle="--")

    ax.plot(dts, total_devices_8000hz, label="Total Devices 8000Hz", color="orange")
    ax.plot(dts, [total_devices_8000hz.mean() for _ in dts], label="Mean Devices 8000Hz", color="orange", linestyle="--")

    ax.plot(dts, total_devices, label="Total Devices", color="red")
    ax.plot(dts, [total_devices.mean() for _ in dts], label="Mean Devices", color="red", linestyle="--")

    ax.set_title("Active Lokahi Sensors")
    ax.set_ylabel("Active Lokahi Sensors")
    ax.set_xlabel("Time (UTC)")
    ax.legend()

    fig.show()

    fig.savefig("/Users/anthony/Development/dissertation/src/figures/lokahi_num_sensors.png")


def main():
    daily_metrics: List[DailyMetric] = load_daily_metrics("metrics.txt")
    plot_active_sensors(daily_metrics)
    # plot_iml()


if __name__ == "__main__":
    main()
