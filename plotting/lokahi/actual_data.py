import collections
import datetime
import os
import os.path
from dataclasses import dataclass
from typing import Dict, Set, List, Callable, Tuple
import urllib.parse

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database
import scipy.stats as stats

seconds_in_day = 86400
seconds_in_two_weeks = seconds_in_day * 14
seconds_in_month = seconds_in_day * 30.4167
seconds_in_year = seconds_in_month * 12
seconds_in_2_years = seconds_in_year * 2


@dataclass
class SeriesSpec:
    values: List
    dt_func: Callable
    v_func: Callable


def bin_dt_by_min(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0, 0, tzinfo=datetime.timezone.utc)


def bin_dt_by_day(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0, 0, tzinfo=datetime.timezone.utc)


def intersect_lists(lists: List[List[datetime.datetime]]) -> Set[datetime.datetime]:
    sets: List[Set] = list(map(set, lists))
    return set.intersection(*sets)


def align_data_multi(series: List[SeriesSpec]) -> List[Tuple[np.ndarray, np.ndarray]]:
    all_dts: List[List[datetime.datetime]] = []
    all_binned_dts: List[List[datetime.datetime]] = []
    all_vals: List[List[float]] = []
    already_seens: List[Set[datetime.datetime]] = []

    for _ in range(len(series)):
        already_seens.append(set())

    for serie in series:
        dts: List[datetime.datetime] = list(map(serie.dt_func, serie.values))
        binned_dts: List[datetime.datetime] = list(map(bin_dt_by_day, dts))
        vals: List = list(map(serie.v_func, serie.values))
        all_dts.append(dts)
        all_binned_dts.append(binned_dts)
        all_vals.append(vals)

    intersecting_dts: Set[datetime.datetime] = intersect_lists(all_binned_dts)

    res: List[Tuple[np.ndarray, np.ndarray]] = []

    for i, binned_dts in enumerate(all_binned_dts):
        dts = []
        vs = []
        for j, binned_dt in enumerate(binned_dts):
            if binned_dt in intersecting_dts and binned_dt not in already_seens[i]:
                dts.append(binned_dt)
                vs.append(all_vals[i][j])
                already_seens[i].add(binned_dt)
        res.append((np.array(dts), np.array(vs)))

    return res


class Data:
    def __init__(self,
                 time: int,
                 total_samples: int,
                 total_samples_b: int,
                 total_measurements: int,
                 total_measurements_b: int,
                 total_orphaned_measurements: int,
                 total_orphaned_measurements_b: int,
                 total_event_measurements: int,
                 total_event_measurements_b: int,
                 total_incident_measurements: int,
                 total_incident_measurements_b: int,
                 total_trends: int,
                 total_trends_b: int,
                 total_orphaned_trends: int,
                 total_orphaned_trends_b: int,
                 total_event_trends: int,
                 total_event_trends_b: int,
                 total_incident_trends: int,
                 total_incident_trends_b: int,
                 total_events: int,
                 total_events_b: int,
                 total_orphaned_events: int,
                 total_orphaned_events_b: int,
                 total_incident_events: int,
                 total_incident_events_b: int,
                 total_incidents: int,
                 total_incidents_b: int,
                 total_laha_b: int,
                 total_iml_b: int,
                 total_aml_b: int,
                 total_dl_b: int,
                 total_il_b: int):
        self.time: int = time
        self.total_samples: int = total_samples
        self.total_samples_b: int = total_samples_b
        self.total_measurements: int = total_measurements
        self.total_measurements_b: int = total_measurements_b
        self.total_orphaned_measurements: int = total_orphaned_measurements
        self.total_orphaned_measurements_b: int = total_orphaned_measurements_b
        self.total_event_measurements: int = total_event_measurements
        self.total_event_measurements_b: int = total_event_measurements_b
        self.total_incident_measurements: int = total_incident_measurements
        self.total_incident_measurements_b: int = total_incident_measurements_b
        self.total_trends: int = total_trends
        self.total_trends_b: int = total_trends_b
        self.total_orphaned_trends: int = total_orphaned_trends
        self.total_orphaned_trends_b: int = total_orphaned_trends_b
        self.total_event_trends: int = total_event_trends
        self.total_event_trends_b: int = total_event_trends_b
        self.total_incident_trends: int = total_incident_trends
        self.total_incident_trends_b: int = total_incident_trends_b
        self.total_events: int = total_events
        self.total_events_b: int = total_events_b
        self.total_orphaned_events: int = total_orphaned_events
        self.total_orphaned_events_b: int = total_orphaned_events_b
        self.total_incident_events: int = total_incident_events
        self.total_incident_events_b: int = total_incident_events_b
        self.total_incidents: int = total_incidents
        self.total_incidents_b: int = total_incidents_b
        self.total_laha_b: int = total_laha_b
        self.total_iml_b: int = total_iml_b
        self.total_aml_b: int = total_aml_b
        self.total_dl_b: int = total_dl_b
        self.total_il_b: int = total_il_b

    @staticmethod
    def from_line(line: str) -> 'Data':
        split_line = line.split(",")
        as_ints = list(map(int, split_line))
        return Data(*as_ints)


def parse_file(path: str) -> List[Data]:
    with open(path, "r") as fin:
        lines = list(map(lambda line: line.strip(), fin.readlines()))
        return list(map(lambda line: Data.from_line(line), lines))


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
        return datetime.datetime(year, month, day, 0, 0, 0, tzinfo=datetime.timezone.utc)

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
    total_devices_80hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_devices_80hz, daily_metrics)))
    total_devices_800hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_devices_800hz, daily_metrics)))
    total_devices_8000hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_devices_8000hz, daily_metrics)))
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
    ax.plot(dts, [total_devices_8000hz.mean() for _ in dts], label="Mean Devices 8000Hz", color="orange",
            linestyle="--")

    ax.plot(dts, total_devices, label="Total Devices", color="red")
    ax.plot(dts, [total_devices.mean() for _ in dts], label="Mean Devices", color="red", linestyle="--")

    ax.set_title("Active Lokahi Sensors")
    ax.set_ylabel("Active Lokahi Sensors")
    ax.set_xlabel("Time (UTC)")
    ax.legend()

    fig.show()

    fig.savefig("/Users/anthony/Development/dissertation/src/figures/lokahi_num_sensors.png")


def sum_series(series: np.ndarray) -> np.ndarray:
    result: List[int] = []
    for i in range(1, len(series)):
        result.append(sum(series[:i]))

    return np.array(result)


def slope_intercept(slope: float, intercept: float) -> str:
    return f"y = {slope} * x + {intercept}"


def plot_iml(daily_metrics: List[DailyMetric]):
    # Data
    dts: np.ndarray = np.array(list(map(DailyMetric.dt, daily_metrics)))

    total_data_bytes_80hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_80hz, daily_metrics)))
    total_data_gb_80hz = total_data_bytes_80hz / 1_000_000_000.0

    total_data_bytes_800hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_800hz, daily_metrics)))
    total_data_gb_800hz = total_data_bytes_800hz / 1_000_000_000.0

    total_data_bytes_8000hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_8000hz, daily_metrics)))
    total_data_gb_8000hz = total_data_bytes_8000hz / 1_000_000_000.0

    total_data_bytes: np.ndarray = total_data_bytes_80hz + total_data_bytes_800hz + total_data_bytes_8000hz
    total_data_gb = total_data_bytes / 1_000_000_000.0

    xs = np.array(list(map(lambda dt: dt.timestamp(), dts[1:])))
    # xs = xs - xs[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, sum_series(total_data_gb))
    print("iml", slope_intercept(slope, intercept), total_data_gb[0])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    # fig.suptitle(f"Laha IML (Lokahi)\n"
    #              f"$LR_{{IML\ Total}}$=($m$={slope} $b$={intercept} $R^2$={r_value ** 2} $\sigma$={std_err})")

    ax.plot(dts[1:], sum_series(total_data_gb_80hz), label="Total IML Data 80 Hz", color="blue")
    ax.plot(dts[1:], sum_series(total_data_gb_800hz), label="Total IML Data 800 Hz", color="green")
    ax.plot(dts[1:], sum_series(total_data_gb_8000hz), label="Total IML Data 8000 Hz", color="orange")
    ax.plot(dts[1:], sum_series(total_data_gb), label="Total IML Data", color="red")

    ax.errorbar(dts[1:], intercept + slope * xs, yerr=std_err,
                label=f"IML Total GB LR ($m$={slope:.5f} $b$={intercept:.5f} $R^2$={r_value ** 2:.5f} $\sigma$="
                      f"{std_err:.5f})",
                color="black", linestyle=":")

    ax.legend()
    ax.set_title("Lokahi IML Growth")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Size GB")

    fig.show()
    fig.savefig("/home/opq/Documents/anthony/dissertation/src/figures/lokahi_actual_iml.png")


def plot_aml(daily_metrics: List[DailyMetric]):
    # Data
    dts: np.ndarray = np.array(list(map(DailyMetric.dt, daily_metrics)))

    total_data_bytes_80hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_80hz(), daily_metrics)))
    total_data_gb_80hz = total_data_bytes_80hz / 1_000_000_000.0

    total_data_bytes_800hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_800hz(), daily_metrics)))
    total_data_gb_800hz = total_data_bytes_800hz / 1_000_000_000.0

    total_data_bytes_8000hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_8000hz(), daily_metrics)))
    total_data_gb_8000hz = total_data_bytes_8000hz / 1_000_000_000.0

    total_data_bytes: np.ndarray = total_data_bytes_80hz + total_data_bytes_800hz + total_data_bytes_8000hz
    total_data_gb = total_data_bytes / 1_000_000_000.0

    xs = np.array(list(map(lambda dt: dt.timestamp(), dts[1:])))
    # xs = xs - xs[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, sum_series(total_data_gb))
    print("aml", slope_intercept(slope, intercept), total_data_gb[0])

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    # fig.suptitle(f"Laha IML (Lokahi)\n"
    #              f"$LR_{{IML\ Total}}$=($m$={slope} $b$={intercept} $R^2$={r_value ** 2} $\sigma$={std_err})")

    ax.plot(dts[1:], sum_series(total_data_gb_80hz), label="Total AML Data 80 Hz", color="blue")
    ax.plot(dts[1:], sum_series(total_data_gb_800hz), label="Total AML Data 800 Hz", color="green")
    ax.plot(dts[1:], sum_series(total_data_gb_8000hz), label="Total AML Data 8000 Hz", color="orange")
    ax.plot(dts[1:], sum_series(total_data_gb), label="Total AML Data", color="red")

    ax.errorbar(dts[1:], intercept + slope * xs, yerr=std_err,
                label=f"AML Total GB LR ($m$={slope:.5f} $b$={intercept:.5f} $R^2$={r_value ** 2:.5f} $\s"
                      f"igma$={std_err:.5f})",
                color="black", linestyle=":")

    ax.legend()
    ax.set_title("Lokahi AML Growth")
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Size GB")

    fig.show()
    fig.savefig("/home/opq/Documents/anthony/dissertation/src/figures/lokahi_actual_aml.png")


def plot_iml_vs_est(daily_metrics: List[DailyMetric]):
    # Data
    dts: np.ndarray = np.array(list(map(DailyMetric.dt, daily_metrics))[1:])
    tss: np.ndarray = np.array(list(map(DailyMetric.ts, daily_metrics))[1:])
    tss = tss - tss[0]

    total_data_bytes_80hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_80hz, daily_metrics)))

    total_data_bytes_800hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_800hz, daily_metrics)))

    total_data_bytes_8000hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_8000hz, daily_metrics)))

    total_data_bytes: np.ndarray = total_data_bytes_80hz + total_data_bytes_800hz + total_data_bytes_8000hz
    total_data_gb = sum_series(total_data_bytes / 1_000_000_000.0)

    est_data_80hz = 4 * 80 * 38 * tss
    est_data_800hz = 4 * 800 * 99 * tss
    est_data_8000hz = 4 * 8000 * 5 * tss
    est_data_total = (est_data_80hz + est_data_800hz + est_data_8000hz) / 1_000_000_000.0

    max_y = max(est_data_total)

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Lokahi: Estimated IML vs Actual IML")

    est_ax = ax[0]
    est_ax.plot(dts, est_data_total, label="Estimated IML Size")

    est_ax.set_ylim(ymax=max_y)
    est_ax.set_title("Estimated IML")
    est_ax.set_ylabel("Size GB")
    est_ax.legend()

    actual_ax = ax[1]
    actual_ax.plot(dts, total_data_gb, label="Actual IML Size")

    actual_ax.set_ylim(ymax=max_y)
    actual_ax.set_title("Actual IML")
    actual_ax.set_ylabel("Size GB")
    actual_ax.legend()

    diff_ax = ax[2]
    diff_ax.plot(dts, est_data_total - total_data_gb, label="Difference")

    diff_ax.set_ylim(ymax=max_y)
    diff_ax.set_title("Difference (Estimated - Actual)")
    diff_ax.set_xlabel("Time (UTC)")
    diff_ax.set_ylabel("Size GB")
    diff_ax.legend()

    # fig.show()
    fig.savefig("/Users/anthony/Development/dissertation/src/figures/lokahi_actual_iml_vs_est.png")


def plot_iml_vs_sim(dts: np.ndarray,
                    aligned_daily_metrics: np.ndarray,
                    aligned_sim_80: np.ndarray,
                    aligned_sim_800: np.ndarray,
                    aligned_sim_8000: np.ndarray):
    # Data

    total_data_bytes_80hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_80hz, aligned_daily_metrics)))

    total_data_bytes_800hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_800hz, aligned_daily_metrics)))

    total_data_bytes_8000hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.total_data_bytes_8000hz, aligned_daily_metrics)))

    total_data_bytes: np.ndarray = total_data_bytes_80hz + total_data_bytes_800hz + total_data_bytes_8000hz
    total_data_gb = sum_series(total_data_bytes / 1_000_000_000.0)

    # Sim data
    sim_80_bytes = np.array(list(map(lambda data: data.total_samples * 4, aligned_sim_80))) * 38
    sim_800_bytes = np.array(list(map(lambda data: data.total_samples * 4, aligned_sim_800))) * 99
    sim_8000_bytes = np.array(list(map(lambda data: data.total_samples * 4, aligned_sim_8000))) * 5
    sim_total_bytes = sim_80_bytes + sim_800_bytes + sim_8000_bytes
    sim_total_gb = sim_total_bytes / 1_000_000_000.0

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Lokahi: Simulated IML vs Actual IML")

    sim_ax = ax[0]
    sim_ax.plot(dts, sim_total_gb, label="Simulated IML Size")

    sim_ax.set_yscale("log")
    sim_ax.set_ylim(ymax=max(total_data_gb))
    sim_ax.legend()
    sim_ax.set_title("Simulated IML")
    sim_ax.set_ylabel("Size GB")


    actual_ax = ax[1]
    actual_ax.plot(dts[1:], total_data_gb, label="Actual IML Size")

    actual_ax.set_yscale("log")
    actual_ax.set_ylim(ymax=max(total_data_gb), ymin=.001)
    actual_ax.legend()
    actual_ax.set_title("Actual IML")
    actual_ax.set_ylabel("Size GB")

    diff_ax = ax[2]
    diff_ax.plot(dts[1:], sim_total_gb[1:] - total_data_gb, label="Difference")

    diff_ax.legend()
    # diff_ax.set_yscale("log")
    # diff_ax.set_ylim(ymax=max(total_data_gb))
    diff_ax.set_title("Difference (Simulated - Actual)")
    diff_ax.set_xlabel("Time (UTC)")
    diff_ax.set_ylabel("Size GB")




    fig.savefig("/Users/anthony/Development/dissertation/src/figures/lokahi_actual_iml_vs_sim.png")


def plot_aml_vs_est(daily_metrics: List[DailyMetric]):
    # Data
    dts: np.ndarray = np.array(list(map(DailyMetric.dt, daily_metrics))[1:])
    tss: np.ndarray = np.array(list(map(DailyMetric.ts, daily_metrics))[1:])
    tss = tss - tss[0]

    total_data_bytes_80hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_80hz(), daily_metrics)))

    total_data_bytes_800hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_800hz(), daily_metrics)))

    total_data_bytes_8000hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_8000hz(), daily_metrics)))

    total_data_bytes: np.ndarray = total_data_bytes_80hz + total_data_bytes_800hz + total_data_bytes_8000hz
    total_data_gb = total_data_bytes / 1_000_000_000.0

    est_data_80hz = 2471 * 38 * (tss / 51.200)
    est_data_800hz = 2471 * 99 * (tss / 40.960)
    est_data_8000hz = 2471 * 5 * (tss / 32.768)
    est_data_total = (est_data_80hz + est_data_800hz + est_data_8000hz) / 1_000_000_000.0

    max_y = max(est_data_total)

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Lokahi: Estimated AML vs Actual AML")

    est_ax = ax[0]
    est_ax.plot(dts, est_data_total, label="Estimated AML Size")

    est_ax.set_ylim(ymax=max_y)
    est_ax.set_title("Estimated AML")
    est_ax.set_ylabel("Size GB")
    est_ax.legend()

    actual_ax = ax[1]
    actual_ax.plot(dts, sum_series(total_data_gb), label="Actual AML Size")

    actual_ax.set_ylim(ymax=max_y)
    actual_ax.set_title("Actual AML")
    actual_ax.set_ylabel("Size GB")
    actual_ax.legend()

    diff_ax = ax[2]
    diff_ax.plot(dts, est_data_total - sum_series(total_data_gb), label="Difference")

    diff_ax.set_ylim(ymax=max_y)
    diff_ax.set_title("Difference (Estimated - Actual)")
    diff_ax.set_xlabel("Time (UTC)")
    diff_ax.set_ylabel("Size GB")
    diff_ax.legend()

    # fig.show()
    fig.savefig("/Users/anthony/Development/dissertation/src/figures/lokahi_actual_aml_vs_est.png")

def plot_aml_vs_sim(dts: np.ndarray,
                    aligned_daily_metrics: np.ndarray,
                    aligned_sim_80: np.ndarray,
                    aligned_sim_800: np.ndarray,
                    aligned_sim_8000: np.ndarray):
    # Data

    total_data_bytes_80hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_80hz(), aligned_daily_metrics)))

    total_data_bytes_800hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_800hz(), aligned_daily_metrics)))

    total_data_bytes_8000hz: np.ndarray = np.array(
            list(map(lambda daily_metric: daily_metric.aml_size_bytes_8000hz(), aligned_daily_metrics)))

    total_data_bytes: np.ndarray = total_data_bytes_80hz + total_data_bytes_800hz + total_data_bytes_8000hz
    total_data_gb = sum_series(total_data_bytes / 1_000_000_000.0)

    # Sim data
    sim_80_bytes = np.array(list(map(lambda data: data.total_trends_b, aligned_sim_80))) * 38
    sim_800_bytes = np.array(list(map(lambda data: data.total_trends_b, aligned_sim_800))) * 99
    sim_8000_bytes = np.array(list(map(lambda data: data.total_trends_b, aligned_sim_8000))) * 5
    sim_total_bytes = sim_80_bytes + sim_800_bytes + sim_8000_bytes
    sim_total_gb = sim_total_bytes / 1_000_000_000.0

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Lokahi: Simulated AML vs Actual AML")

    sim_ax = ax[0]
    sim_ax.plot(dts, sim_total_gb, label="Simulated AML Size")

    # sim_ax.set_yscale("log")
    sim_ax.set_ylim(ymax=max(total_data_gb))
    sim_ax.legend()
    sim_ax.set_title("Simulated AML")
    sim_ax.set_ylabel("Size GB")

    actual_ax = ax[1]
    actual_ax.plot(dts[1:], total_data_gb, label="Actual AML Size")

    # actual_ax.set_yscale("log")
    actual_ax.set_ylim(ymax=max(total_data_gb))
    actual_ax.legend()
    actual_ax.set_title("Actual AML")
    actual_ax.set_ylabel("Size GB")

    diff_ax = ax[2]
    diff_ax.plot(dts[1:], sim_total_gb[1:] - total_data_gb, label="Difference")

    diff_ax.legend()
    # diff_ax.set_yscale("log")
    # diff_ax.set_ylim(ymax=max(total_data_gb))
    diff_ax.set_title("Difference (Simulated - Actual)")
    diff_ax.set_xlabel("Time (UTC)")
    diff_ax.set_ylabel("Size GB")

    # fig.show()
    fig.savefig("/Users/anthony/Development/dissertation/src/figures/lokahi_actual_aml_vs_sim.png")


def main():
    daily_metrics: List[DailyMetric] = load_daily_metrics("metrics.txt")
    sim_data_80: List[Data] = parse_file("sim_data_80.txt")
    sim_data_800: List[Data] = parse_file("sim_data_800.txt")
    sim_data_8000: List[Data] = parse_file("sim_data_8000.txt")

    first_metric_timestamp: int = daily_metrics[0].ts()

    series_specs: List[SeriesSpec] = [
        SeriesSpec(daily_metrics,
                   lambda daily_metric: daily_metric.dt(),
                   lambda daily_metric: daily_metric),
        SeriesSpec(sim_data_80,
                   lambda data: datetime.datetime.utcfromtimestamp(data.time + first_metric_timestamp),
                   lambda data: data),
        SeriesSpec(sim_data_800,
                   lambda data: datetime.datetime.utcfromtimestamp(data.time + first_metric_timestamp),
                   lambda data: data),
        SeriesSpec(sim_data_8000,
                   lambda data: datetime.datetime.utcfromtimestamp(data.time + first_metric_timestamp),
                   lambda data: data),
    ]

    aligned_data: List[Tuple[np.ndarray, np.ndarray]] = align_data_multi(series_specs)
    aligned_metric_data: Tuple[np.ndarray, np.ndarray] = aligned_data[0]
    aligned_sim_80_data: Tuple[np.ndarray, np.ndarray] = aligned_data[1]
    aligned_sim_800_data: Tuple[np.ndarray, np.ndarray] = aligned_data[2]
    aligned_sim_8000_data: Tuple[np.ndarray, np.ndarray] = aligned_data[3]

    aligned_dts: np.ndarray = aligned_metric_data[0]
    aligned_daily_metrics: np.ndarray = aligned_metric_data[1]
    aligned_sim_80: np.ndarray = aligned_sim_80_data[1]
    aligned_sim_800: np.ndarray = aligned_sim_800_data[1]
    aligned_sim_8000: np.ndarray = aligned_sim_8000_data[1]


    # plot_active_sensors(daily_metrics)
    # plot_iml(daily_metrics)
    # plot_aml(daily_metrics)

    # plot_iml_vs_est(daily_metrics)

    # plot_iml_vs_sim(aligned_dts,
    #                 aligned_daily_metrics,
    #                 aligned_sim_80,
    #                 aligned_sim_800,
    #                 aligned_sim_8000)

    # plot_aml_vs_est(daily_metrics)

    plot_aml_vs_sim(aligned_dts,
                    aligned_daily_metrics,
                    aligned_sim_80,
                    aligned_sim_800,
                    aligned_sim_8000)

if __name__ == "__main__":
    main()
