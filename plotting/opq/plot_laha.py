from typing import *

import matplotlib.pyplot as plt
import numpy as np


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


def plot_iml(data: List[Data]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    x = np.array(list(map(lambda d: d.time, data)))
    total_samples = np.array(list(map(lambda d: d.total_samples, data)))
    total_bytes = np.array(list(map(lambda d: d.total_samples_b, data)))
    total_mb = total_bytes / 1_000_000.0

    ax.plot(x, total_samples, label="IML Samples")
    ax.set_title("OPQ IML Single Sensor Data Growth 24 Hours")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("# Samples")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.axvline(900, 0, total_samples.max(), color="red", linestyle="--", label="IML TTL (15 Min)")

    ax_size: plt.Axes = ax.twinx()
    ax_size.plot(x, total_mb)
    ax_size.set_ylabel("IML Size MB")

    ax.legend()
    fig.show()

def plot_aml(data: List[Data]) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    x = np.array(list(map(lambda d: d.time, data)))
    total_measurements = np.array(list(map(lambda d: d.total_measurements, data)))
    total_measurements_b = np.array(list(map(lambda d: d.total_measurements_b, data)))
    total_measurements_mb = total_measurements_b / 1_000_000.0
    total_orphaned_measurements = np.array(list(map(lambda d: d.total_orphaned_measurements, data)))
    total_orphaned_measurements_b = np.array(list(map(lambda d: d.total_orphaned_measurements_b, data)))
    total_orphaned_measurements_mb = total_orphaned_measurements_b / 1_000_000.0
    total_event_measurements = np.array(list(map(lambda d: d.total_event_measurements, data)))
    total_event_measurements_b = np.array(list(map(lambda d: d.total_event_measurements_b, data)))
    total_event_measurements_mb = total_event_measurements_b / 1_000_000.0
    total_incident_measurements = np.array(list(map(lambda d: d.total_incident_measurements, data)))
    total_incident_measurements_b = np.array(list(map(lambda d: d.total_incident_measurements_b, data)))
    total_incident_measurements_mb = total_incident_measurements_b / 1_000_000.0

    total_trends = np.array(list(map(lambda d: d.total_trends, data)))
    total_trends_b = np.array(list(map(lambda d: d.total_trends_b, data)))
    total_trends_mb = total_trends_b / 1_000_000.0
    total_orphaned_trends = np.array(list(map(lambda d: d.total_orphaned_trends, data)))
    total_orphaned_trends_b = np.array(list(map(lambda d: d.total_orphaned_trends_b, data)))
    total_orphaned_trends_mb = total_orphaned_trends_b / 1_000_000.0
    total_event_trends = np.array(list(map(lambda d: d.total_event_trends, data)))
    total_event_trends_b = np.array(list(map(lambda d: d.total_event_trends_b, data)))
    total_event_trends_mb = total_event_trends_b / 1_000_000.0
    total_incident_trends = np.array(list(map(lambda d: d.total_incident_trends, data)))
    total_incident_trends_b = np.array(list(map(lambda d: d.total_incident_trends_b, data)))
    total_incident_trends_mb = total_incident_trends_b / 1_000_000.0

    seconds_in_day = 86400
    seconds_in_two_weeks = seconds_in_day * 14
    seconds_in_month = seconds_in_day * 30.4167
    seconds_in_year = seconds_in_month * 12
    seconds_in_2_years = seconds_in_year * 2

    # Measurements
    measurement_ax = ax[0]
    measurement_ax.plot(x, total_orphaned_measurements, label="Orphaned Measurements")
    measurement_ax.plot(x, total_event_measurements, label="Event Measurements")
    measurement_ax.plot(x, total_incident_measurements, label="Incident Measurements")
    measurement_ax.plot(x, total_measurements, label="Total Measurements")
    measurement_ax.axvline(seconds_in_day, 0, total_measurements.max(), color="red", linestyle="--", label="Measurements TTL (1 Day)")
    measurement_ax.axvline(seconds_in_month, 0, total_measurements.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    measurement_ax.axvline(seconds_in_year, 0, total_measurements.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")

    measurement_ax.set_title("OPQ AML Single Device Measurements")
    measurement_ax.set_ylabel("# Measurements")
    measurement_ax.legend()

    measurement_mb_ax: plt.Axes = measurement_ax.twinx()
    measurement_mb_ax.plot(x, total_measurements_mb)
    measurement_mb_ax.set_ylabel("Size MB")

    # Trends
    trend_ax = ax[1]
    trend_ax.plot(x, total_orphaned_trends, label="Orphaned Trends")
    trend_ax.plot(x, total_event_trends, label="Event Trends")
    trend_ax.plot(x, total_incident_trends, label="Incident Trends")
    trend_ax.plot(x, total_trends, label="Total Trends")
    trend_ax.axvline(seconds_in_two_weeks, 0, total_trends.max(), color="black", linestyle="--", label="Trends TTL (2 Weeks)")
    trend_ax.axvline(seconds_in_month, 0, total_trends.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    trend_ax.axvline(seconds_in_year, 0, total_trends.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")

    trend_ax.set_title("OPQ AML Single Device Trends")
    trend_ax.set_ylabel("# Trends")
    trend_ax.legend()

    trend_mb_ax: plt.Axes = trend_ax.twinx()
    trend_mb_ax.plot(x, total_trends_mb)
    trend_mb_ax.set_ylabel("Size MB")

    # AML
    aml_ax = ax[2]
    aml_ax.plot(x, total_measurements, label="AML Measurements")
    aml_ax.plot(x, total_trends, label="AML Trends")
    aml_ax.plot(x, total_measurements + total_trends, label="AML")
    aml_ax.axvline(seconds_in_day, 0, total_measurements.max() + total_trends.max(), color="red", linestyle="--", label="Measurements TTL (1 Day)")
    aml_ax.axvline(seconds_in_two_weeks, 0, total_measurements.max() + total_trends.max(), color="black", linestyle="--", label="Trends TTL (2 Weeks)")
    aml_ax.axvline(seconds_in_month, 0, total_measurements.max() + total_trends.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    aml_ax.axvline(seconds_in_year, 0, total_measurements.max() + total_trends.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")

    aml_ax.set_title("OPQ AML Single Device")
    aml_ax.set_ylabel("# AML Items")
    aml_ax.set_xlabel("Seconds")
    aml_ax.legend()

    aml_mb_ax: plt.Axes = aml_ax.twinx()
    aml_mb_ax.plot(x, total_measurements_mb + total_trends_mb)
    aml_mb_ax.set_ylabel("Size MB")
    aml_mb_ax.set_xscale("log")
    fig.show()


if __name__ == "__main__":
    iml_data = parse_file("./sim_data_iml.txt")
    data = parse_file('./sim_data.txt')

    print(f"len(iml_data)={len(iml_data)}")
    print(f"len(data)={len(data)}")

    # plot_iml(iml_data)
    plot_aml(data)
