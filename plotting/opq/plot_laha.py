from typing import *

import matplotlib.pyplot as plt
import numpy as np

seconds_in_day = 86400
seconds_in_two_weeks = seconds_in_day * 14
seconds_in_month = seconds_in_day * 30.4167
seconds_in_year = seconds_in_month * 12
seconds_in_2_years = seconds_in_year * 2


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
                 total_phenomena_measurements: int,
                 total_phenomena_measurements_b: int,
                 total_trends: int,
                 total_trends_b: int,
                 total_orphaned_trends: int,
                 total_orphaned_trends_b: int,
                 total_event_trends: int,
                 total_event_trends_b: int,
                 total_incident_trends: int,
                 total_incident_trends_b: int,
                 total_phenomena_trends: int,
                 total_phenomena_trends_b: int,
                 total_events: int,
                 total_events_b: int,
                 total_orphaned_events: int,
                 total_orphaned_events_b: int,
                 total_incident_events: int,
                 total_incident_events_b: int,
                 total_phenomena_events: int,
                 total_phenomena_events_b: int,
                 total_incidents: int,
                 total_incidents_b: int,
                 total_phenomena_incidents: int,
                 total_phenomena_incidents_b: int,
                 total_phenomena: int,
                 total_phenomena_b: int,
                 total_laha_b: int,
                 total_iml_b: int,
                 total_aml_b: int,
                 total_dl_b: int,
                 total_il_b: int,
                 total_pl_b: int):
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

        self.total_phenomena_measurements = total_phenomena_measurements
        self.total_phenomena_measurements_b = total_phenomena_measurements_b
        self.total_phenomena_trends = total_phenomena_trends
        self.total_phenomena_trends_b = total_phenomena_trends_b
        self.total_phenomena_events = total_phenomena_events
        self.total_phenomena_events_b = total_phenomena_events_b
        self.total_phenomena_incidents = total_phenomena_incidents
        self.total_phenomena_incidents_b = total_phenomena_incidents_b
        self.total_phenomena = total_phenomena
        self.total_phenomena_b = total_phenomena_b
        self.total_pl_b = total_pl_b

    @staticmethod
    def from_line(line: str) -> 'Data':
        split_line = line.split(",")
        as_ints = list(map(int, split_line))
        return Data(*as_ints)


def parse_file(path: str) -> List[Data]:
    with open(path, "r") as fin:
        lines = list(map(lambda line: line.strip(), fin.readlines()))
        return list(map(lambda line: Data.from_line(line), lines))


def plot_iml(data: List[Data], out_dir: str) -> None:
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
    fig.savefig(f"{out_dir}/sim_iml_opq.png")


def plot_aml(data: List[Data], out_dir: str) -> None:
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

    fig.suptitle("OPQ AML Single Device Data Growth 3 Years")

    # Measurements
    measurement_ax = ax[0]
    measurement_ax.plot(x, total_orphaned_measurements, label="Orphaned Measurements")
    measurement_ax.plot(x, total_event_measurements, label="Event Measurements")
    measurement_ax.plot(x, total_incident_measurements, label="Incident Measurements")
    measurement_ax.plot(x, total_measurements, label="Total Measurements")
    measurement_ax.axvline(seconds_in_day, 0, total_measurements.max(), color="red", linestyle="--",
                           label="Measurements TTL (1 Day)")
    measurement_ax.axvline(seconds_in_month, 0, total_measurements.max(), color="green", linestyle="--",
                           label="Events TTL (1 Month)")
    measurement_ax.axvline(seconds_in_year, 0, total_measurements.max(), color="blue", linestyle="--",
                           label="Incidents TTL (1 Year)")

    measurement_ax.set_title("Measurements")
    measurement_ax.set_ylabel("# Measurements")
    measurement_ax.legend()

    measurement_mb_ax: plt.Axes = measurement_ax.twinx()
    measurement_mb_ax.plot(x, total_measurements_mb, visible=False)
    measurement_mb_ax.set_ylabel("Size MB")

    # Trends
    trend_ax = ax[1]
    trend_ax.plot(x, total_orphaned_trends, label="Orphaned Trends")
    trend_ax.plot(x, total_event_trends, label="Event Trends")
    trend_ax.plot(x, total_incident_trends, label="Incident Trends")
    trend_ax.plot(x, total_trends, label="Total Trends")
    trend_ax.axvline(seconds_in_two_weeks, 0, total_trends.max(), color="black", linestyle="--",
                     label="Trends TTL (2 Weeks)")
    trend_ax.axvline(seconds_in_month, 0, total_trends.max(), color="green", linestyle="--",
                     label="Events TTL (1 Month)")
    trend_ax.axvline(seconds_in_year, 0, total_trends.max(), color="blue", linestyle="--",
                     label="Incidents TTL (1 Year)")

    trend_ax.axvline(seconds_in_year * 2, 0, total_trends.max(), color="blue", linestyle="--",
                     label="Phenomena TTL (2 Years)")

    trend_ax.set_title("Trends")
    trend_ax.set_ylabel("# Trends")
    trend_ax.legend()

    trend_mb_ax: plt.Axes = trend_ax.twinx()
    trend_mb_ax.plot(x, total_trends_mb, visible=False)
    trend_mb_ax.set_ylabel("Size MB")

    # AML
    aml_ax = ax[2]
    aml_ax.plot(x, total_measurements, label="AML Measurements")
    aml_ax.plot(x, total_trends, label="AML Trends")
    aml_ax.plot(x, total_measurements + total_trends, label="AML", color="red")
    aml_ax.axvline(seconds_in_day, 0, total_measurements.max() + total_trends.max(), color="red", linestyle="--",
                   label="Measurements TTL (1 Day)")
    aml_ax.axvline(seconds_in_two_weeks, 0, total_measurements.max() + total_trends.max(), color="black",
                   linestyle="--", label="Trends TTL (2 Weeks)")
    aml_ax.axvline(seconds_in_month, 0, total_measurements.max() + total_trends.max(), color="green", linestyle="--",
                   label="Events TTL (1 Month)")
    aml_ax.axvline(seconds_in_year, 0, total_measurements.max() + total_trends.max(), color="blue", linestyle="--",
                   label="Incidents TTL (1 Year)")

    aml_ax.set_title("AML")
    aml_ax.set_ylabel("# AML Items")
    aml_ax.set_xlabel("Seconds")
    aml_ax.legend()

    aml_mb_ax: plt.Axes = aml_ax.twinx()
    aml_mb_ax.plot(x, total_measurements_mb + total_trends_mb, visible=False)
    aml_mb_ax.set_ylabel("Size MB")
    aml_mb_ax.set_xscale("log")

    fig.savefig(f"{out_dir}/sim_aml_opq.png")


def plot_dl(data: List[Data], out_dir: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    x = np.array(list(map(lambda d: d.time, data)))
    total_events = np.array(list(map(lambda d: d.total_events, data)))
    total_events_mb = np.array(list(map(lambda d: d.total_events_b, data))) / 1_000_000.0
    total_orphaned_events = np.array(list(map(lambda d: d.total_orphaned_events, data)))
    total_orphaned_events_mb = np.array(list(map(lambda d: d.total_orphaned_events_b, data))) / 1_000_000.0
    total_incident_events = np.array(list(map(lambda d: d.total_incident_events, data)))
    total_incident_events_mb = np.array(list(map(lambda d: d.total_incident_events_b, data))) / 1_000_000.0

    ax.plot(x, total_events, label="Total Events")
    ax.plot(x, total_orphaned_events, label="Orphaned Events")
    ax.plot(x, total_incident_events, label="Incident Events")
    ax.axvline(seconds_in_month, 0, total_events.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax.axvline(seconds_in_year, 0, total_events.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")
    ax.axvline(seconds_in_year * 2, 0, total_events.max(), color="blue", linestyle="--", label="Phenomena TTL (2 Years)")

    ax_mb: plt.Axes = ax.twinx()
    ax_mb.plot(x, total_events_mb, visible=False)
    ax_mb.set_ylabel("Size MB")

    ax.set_title("OPQ DL Single Device Data Growth 3 Years")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("# Events")
    ax.set_xscale("log")
    ax.legend()

    fig.savefig(f"{out_dir}/sim_dl_opq.png")


def plot_il(data: List[Data], out_dir: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    x = np.array(list(map(lambda d: d.time, data)))
    total_incidents = np.array(list(map(lambda d: d.total_incidents, data)))
    total_incidents_mb = np.array(list(map(lambda d: d.total_incidents_b, data))) / 1_000_000.0

    ax.plot(x, total_incidents, label="Total Incidents")
    ax.axvline(seconds_in_year, 0, total_incidents.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")
    ax.axvline(seconds_in_year * 2, 0, total_incidents.max(), color="blue", linestyle="--", label="Phenomena TTL (2 Years)")

    ax.set_title("OPQ IL Single Device Data Growth 3 Years")
    ax.set_ylabel("# Incidents")
    ax.set_xlabel("Seconds")
    ax.set_xscale("log")

    ax_mb: plt.Axes = ax.twinx()
    ax_mb.plot(x, total_incidents_mb, visible=False)
    ax_mb.set_ylabel("Size MB")

    ax.legend()

    fig.savefig(f"{out_dir}/sim_il_opq.png")

def plot_pl(data: List[Data], out_dir: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    x = np.array(list(map(lambda d: d.time, data)))
    total_phenomena = np.array(list(map(lambda d: d.total_phenomena, data)))
    total_phenomena_mb = np.array(list(map(lambda d: d.total_phenomena_b, data))) / 1_000_000.0

    ax.plot(x, total_phenomena, label="Total Phenomena")
    ax.axvline(seconds_in_year * 2, 0, total_phenomena.max(), color="blue", linestyle="--", label="Phenomena TTL (2 Years)")

    ax.set_title("OPQ PL Single Device Data Growth 3 Years")
    ax.set_ylabel("# Phenomena")
    ax.set_xlabel("Seconds")
    ax.set_xscale("log")

    ax_mb: plt.Axes = ax.twinx()
    ax_mb.plot(x, total_phenomena_mb, visible=False)
    ax_mb.set_ylabel("Size MB")

    ax.legend()

    fig.show()
    fig.savefig(f"{out_dir}/sim_pl_opq.png")


def plot_laha(data: List[Data], out_dir: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    x = np.array(list(map(lambda d: d.time, data)))
    total_laha_mb = np.array(list(map(lambda d: d.total_laha_b, data))) / 1_000_000.0
    total_iml_mb = np.array(list(map(lambda d: d.total_iml_b, data))) / 1_000_000.0
    total_aml_mb = np.array(list(map(lambda d: d.total_aml_b, data))) / 1_000_000.0
    total_dl_mb = np.array(list(map(lambda d: d.total_dl_b, data))) / 1_000_000.0
    total_il_mb = np.array(list(map(lambda d: d.total_il_b, data))) / 1_000_000.0
    total_pl_mb = np.array(list(map(lambda d: d.total_pl_b, data))) / 1_000_000.0

    ax.plot(x, total_laha_mb, label="Total Laha")
    ax.plot(x, total_iml_mb, label="IML")
    ax.plot(x, total_aml_mb, label="AML")
    ax.plot(x, total_dl_mb, label="DL")
    ax.plot(x, total_il_mb, label="IL")
    ax.plot(x, total_pl_mb, label="PL")

    ax.axvline(seconds_in_day, 0, total_laha_mb.max(), color="red", linestyle="--", label="Measurements TTL (1 Day)")
    ax.axvline(seconds_in_two_weeks, 0, total_laha_mb.max(), color="black", linestyle="--",
               label="Trends TTL (2 Weeks)")
    ax.axvline(seconds_in_month, 0, total_laha_mb.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax.axvline(seconds_in_year, 0, total_laha_mb.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")
    ax.axvline(seconds_in_year * 2, 0, total_laha_mb.max(), color="blue", linestyle="--", label="Phenomena TTL (2 Years)")

    ax.set_title("OPQ Laha Single Device Data Growth 3 Years")
    ax.set_ylabel("Size MB")
    ax.set_xlabel("Seconds")
    ax.set_xscale("log")
    # ax.set_yscale("log")
    ax.legend()

    fig.show()
    fig.savefig(f"{out_dir}/sim_laha_opq.png")


if __name__ == "__main__":
    # iml_data = parse_file("./sim_data_iml.txt")
    # data = parse_file('./sim_data.txt')
    data = parse_file("/home/opq/scrap/sim_data_opq.txt")

    # print(f"len(iml_data)={len(iml_data)}")
    # print(f"len(data)={len(data)}")

    out_dir = "/home/opq/Documents/anthony/dissertation/src/figures"

    # plot_iml(iml_data, out_dir)
    plot_aml(data, out_dir)
    plot_dl(data, out_dir)
    plot_il(data, out_dir)
    plot_laha(data, out_dir)
    plot_pl(data, out_dir)
