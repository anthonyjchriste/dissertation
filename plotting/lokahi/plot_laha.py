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


def plot_iml(data_80: List[Data],
             data_800: List[Data],
             data_8000: List[Data],
             out_dir: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    x_80 = np.array(list(map(lambda d: d.time, data_80)))
    total_samples_80 = np.array(list(map(lambda d: d.total_samples, data_80)))
    total_mb_80 = np.array(list(map(lambda d: d.total_samples_b, data_80))) / 1_000_000.0

    x_800 = np.array(list(map(lambda d: d.time, data_800)))
    total_samples_800 = np.array(list(map(lambda d: d.total_samples, data_800)))
    total_mb_800 = np.array(list(map(lambda d: d.total_samples_b, data_800))) / 1_000_000.0

    x_8000 = np.array(list(map(lambda d: d.time, data_8000)))
    total_samples_8000 = np.array(list(map(lambda d: d.total_samples, data_8000)))
    total_mb_8000 = np.array(list(map(lambda d: d.total_samples_b, data_8000))) / 1_000_000.0

    ax.plot(x_80, total_samples_80, label="IML Samples (80 Hz)")
    ax.plot(x_800, total_samples_800, label="IML Samples (800 Hz)")
    ax.plot(x_8000, total_samples_8000, label="IML Samples (8000 Hz)")


    ax.set_title("Lokahi IML Single Sensor Data Growth 24 Hours")
    ax.set_xlabel("Seconds")
    ax.set_ylabel("# Samples")


    ax.ticklabel_format(useOffset=False, style="plain")
    ax.axvline(900, 0, total_samples_8000.max(), color="red", linestyle="--", label="IML TTL (15 Min)")

    ax_size: plt.Axes = ax.twinx()
    ax_size.plot(x_8000, total_mb_8000, visible=False)
    ax_size.set_ylabel("Size MB")

    ax.legend()
    # fig.show()
    fig.savefig(f"{out_dir}/sim_iml_lokahi.png")


def plot_aml(data_80: List[Data],
             data_800: List[Data],
             data_8000: List[Data],
             out_dir: str) -> None:
    fig, ax = plt.subplots(4, 1, figsize=(16, 9), sharex="all", sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    x_80 = np.array(list(map(lambda d: d.time, data_80)))
    x_800 = np.array(list(map(lambda d: d.time, data_800)))
    x_8000 = np.array(list(map(lambda d: d.time, data_8000)))

    total_trends_80 = np.array(list(map(lambda d: d.total_trends, data_80)))
    total_trends_mb_80 = np.array(list(map(lambda d: d.total_trends_b, data_80))) / 1_000_000.0
    total_orphaned_trends_80 = np.array(list(map(lambda d: d.total_orphaned_trends, data_80)))
    total_orphaned_trends_mb_80 = np.array(list(map(lambda d: d.total_orphaned_trends_b, data_80))) / 1_000_000.0
    total_event_trends_80 = np.array(list(map(lambda d: d.total_event_trends, data_80)))
    total_event_trends_mb_80 = np.array(list(map(lambda d: d.total_event_trends_b, data_80))) / 1_000_000.0
    total_incident_trends_80 = np.array(list(map(lambda d: d.total_incident_trends, data_80)))
    total_incident_trends_mb_80 = np.array(list(map(lambda d: d.total_incident_trends_b, data_80))) / 1_000_000.0

    total_trends_800 = np.array(list(map(lambda d: d.total_trends, data_800)))
    total_trends_mb_800 = np.array(list(map(lambda d: d.total_trends_b, data_800))) / 1_000_000.0
    total_orphaned_trends_800 = np.array(list(map(lambda d: d.total_orphaned_trends, data_800)))
    total_orphaned_trends_mb_800 = np.array(list(map(lambda d: d.total_orphaned_trends_b, data_800))) / 1_000_000.0
    total_event_trends_800 = np.array(list(map(lambda d: d.total_event_trends, data_800)))
    total_event_trends_mb_800 = np.array(list(map(lambda d: d.total_event_trends_b, data_800))) / 1_000_000.0
    total_incident_trends_800 = np.array(list(map(lambda d: d.total_incident_trends, data_800)))
    total_incident_trends_mb_800 = np.array(list(map(lambda d: d.total_incident_trends_b, data_800))) / 1_000_000.0

    total_trends_8000 = np.array(list(map(lambda d: d.total_trends, data_8000)))
    total_trends_mb_8000 = np.array(list(map(lambda d: d.total_trends_b, data_8000))) / 1_000_000.0
    total_orphaned_trends_8000 = np.array(list(map(lambda d: d.total_orphaned_trends, data_8000)))
    total_orphaned_trends_mb_8000 = np.array(list(map(lambda d: d.total_orphaned_trends_b, data_8000))) / 1_000_000.0
    total_event_trends_8000 = np.array(list(map(lambda d: d.total_event_trends, data_8000)))
    total_event_trends_mb_8000 = np.array(list(map(lambda d: d.total_event_trends_b, data_8000))) / 1_000_000.0
    total_incident_trends_8000 = np.array(list(map(lambda d: d.total_incident_trends, data_8000)))
    total_incident_trends_mb_8000 = np.array(list(map(lambda d: d.total_incident_trends_b, data_8000))) / 1_000_000.0

    fig.suptitle("Lokahi AML Single Device Growth 3 Years")

    # Trends 80
    ax_80 = ax[0]
    ax_80.plot(x_80, total_trends_80, label="Total Trends")
    ax_80.plot(x_80, total_orphaned_trends_80, label="Orphaned Trends")
    ax_80.plot(x_80, total_event_trends_80, label="Event Trends")
    ax_80.plot(x_80, total_incident_trends_80, label="Incident Trends")
    ax_80.axvline(seconds_in_two_weeks, 0, total_trends_80.max(), color="black", linestyle="--", label="Trends TTL (2 Weeks)")
    ax_80.axvline(seconds_in_month, 0, total_trends_80.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_80.axvline(seconds_in_year, 0, total_trends_80.max(), color="blue", linestyle="--",label="Incidents TTL (1 Year)")
    ax_80.set_title("Trends @ 80Hz")
    ax_80.set_ylabel("# Trends")
    ax_80.legend()
    ax_mb_80: plt.Axes = ax_80.twinx()
    ax_mb_80.plot(x_80, total_trends_mb_80, visible=False)
    ax_mb_80.set_ylabel("Size MB")

    # Trends 800
    ax_800 = ax[1]
    ax_800.plot(x_800, total_trends_800, label="Total Trends")
    ax_800.plot(x_800, total_orphaned_trends_800, label="Orphaned Trends")
    ax_800.plot(x_800, total_event_trends_800, label="Event Trends")
    ax_800.plot(x_800, total_incident_trends_800, label="Incident Trends")
    ax_800.axvline(seconds_in_two_weeks, 0, total_trends_800.max(), color="black", linestyle="--", label="Trends TTL (2 Weeks)")
    ax_800.axvline(seconds_in_month, 0, total_trends_800.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_800.axvline(seconds_in_year, 0, total_trends_800.max(), color="blue", linestyle="--",label="Incidents TTL (1 Year)")
    ax_800.set_title("Trends @ 800Hz")
    ax_800.set_ylabel("# Trends")
    ax_800.legend()
    ax_mb_800: plt.Axes = ax_800.twinx()
    ax_mb_800.plot(x_800, total_trends_mb_800, visible=False)
    ax_mb_800.set_ylabel("Size MB")

    # Trends 8000
    ax_8000 = ax[2]
    ax_8000.plot(x_8000, total_trends_8000, label="Total Trends")
    ax_8000.plot(x_8000, total_orphaned_trends_8000, label="Orphaned Trends")
    ax_8000.plot(x_8000, total_event_trends_8000, label="Event Trends")
    ax_8000.plot(x_8000, total_incident_trends_8000, label="Incident Trends")
    ax_8000.axvline(seconds_in_two_weeks, 0, total_trends_8000.max(), color="black", linestyle="--", label="Trends TTL (2 Weeks)")
    ax_8000.axvline(seconds_in_month, 0, total_trends_8000.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_8000.axvline(seconds_in_year, 0, total_trends_8000.max(), color="blue", linestyle="--",label="Incidents TTL (1 Year)")
    ax_8000.set_title("Trends @ 8000Hz")
    ax_8000.set_ylabel("# Trends")
    ax_8000.legend()
    ax_mb_8000: plt.Axes = ax_8000.twinx()
    ax_mb_8000.plot(x_8000, total_trends_mb_8000, visible=False)
    ax_mb_8000.set_ylabel("Size MB")

    # AML
    ax_aml = ax[3]
    ax_aml.plot(x_80, total_trends_80, label="AML (80 Hz)")
    ax_aml.plot(x_800, total_trends_800, label="AML (800 Hz)")
    ax_aml.plot(x_8000, total_trends_8000, label="AML (8000 Hz)")
    ax_aml.axvline(seconds_in_two_weeks, 0, total_trends_8000.max(), color="black", linestyle="--", label="Trends TTL (2 Weeks)")
    ax_aml.axvline(seconds_in_month, 0, total_trends_8000.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_aml.axvline(seconds_in_year, 0, total_trends_8000.max(), color="blue", linestyle="--",label="Incidents TTL (1 Year)")
    ax_mb_aml: plt.Axes = ax_aml.twinx()
    ax_mb_aml.plot(x_8000, total_trends_mb_8000, visible=False)
    ax_mb_aml.set_ylabel("Size MB")

    ax_aml.set_xscale("log")
    ax_aml.set_title("AML Total")
    ax_aml.set_ylabel("# Trends")
    ax_aml.set_xlabel("Seconds")
    ax_aml.legend()

    # fig.show()
    fig.savefig(f"{out_dir}/sim_aml_lokahi.png")

def plot_dl(data_80: List[Data],
            data_800: List[Data],
            data_8000: List[Data],
            out_dir: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    x_80 = np.array(list(map(lambda d: d.time, data_80)))
    x_800 = np.array(list(map(lambda d: d.time, data_800)))
    x_8000 = np.array(list(map(lambda d: d.time, data_8000)))

    total_events_80 = np.array(list(map(lambda d: d.total_events, data_80)))
    total_events_mb_80 = np.array(list(map(lambda d: d.total_events_b, data_80))) / 1_000_000.0
    total_events_800 = np.array(list(map(lambda d: d.total_events, data_800)))
    total_events_mb_800 = np.array(list(map(lambda d: d.total_events_b, data_800))) / 1_000_000.0
    total_events_8000 = np.array(list(map(lambda d: d.total_events, data_8000)))
    total_events_mb_8000 = np.array(list(map(lambda d: d.total_events_b, data_8000))) / 1_000_000.0

    fig.suptitle("Lokahi DL Single Device Data Growth 3 Years")

    # 80
    ax_80 = ax[0]
    ax_80.plot(x_80, total_events_80, label="Events")
    ax_mb_80: plt.Axes = ax_80.twinx()
    ax_mb_80.plot(x_80, total_events_mb_80, visible=False)
    ax_80.axvline(seconds_in_month, 0, total_events_8000.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_mb_80.set_ylabel("Size MB")
    ax_80.set_title("80 Hz")
    ax_80.set_ylabel("# Events")
    ax_80.legend()

    # 800
    ax_800 = ax[1]
    ax_800.plot(x_800, total_events_800, label="Events")
    ax_mb_800: plt.Axes = ax_800.twinx()
    ax_mb_800.plot(x_800, total_events_mb_800, visible=False)
    ax_800.axvline(seconds_in_month, 0, total_events_8000.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_mb_800.set_ylabel("Size MB")
    ax_800.set_title("800 Hz")
    ax_800.set_ylabel("# Events")
    ax_800.legend()

    # 8000
    ax_8000 = ax[2]
    ax_8000.plot(x_8000, total_events_8000, label="Events")
    ax_mb_8000: plt.Axes = ax_8000.twinx()
    ax_mb_8000.plot(x_8000, total_events_mb_8000, visible=False)
    ax_8000.axvline(seconds_in_month, 0, total_events_8000.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_mb_8000.set_ylabel("Size MB")
    ax_8000.set_title("8000 Hz")
    ax_8000.set_ylabel("# Events")
    ax_8000.set_xscale("log")
    ax_8000.set_xlabel("Seconds")
    ax_8000.legend()

    # fig.show()
    fig.savefig(f"{out_dir}/sim_dl_lokahi.png")

def plot_il(data_80: List[Data],
            data_800: List[Data],
            data_8000: List[Data],
            out_dir: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    x_80 = np.array(list(map(lambda d: d.time, data_80)))
    x_800 = np.array(list(map(lambda d: d.time, data_800)))
    x_8000 = np.array(list(map(lambda d: d.time, data_8000)))

    total_incidents_80 = np.array(list(map(lambda d: d.total_incidents, data_80)))
    total_incidents_mb_80 = np.array(list(map(lambda d: d.total_incidents_b, data_80))) / 1_000_000.0
    total_incidents_800 = np.array(list(map(lambda d: d.total_incidents, data_800)))
    total_incidents_mb_800 = np.array(list(map(lambda d: d.total_incidents_b, data_800))) / 1_000_000.0
    total_incidents_8000 = np.array(list(map(lambda d: d.total_incidents, data_8000)))
    total_incidents_mb_8000 = np.array(list(map(lambda d: d.total_incidents_b, data_8000))) / 1_000_000.0

    fig.suptitle("Lokahi IL Single Device Data Growth 3 Years")

    # 80
    ax_80 = ax[0]
    ax_80.plot(x_80, total_incidents_80, label="Incidents")
    ax_mb_80: plt.Axes = ax_80.twinx()
    ax_mb_80.plot(x_80, total_incidents_mb_80, visible=False)
    ax_80.axvline(seconds_in_year, 0, total_incidents_80.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")
    ax_mb_80.set_ylabel("Size MB")
    ax_80.set_title("80 Hz")
    ax_80.set_ylabel("# Incidents")
    ax_80.legend()

    # 800
    ax_800 = ax[1]
    ax_800.plot(x_800, total_incidents_800, label="Incidents")
    ax_mb_800: plt.Axes = ax_800.twinx()
    ax_mb_800.plot(x_800, total_incidents_mb_800, visible=False)
    ax_800.axvline(seconds_in_year, 0, total_incidents_800.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")
    ax_mb_800.set_ylabel("Size MB")
    ax_800.set_title("800 Hz")
    ax_800.set_ylabel("# Incidents")
    ax_800.legend()

    # 8000
    ax_8000 = ax[2]
    ax_8000.plot(x_8000, total_incidents_8000, label="Incidents")
    ax_mb_8000: plt.Axes = ax_8000.twinx()
    ax_mb_8000.plot(x_8000, total_incidents_mb_8000, visible=False)
    ax_8000.axvline(seconds_in_year, 0, total_incidents_8000.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")
    ax_mb_8000.set_ylabel("Size MB")
    ax_8000.set_title("8000 Hz")
    ax_8000.set_ylabel("# Incidents")
    ax_8000.set_xscale("log")
    ax_8000.set_xlabel("Seconds")
    ax_8000.legend()

    # fig.show()
    fig.savefig(f"{out_dir}/sim_il_lokahi.png")


def plot_laha(data_80: List[Data],
              data_800: List[Data],
              data_8000: List[Data],
              out_dir: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    x_80 = np.array(list(map(lambda d: d.time, data_80)))
    total_laha_mb_80 = np.array(list(map(lambda d: d.total_laha_b, data_80))) / 1_000_000.0
    total_iml_mb_80 = np.array(list(map(lambda d: d.total_iml_b, data_80))) / 1_000_000.0
    total_aml_mb_80 = np.array(list(map(lambda d: d.total_aml_b, data_80))) / 1_000_000.0
    total_dl_mb_80 = np.array(list(map(lambda d: d.total_dl_b, data_80))) / 1_000_000.0
    total_il_mb_80 = np.array(list(map(lambda d: d.total_il_b, data_80))) / 1_000_000.0

    x_800 = np.array(list(map(lambda d: d.time, data_800)))
    total_laha_mb_800 = np.array(list(map(lambda d: d.total_laha_b, data_800))) / 1_000_000.0
    total_iml_mb_800 = np.array(list(map(lambda d: d.total_iml_b, data_800))) / 1_000_000.0
    total_aml_mb_800 = np.array(list(map(lambda d: d.total_aml_b, data_800))) / 1_000_000.0
    total_dl_mb_800 = np.array(list(map(lambda d: d.total_dl_b, data_800))) / 1_000_000.0
    total_il_mb_800 = np.array(list(map(lambda d: d.total_il_b, data_800))) / 1_000_000.0

    x_8000 = np.array(list(map(lambda d: d.time, data_8000)))
    total_laha_mb_8000 = np.array(list(map(lambda d: d.total_laha_b, data_8000))) / 1_000_000.0
    total_iml_mb_8000 = np.array(list(map(lambda d: d.total_iml_b, data_8000))) / 1_000_000.0
    total_aml_mb_8000 = np.array(list(map(lambda d: d.total_aml_b, data_8000))) / 1_000_000.0
    total_dl_mb_8000 = np.array(list(map(lambda d: d.total_dl_b, data_8000))) / 1_000_000.0
    total_il_mb_8000 = np.array(list(map(lambda d: d.total_il_b, data_8000))) / 1_000_000.0

    fig.suptitle("Lokahi Laha Data Growth Single Device 3 Years")

    # 80
    ax_80 = ax[0]
    ax_80.plot(x_80, total_laha_mb_80, label="Total Laha")
    ax_80.plot(x_80, total_iml_mb_80, label="IML")
    ax_80.plot(x_80, total_aml_mb_80, label="AML")
    ax_80.plot(x_80, total_dl_mb_80, label="DL")
    ax_80.plot(x_80, total_il_mb_80, label="IL")

    ax_80.axvline(seconds_in_day, 0, total_laha_mb_8000.max(), color="red", linestyle="--", label="Measurements TTL (1 Day)")
    ax_80.axvline(seconds_in_two_weeks, 0, total_laha_mb_8000.max(), color="black", linestyle="--",
               label="Trends TTL (2 Weeks)")
    ax_80.axvline(seconds_in_month, 0, total_laha_mb_8000.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_80.axvline(seconds_in_year, 0, total_laha_mb_8000.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")

    ax_80.set_title("Lokahi Laha Single Device Data Growth 3 Years")
    ax_80.set_ylabel("Size MB")
    ax_80.legend()
    ax_80.set_title("80 HZ")

    # 800
    ax_800 = ax[1]
    ax_800.plot(x_800, total_laha_mb_800, label="Total Laha")
    ax_800.plot(x_800, total_iml_mb_800, label="IML")
    ax_800.plot(x_800, total_aml_mb_800, label="AML")
    ax_800.plot(x_800, total_dl_mb_800, label="DL")
    ax_800.plot(x_800, total_il_mb_800, label="IL")

    ax_800.axvline(seconds_in_day, 0, total_laha_mb_800.max(), color="red", linestyle="--", label="Measurements TTL (1 Day)")
    ax_800.axvline(seconds_in_two_weeks, 0, total_laha_mb_800.max(), color="black", linestyle="--",
                  label="Trends TTL (2 Weeks)")
    ax_800.axvline(seconds_in_month, 0, total_laha_mb_800.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_800.axvline(seconds_in_year, 0, total_laha_mb_800.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")

    ax_800.set_title("Lokahi Laha Single Device Data Growth 3 Years")
    ax_800.set_ylabel("Size MB")
    ax_800.legend()
    ax_800.set_title("800 HZ")

    # 8000
    ax_8000 = ax[2]
    ax_8000.plot(x_8000, total_laha_mb_8000, label="Total Laha")
    ax_8000.plot(x_8000, total_iml_mb_8000, label="IML")
    ax_8000.plot(x_8000, total_aml_mb_8000, label="AML")
    ax_8000.plot(x_8000, total_dl_mb_8000, label="DL")
    ax_8000.plot(x_8000, total_il_mb_8000, label="IL")

    ax_8000.axvline(seconds_in_day, 0, total_laha_mb_8000.max(), color="red", linestyle="--", label="Measurements TTL (1 Day)")
    ax_8000.axvline(seconds_in_two_weeks, 0, total_laha_mb_8000.max(), color="black", linestyle="--",
                   label="Trends TTL (2 Weeks)")
    ax_8000.axvline(seconds_in_month, 0, total_laha_mb_8000.max(), color="green", linestyle="--", label="Events TTL (1 Month)")
    ax_8000.axvline(seconds_in_year, 0, total_laha_mb_8000.max(), color="blue", linestyle="--", label="Incidents TTL (1 Year)")

    ax_8000.set_title("Lokahi Laha Single Device Data Growth 3 Years")
    ax_8000.set_ylabel("Size MB")
    ax_8000.legend()
    ax_8000.set_title("8000 HZ")
    ax_8000.set_xscale("log")



    fig.savefig(f"{out_dir}/sim_laha_lokahi.png")
    # fig.show()


if __name__ == "__main__":
    iml_data_80 = parse_file("./sim_data_80_iml.txt")
    iml_data_800 = parse_file("./sim_data_800_iml.txt")
    iml_data_8000 = parse_file("./sim_data_8000_iml.txt")
    data_80 = parse_file('./sim_data_80.txt')
    data_800 = parse_file('./sim_data_800.txt')
    data_8000 = parse_file('./sim_data_8000.txt')

    # print(f"len(iml_data)={len(iml_data)}")
    # print(f"len(data)={len(data)}")

    out_dir = "/Users/anthony/Development/dissertation/src/figures"

    plot_iml(iml_data_80, iml_data_800, iml_data_8000, out_dir)
    plot_aml(data_80, data_800, data_8000, out_dir)
    plot_dl(data_80, data_800, data_8000, out_dir)
    plot_il(data_80, data_800, data_8000, out_dir)
    plot_laha(data_80, data_800, data_8000, out_dir)
