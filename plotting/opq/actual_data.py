from dataclasses import dataclass
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, TypeVar, Callable, Set
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database

DB: str = "opq"
COLL: str = "laha_stats"

S_IN_DAY = 86_400
S_IN_YEAR = 31_540_000

seconds_in_day = 86400
seconds_in_two_weeks = seconds_in_day * 14
seconds_in_month = seconds_in_day * 30.4167
seconds_in_year = seconds_in_month * 12
seconds_in_2_years = seconds_in_year * 2

A = TypeVar("A")
B = TypeVar("B")
C = TypeVar("C")
D = TypeVar("D")


def bin_dt_by_min(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0, 0, tzinfo=datetime.timezone.utc)


def align_data(series_a: List,
               series_b: List,
               dt_func_a: Callable,
               dt_func_b: Callable,
               val_func_a: Callable,
               val_func_b: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a_dts: List[datetime.datetime] = list(map(dt_func_a, series_a))
    b_dts: List[datetime.datetime] = list(map(dt_func_b, series_b))
    a_vals: List = list(map(val_func_a, series_a))
    b_vals: List = list(map(val_func_b, series_b))
    a_binned_dts: List[datetime.datetime] = list(map(bin_dt_by_min, a_dts))
    b_binned_dts: List[datetime.datetime] = list(map(bin_dt_by_min, b_dts))

    intersecting_dts: Set[datetime.datetime] = set(a_binned_dts).intersection(set(b_binned_dts))

    resulting_dts_a: List[datetime.datetime] = []
    resulting_dts_b: List[datetime.datetime] = []
    resulting_a_vals: List = []
    resulting_b_vals: List = []

    already_seen_a_dts: Set[datetime.datetime] = set()
    already_seen_b_dts: Set[datetime.datetime] = set()

    for i in range(len(series_a)):
        dt = a_binned_dts[i]
        if dt in intersecting_dts and dt not in already_seen_a_dts:
            resulting_dts_a.append(dt)
            resulting_a_vals.append(a_vals[i])
            already_seen_a_dts.add(dt)

    for i in range(len(series_b)):
        dt = b_binned_dts[i]
        if dt in intersecting_dts and dt not in already_seen_b_dts:
            resulting_dts_b.append(dt)
            resulting_b_vals.append(b_vals[i])
            already_seen_b_dts.add(dt)

    return np.array(resulting_dts_a), \
           np.array(resulting_a_vals), \
           np.array(resulting_dts_b), \
           np.array(resulting_b_vals)


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
class PluginStat:
    name: str
    messages_received: int
    messages_published: int
    bytes_received: int
    bytes_published: int

    @staticmethod
    def from_doc(name: str, doc: Dict[str, int]) -> 'PluginStat':
        return PluginStat(
                name,
                doc["messages_received"],
                doc["messages_published"],
                doc["bytes_received"],
                doc["bytes_published"]
        )


@dataclass
class SystemStat:
    min: float
    max: float
    mean: float
    var: float
    cnt: int
    start_timestamp_s: int
    end_timestamp_s: int

    @staticmethod
    def from_doc(doc: Dict[str, Union[float, int]]) -> 'SystemStat':
        return SystemStat(
                doc["min"],
                doc["max"],
                doc["mean"],
                doc["var"],
                doc["cnt"],
                doc["start_timestamp_s"],
                doc["end_timestamp_s"]
        )


@dataclass
class SystemStats:
    cpu_load_percent: SystemStat
    memory_use_bytes: SystemStat
    disk_use_bytes: SystemStat

    @staticmethod
    def from_doc(doc: Dict[str, Dict[str, Union[float, int]]]) -> 'SystemStats':
        return SystemStats(
                SystemStat.from_doc(doc["cpu_load_percent"]),
                SystemStat.from_doc(doc["memory_use_bytes"]),
                SystemStat.from_doc(doc["disk_use_bytes"])
        )


@dataclass
class LahaMetric:
    name: str
    ttl: int
    count: int
    size_bytes: int

    @staticmethod
    def from_doc(name: str, doc: Dict[str, int]) -> 'LahaMetric':
        return LahaMetric(
                name,
                doc["ttl"] if "ttl" in doc else -1,
                doc["count"] if "count" in doc else -1,
                doc["size_bytes"] if "size_bytes" in doc else -1
        )


@dataclass
class GcStats:
    samples: int
    measurements: int
    trends: int
    events: int
    incidents: int
    phenomena: int

    @staticmethod
    def from_doc(doc: Dict[str, int]) -> 'GcStats':
        return GcStats(
                doc["samples"],
                doc["measurements"],
                doc["trends"],
                doc["events"],
                doc["incidents"],
                0
        )


@dataclass
class BoxTriggeringThreshold:
    box_id: str
    ref_f: int
    ref_v: int
    threshold_percent_f_low: float
    threshold_percent_f_high: float
    threshold_percent_v_low: float
    threshold_percent_v_high: float
    threshold_percent_thd_high: float

    @staticmethod
    def from_doc(doc: Dict[str, Union[str, int, float]]) -> 'BoxTriggeringThreshold':
        return BoxTriggeringThreshold(
                doc["box_id"],
                doc["ref_f"],
                doc["ref_v"],
                doc["threshold_percent_f_low"],
                doc["threshold_percent_f_high"],
                doc["threshold_percent_v_low"],
                doc["threshold_percent_v_high"],
                doc["threshold_percent_thd_high"]
        )


@dataclass
class BoxMeasurementRate:
    box_id: str
    measurement_rate: int

    @staticmethod
    def from_doc(doc: Dict[str, Union[str, int]]) -> 'BoxMeasurementRate':
        return BoxMeasurementRate(
                doc["box_id"],
                doc["measurement_rate"]
        )


@dataclass
class LahaStats:
    laha_metrics: List[LahaMetric]
    gc_stats: GcStats
    active_devices: int
    box_triggering_thresholds: List[BoxTriggeringThreshold]
    box_measurement_rates: List[BoxMeasurementRate]

    @staticmethod
    def from_doc(doc: Dict[str, Any]) -> 'LahaStats':
        laha_metrics: List[LahaMetric] = [
            LahaMetric.from_doc("box_samples", doc["instantaneous_measurements_stats"]["box_samples"]),
            LahaMetric.from_doc("measurements", doc["aggregate_measurements_stats"]["measurements"]),
            LahaMetric.from_doc("trends", doc["aggregate_measurements_stats"]["trends"]),
            LahaMetric.from_doc("events", doc["detections_stats"]["events"]),
            LahaMetric.from_doc("incidents", doc["incidents_stats"]["incidents"]),
            LahaMetric.from_doc("phenomena", doc["phenomena_stats"]["phenomena"])
        ]

        gc_stats: GcStats = GcStats.from_doc(doc["gc_stats"])
        active_devices: int = doc["active_devices"]
        box_triggering_thresholds: List[BoxTriggeringThreshold] = []
        for box_triggering_threshold in doc["box_triggering_thresholds"]:
            box_triggering_thresholds.append(BoxTriggeringThreshold.from_doc(box_triggering_threshold))

        box_measurement_rates: List[BoxMeasurementRate] = []
        for box_measurement_rate in doc["box_measurement_rates"]:
            box_measurement_rates.append(BoxMeasurementRate.from_doc(box_measurement_rate))

        return LahaStats(
                laha_metrics,
                gc_stats,
                active_devices,
                box_triggering_thresholds,
                box_measurement_rates
        )


@dataclass
class LahaStat:
    timestamp_s: int
    plugin_stats: List[PluginStat]
    system_stats: SystemStats
    laha_stats: LahaStats

    @staticmethod
    def from_dict(doc: Dict) -> 'LahaStat':
        plugin_stats: List[PluginStat] = []

        dict_plugin_stats: Dict[str, Dict[str, int]] = doc["plugin_stats"]

        for plugin_name, plugin_dict in dict_plugin_stats.items():
            plugin_stats.append(PluginStat.from_doc(plugin_name, plugin_dict))

        dict_system_stats: Dict[str, Dict[str, Union[float, int]]] = doc["system_stats"]
        system_stats: SystemStats = SystemStats.from_doc(dict_system_stats)

        return LahaStat(doc["timestamp_s"],
                        plugin_stats,
                        system_stats,
                        LahaStats.from_doc(doc["laha_stats"]))


def map_laha_metric(laha_stat: LahaStat, name: str) -> Optional[LahaMetric]:
    for laha_metric in laha_stat.laha_stats.laha_metrics:
        if laha_metric.name == name:
            return laha_metric

    return None


def get_laha_stats(mongo_client: pymongo.MongoClient) -> List[LahaStat]:
    db: pymongo.database.Database = mongo_client[DB]
    coll: pymongo.collection.Collection = db[COLL]

    query = {"timestamp_s": {"$gt": 1568937600}}  # Sept 20 is the latest schema update and most complete stats
    projection = {"_id": False}

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    docs: List[Dict] = list(cursor)

    return list(map(LahaStat.from_dict, docs))


def save_laha_stats(to_pickle: List[LahaStat], path: str) -> None:
    with open(path, "wb") as fout:
        pickle.dump(to_pickle, fout)


def load_laha_stats(path: str) -> List[LahaStat]:
    with open(path, "rb") as fin:
        return pickle.load(fin)


def correct_counts(counts: np.ndarray) -> np.ndarray:
    diffs = np.diff(counts)
    diffs[np.where(diffs < 0)] = 0

    corrected_counts: List[int] = []
    for i in range(len(diffs)):
        corrected_counts.append(diffs[0:i].sum())

    return np.array(corrected_counts)


def plot_iml(laha_stat_dts: np.ndarray,
             active_devices: np.ndarray,
             laha_iml: np.ndarray,
             out_dir: str):
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.plot(laha_stat_dts, laha_iml, color="blue", label="IML Size")
    ax.set_ylabel("Size MB")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Actual IML (OPQ)")
    ax.legend(loc="upper left")

    ax_active = ax.twinx()
    ax_active.plot(laha_stat_dts, active_devices, visible=False)
    ax_active.set_ylabel("Active OPQ Boxes")

    fig.show()
    fig.savefig(f"{out_dir}/actual_iml_opq.png")


def plot_aml(laha_stat_dts: np.ndarray,
             measurements_gb: np.ndarray,
             trends_gb: np.ndarray,
             total_gb: np.ndarray,
             measurements_cnt: np.ndarray,
             trends_cnt: np.ndarray,
             total_cnt: np.ndarray,
             measurements_gc: np.ndarray,
             trends_gc: np.ndarray,
             total_gc: np.ndarray,
             active_devices: np.ndarray,
             out_dir: str) -> None:
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Laha AML (OPQ)")

    # Size
    ax_size = ax[0]
    ax_size.plot(laha_stat_dts, measurements_gb, label="Measurements Size GB", color="blue")
    ax_size.plot(laha_stat_dts, trends_gb, label="Trends Size GB", color="green")
    ax_size.plot(laha_stat_dts, total_gb, label="AML Total GB", color="red")
    ax_size.set_ylabel("Size GB")
    ax_size.set_title("Actual AML Size")
    ax_size.legend(loc="upper left")

    # cnt
    ax_cnt = ax_size.twinx()
    ax_cnt.plot(laha_stat_dts, measurements_cnt, label="Measurements Count", color="blue", linestyle="--")
    ax_cnt.plot(laha_stat_dts, trends_cnt, label="Trends Count", color="green", linestyle="--")
    ax_cnt.plot(laha_stat_dts, total_cnt, label="Total Count", color="red", linestyle="--")
    ax_cnt.set_ylabel("Count")

    ax_cnt.legend(loc="lower left")

    # GC
    ax_gc = ax[1]
    ax_gc.plot(laha_stat_dts[1::], measurements_gc, label="Measurements GC", color="blue")
    ax_gc.plot(laha_stat_dts[1::], trends_gc, label="Trends GC", color="green")
    ax_gc.plot(laha_stat_dts[1::], total_gc, label="Total GC", color="red")

    ax_gc.set_title("AML Garbage Collection")
    ax_gc.set_yscale("log")

    ax_gc.set_ylabel("Items Garbage Collected")
    ax_gc.legend(loc="upper left")

    # % GC
    ax_gc_p: plt.Axes = ax_gc.twinx()

    total_measurements: np.ndarray = measurements_cnt[1::] + measurements_gc
    measurements_pct: np.ndarray = measurements_gc / total_measurements * 100.0

    total_trends: np.ndarray = trends_cnt[1::] + trends_gc
    trends_pct: np.ndarray = trends_gc / total_trends * 100.0

    total_trends: np.ndarray = trends_cnt[1::] + trends_gc
    trends_pct: np.ndarray = trends_gc / total_trends * 100.0

    total: np.ndarray = total_cnt[1::] + total_gc
    total_pct: np.ndarray = total_gc / total * 100.0

    ax_gc_p.plot(laha_stat_dts[1::], measurements_pct, label="Percent Measurements GC", color="blue", linestyle="--")
    ax_gc_p.plot(laha_stat_dts[1::], trends_pct, label="Percent Trends GC", color="green", linestyle="--")
    # ax_gc_p.plot(dts[1::], total_pct, label="Percent AML GCed", color="red", linestyle="--")
    ax_gc_p.legend(loc="lower left")
    ax_gc_p.set_ylabel("Percent Garbage Collected")

    # Active devices
    ax_active = ax[2]
    ax_active.plot(laha_stat_dts, active_devices, label="Active Devices", color="blue")
    ax_active.set_ylabel("Active OPQ Boxes")
    ax_active.set_title("Active OPQ Boxes")
    ax_active.set_xlabel("Time (UTC)")
    ax_active.legend(loc="upper left")
    fig.show()
    fig.savefig(f"{out_dir}/actual_aml_opq.png")


def plot_dl(laha_stat_dts: np.ndarray,
            events_gb: np.ndarray,
            events_cnt: np.ndarray,
            events_gc: np.ndarray,
            active_devices: np.ndarray,
            out_dir: str) -> None:

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Laha DL (OPQ)")

    # Size
    ax_size = ax[0]
    ax_size.plot(laha_stat_dts, events_gb, label="Events Size GB", color="blue")
    ax_size.set_ylabel("Size GB")
    ax_size.set_title("Actual DL Size")
    ax_size.legend(loc="upper left")

    # cnt
    ax_cnt = ax_size.twinx()
    ax_cnt.plot(laha_stat_dts, events_cnt, label="Events Count", color="blue", linestyle="--")
    ax_cnt.set_ylabel("Count")

    ax_cnt.legend(loc="lower left")

    # GC
    ax_gc = ax[1]
    ax_gc.plot(laha_stat_dts[1::], events_gc, label="Events GC", color="blue")

    ax_gc.set_title("DL Garbage Collection")
    ax_gc.set_yscale("log")

    ax_gc.set_ylabel("Items Garbage Collected")
    ax_gc.legend(loc="upper left")

    # % GC
    ax_gc_p: plt.Axes = ax_gc.twinx()

    total_events: np.ndarray = events_cnt[1::] + events_gc
    trends_pct: np.ndarray = events_gc / total_events * 100.0

    ax_gc_p.plot(laha_stat_dts[1::], trends_pct, label="Percent Events GC", color="blue", linestyle="--")

    ax_gc_p.legend(loc="lower left")
    ax_gc_p.set_ylabel("Percent Garbage Collected")

    # Active devices
    ax_active = ax[2]
    ax_active.plot(laha_stat_dts, active_devices, label="Active Devices", color="blue")
    ax_active.set_ylabel("Active OPQ Boxes")
    ax_active.set_title("Active OPQ Boxes")
    ax_active.set_xlabel("Time (UTC)")
    ax_active.legend(loc="upper left")

    fig.show()
    fig.savefig(f"{out_dir}/actual_dl_opq.png")


def plot_il(laha_stat_dts: np.ndarray,
            incidents_gb: np.ndarray,
            incidents_cnt: np.ndarray,
            incidents_gc: np.ndarray,
            active_devices: np.ndarray,
            out_dir: str) -> None:

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Laha IL (OPQ)")

    # Size
    ax_size = ax[0]
    ax_size.plot(laha_stat_dts, incidents_gb, label="Incidents Size GB", color="blue")
    ax_size.set_ylabel("Size GB")
    ax_size.set_title("Actual IL Size")
    ax_size.legend(loc="upper left")

    # cnt
    ax_cnt = ax_size.twinx()
    ax_cnt.plot(laha_stat_dts, incidents_cnt, label="Incidents Count", color="blue", linestyle="--")
    ax_cnt.set_ylabel("Count")

    ax_cnt.legend(loc="lower left")

    # GC
    ax_gc = ax[1]
    ax_gc.plot(laha_stat_dts[1::], incidents_gc, label="Incidents GC", color="blue")

    ax_gc.set_title("IL Garbage Collection")
    ax_gc.set_yscale("log")

    ax_gc.set_ylabel("Items Garbage Collected")
    ax_gc.legend(loc="upper left")

    # % GC
    ax_gc_p: plt.Axes = ax_gc.twinx()

    total_incidents: np.ndarray = incidents_cnt[1::] + incidents_gc
    trends_pct: np.ndarray = incidents_gc / total_incidents * 100.0

    ax_gc_p.plot(laha_stat_dts[1::], trends_pct, label="Percent Incidents GC", color="blue", linestyle="--")

    ax_gc_p.legend(loc="lower left")
    ax_gc_p.set_ylabel("Percent Garbage Collected")

    # Active devices
    ax_active = ax[2]
    ax_active.plot(laha_stat_dts, active_devices, label="Active Devices", color="blue")
    ax_active.set_ylabel("Active OPQ Boxes")
    ax_active.set_title("Active OPQ Boxes")
    ax_active.set_xlabel("Time (UTC)")
    ax_active.legend(loc="upper left")

    fig.show()
    fig.savefig(f"{out_dir}/actual_il_opq.png")


def plot_laha(laha_stat_dts: np.ndarray,
              iml_gb: np.ndarray,
              aml_gb: np.ndarray,
              dl_gb: np.ndarray,
              il_gb: np.ndarray,
              total_gb: np.ndarray,
              out_dir: str):

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    fig.suptitle("Laha IL (OPQ)")

    # Size
    size_ax = ax
    size_ax.plot(laha_stat_dts, iml_gb, label="IML Total")
    size_ax.plot(laha_stat_dts, aml_gb, label="AML Total")
    size_ax.plot(laha_stat_dts, dl_gb, label="DL Total")
    size_ax.plot(laha_stat_dts, il_gb, label="IL Total")
    size_ax.plot(laha_stat_dts, total_gb, label="Total")

    size_ax.set_yscale("log")
    size_ax.set_ylabel("Size GB")
    size_ax.legend(loc="upper left")

    # GC

    # Active Devices
    # ax_active = ax[2]
    # ax_active.plot(dts, active_devices, label="Active Devices", color="blue")
    # ax_active.set_ylabel("Active OPQ Boxes")
    # ax_active.set_title("Active OPQ Boxes")
    # ax_active.set_xlabel("Time (UTC)")
    # ax_active.legend(loc="upper left")

    fig.show()
    fig.savefig(f"{out_dir}/actual_laha_opq.png")


def plot_system_resources(laha_stats: List[LahaStat], out_dir: str):
    timestamps_s: List[int] = list(map(lambda laha_stat: laha_stat.timestamp_s, laha_stats))
    dts: List[datetime.datetime] = list(map(datetime.datetime.utcfromtimestamp, timestamps_s))
    active_devices: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, laha_stats))
    system_stats: List[SystemStats] = list(map(lambda laha_stat: laha_stat.system_stats, laha_stats))

    cpu_load_percent_mins: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.cpu_load_percent.min, system_stats)))
    cpu_load_percent_maxes: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.cpu_load_percent.max, system_stats)))
    cpu_load_percent_means: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.cpu_load_percent.mean, system_stats)))

    memory_use_bytes_mins: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.memory_use_bytes.min, system_stats)))
    memory_use_bytes_maxes: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.memory_use_bytes.max, system_stats)))
    memory_use_bytes_means: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.memory_use_bytes.mean, system_stats)))

    disk_use_bytes_mins: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.disk_use_bytes.min, system_stats)))
    disk_use_bytes_maxes: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.disk_use_bytes.max, system_stats)))
    disk_use_bytes_means: np.ndarray = np.array(
            list(map(lambda system_stat: system_stat.disk_use_bytes.mean, system_stats)))

    # Plot
    fig, ax = plt.subplots(4, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("OPQ Cloud Resource Utilization")

    # CPU
    ax_cpu = ax[0]
    ax_cpu.plot(dts, cpu_load_percent_mins, label="Min")
    ax_cpu.plot(dts, cpu_load_percent_means, label="Mean")
    ax_cpu.plot(dts, cpu_load_percent_maxes, label="Max", color="red")

    ax_cpu.set_title("CPU Load Percent")
    ax_cpu.set_ylabel("Percent Load")
    ax_cpu.legend(loc="upper left")

    # Memory
    ax_memory = ax[1]
    ax_memory.plot(dts, memory_use_bytes_mins / 1_000_000., label="Min")
    ax_memory.plot(dts, memory_use_bytes_means / 1_000_000., label="Mean")
    ax_memory.plot(dts, memory_use_bytes_maxes / 1_000_000., label="Max", color="red")

    ax_memory.set_title("Memory Used")
    ax_memory.set_ylabel("Size MB")
    ax_memory.legend(loc="upper left")

    # Disk
    ax_disk = ax[2]
    ax_disk.plot(dts, disk_use_bytes_mins / 1_000_000_000., label="Min")
    ax_disk.plot(dts, disk_use_bytes_means / 1_000_000_000., label="Mean")
    ax_disk.plot(dts, disk_use_bytes_maxes / 1_000_000_000., label="Max", color="red")

    ax_disk.set_title("Disk Used")
    ax_disk.set_ylabel("Size GB")
    ax_disk.legend(loc="upper left")

    # Active Devices
    ax_active = ax[3]
    ax_active.plot(dts, active_devices, label="Active OPQ Boxes", color="blue")

    ax_active.set_title("Active OPQ Boxes")
    ax_active.set_ylabel("Active OPQ Boxes")
    ax_active.set_xlabel("Time (UTC)")
    ax_active.legend(loc="upper left")

    print(np.array(active_devices).std())

    fig.show()
    fig.savefig(f"{out_dir}/actual_system_opq.png")


def plot_iml_vs_no_tll(laha_stats: List[LahaStat], out_dir: str) -> None:
    # Estimated Data
    s_samp = 2
    sr = 12_000
    mu_n_sen = 15

    # Align the data
    first_laha_stat_timestamp_s = laha_stats[0].timestamp_s
    last_laha_stat_timestamp_s = laha_stats[-1].timestamp_s
    time_range = last_laha_stat_timestamp_s - first_laha_stat_timestamp_s
    aligned_actual_dts, aligned_actual_vals, aligned_est_dts, aligned_est_vals = align_data(
            laha_stats,
            list(range(1, time_range + 1)),
            lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s),
            lambda x: datetime.datetime.utcfromtimestamp(first_laha_stat_timestamp_s + x),
            lambda laha_stat: laha_stat,
            lambda x: s_samp * sr * mu_n_sen * x / 1_000_000.0
    )

    # Actual Data
    active_devices: np.ndarray = np.array(
        list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, aligned_actual_vals)))
    iml_actual: np.ndarray = active_devices * 12_000 * 2 * 60 * 15 / 1_000_000.0
    # iml_actual = iml_actual - iml_actual[0]

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual IML vs IML w/o TTL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(aligned_est_dts, aligned_est_vals)

    ax_estimated.set_title("Estimated Unbounded IML with 15 Sensors")
    ax_estimated.set_ylabel("Size MB")

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(aligned_actual_dts, iml_actual)

    ax_actual.set_title("Actual IML")
    ax_actual.set_ylabel("Size MB")

    # Estimated - Actual
    ax_diff = ax[2]
    ax_diff.plot(aligned_actual_dts, aligned_est_vals - iml_actual)

    ax_diff.set_title("Difference (Estimated IML - Actual IML)")
    ax_diff.set_ylabel("Size MB")
    ax_diff.set_xlabel("Time (UTC)")

    fig.show()
    fig.savefig(f"{out_dir}/actual_iml_vs_unbounded_opq.png")


def plot_aml_vs_no_tll(laha_stats: List[LahaStat], out_dir: str) -> None:
    # Align the data
    first_laha_stat_timestamp_s = laha_stats[0].timestamp_s
    last_laha_stat_timestamp_s = laha_stats[-1].timestamp_s
    time_range = last_laha_stat_timestamp_s - first_laha_stat_timestamp_s
    aligned_actual_dts, aligned_actual_vals, aligned_est_dts, aligned_est_vals = align_data(
            laha_stats,
            list(range(1, time_range + 1)),
            lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s),
            lambda x: datetime.datetime.utcfromtimestamp(first_laha_stat_timestamp_s + x),
            lambda laha_stat: laha_stat,
            lambda x: ((x * 145 * 1.0 / 1.0) + (x * 365 * 1.0 / 60.0)) * 15.0 / 1_000_000_000.0
    )

    # Actual Data
    measurements: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "measurements"), aligned_actual_vals))
    measurement_bytes: np.ndarray = np.array(list(map(lambda measurement: measurement.size_bytes, measurements)))
    measurement_gb: np.ndarray = measurement_bytes / 1_000_000_000.0

    trends: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "trends"), aligned_actual_vals))
    trends_bytes: np.ndarray = np.array(list(map(lambda trend: trend.size_bytes, trends)))
    trends_gb: np.ndarray = trends_bytes / 1_000_000_000.0

    total_gb = trends_gb + measurement_gb

    total_gb = total_gb - total_gb[0]

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True, sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual AML vs AML w/o TTL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(aligned_est_dts, aligned_est_vals)

    ax_estimated.set_title("Estimated Unbounded AML with 15 Sensors")
    ax_estimated.set_ylabel("Size GB")

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(aligned_actual_dts, total_gb)

    ax_actual.set_title("Actual AML")
    ax_actual.set_ylabel("Size GB")

    # Estimated - Actual
    ax_diff = ax[2]
    diff = aligned_est_vals - total_gb
    ax_diff.plot(aligned_actual_dts, diff)

    ax_diff.set_title("Difference (Estimated AML - Actual AML)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")

    fig.show()
    fig.savefig(f"{out_dir}/actual_aml_vs_unbounded_opq.png")


def plot_dl_vs_no_tll(laha_stats: List[LahaStat], out_dir: str) -> None:
    # Estimated Data
    mu_dr = 3138.2362879905268

    # Align the data
    first_laha_stat_timestamp_s = laha_stats[0].timestamp_s
    last_laha_stat_timestamp_s = laha_stats[-1].timestamp_s
    time_range = last_laha_stat_timestamp_s - first_laha_stat_timestamp_s
    aligned_actual_dts, aligned_actual_vals, aligned_est_dts, aligned_est_vals = align_data(
            laha_stats,
            list(range(1, time_range + 1)),
            lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s),
            lambda x: datetime.datetime.utcfromtimestamp(first_laha_stat_timestamp_s + x),
            lambda laha_stat: laha_stat,
            lambda x: mu_dr * x / 1_000_000_000.0
    )

    # Actual Data
    events: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "events"), aligned_actual_vals))
    events_bytes: np.ndarray = np.array(list(map(lambda event: event.size_bytes, events)))
    events_gb: np.ndarray = events_bytes / 1_000_000_000.0
    events_gb = events_gb - events_gb[0]

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True, sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual DL vs DL w/o TTL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(aligned_est_dts, aligned_est_vals)

    ax_estimated.set_title("Estimated Unbounded DL with 15 Sensors")
    ax_estimated.set_ylabel("Size GB")

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(aligned_actual_dts, events_gb)

    ax_actual.set_title("Actual DL")
    ax_actual.set_ylabel("Size GB")

    # Estimated - Actual
    ax_diff = ax[2]
    diff = aligned_est_vals - events_gb
    ax_diff.plot(aligned_actual_dts, diff)

    ax_diff.set_title("Difference (Estimated DL - Actual DL)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")

    fig.show()
    fig.savefig(f"{out_dir}/actual_dl_vs_unbounded_opq.png")


def plot_il_vs_no_tll(laha_stats: List[LahaStat], out_dir: str) -> None:
    # Estimated Data
    mu_dr = 234.83720746762222

    # Align the data
    first_laha_stat_timestamp_s = laha_stats[0].timestamp_s
    last_laha_stat_timestamp_s = laha_stats[-1].timestamp_s
    time_range = last_laha_stat_timestamp_s - first_laha_stat_timestamp_s
    aligned_actual_dts, aligned_actual_vals, aligned_est_dts, aligned_est_vals = align_data(
            laha_stats,
            list(range(1, time_range + 1)),
            lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s),
            lambda x: datetime.datetime.utcfromtimestamp(first_laha_stat_timestamp_s + x),
            lambda laha_stat: laha_stat,
            lambda x: (mu_dr * x) / 1_000_000.0
    )

    # Actual Data
    incidents: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "incidents"), aligned_actual_vals))
    incidents_bytes: np.ndarray = np.array(list(map(lambda incident: incident.size_bytes, incidents)))
    incidents_gb: np.ndarray = incidents_bytes / 1_000_000_000.0
    incidents_gb = incidents_gb - incidents_gb[0]

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True, sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual IL vs IL w/o TTL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(aligned_est_dts, aligned_est_vals)

    ax_estimated.set_title("Estimated Unbounded IL with 15 Sensors")
    ax_estimated.set_ylabel("Size GB")

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(aligned_actual_dts, incidents_gb)

    ax_actual.set_title("Actual IL")
    ax_actual.set_ylabel("Size GB")

    # Estimated - Actual
    ax_diff = ax[2]
    diff = aligned_est_vals - incidents_gb
    ax_diff.plot(aligned_actual_dts, diff)

    ax_diff.set_title("Difference (Estimated IL - Actual IL)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")

    fig.show()
    fig.savefig(f"{out_dir}/actual_il_vs_unbounded_opq.png")


def plot_laha_vs_no_tll(laha_stats: List[LahaStat], out_dir: str) -> None:
    # Estimated Data
    s_samp = 2
    sr = 12_000
    mu_n_sen = 15
    mu_dr_dl = 3138.2362879905268
    mu_dr_il = 234.83720746762222

    # Align the data
    first_laha_stat_timestamp_s = laha_stats[0].timestamp_s
    last_laha_stat_timestamp_s = laha_stats[-1].timestamp_s
    time_range = last_laha_stat_timestamp_s - first_laha_stat_timestamp_s
    aligned_actual_dts, aligned_actual_vals, aligned_est_dts, aligned_est_vals = align_data(
            laha_stats,
            list(range(1, time_range + 1)),
            lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s),
            lambda x: datetime.datetime.utcfromtimestamp(first_laha_stat_timestamp_s + x),
            lambda laha_stat: laha_stat,
            lambda x: (s_samp * sr * mu_n_sen * x) + ((x * 145 * 1.0 / 1.0) + (x * 365 * 1.0 / 60.0) * 15.0) + (
                        x * mu_dr_dl) + (x * mu_dr_il) / 1_000_000_000.0
    )

    # Actual Data
    active_devices: np.ndarray = np.array(
        list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, aligned_actual_vals)))

    # IML
    iml_actual: np.ndarray = active_devices * 12_000 * 2 * 60 * 15 / 1_000_000.0
    # iml_actual = iml_actual - iml_actual[0]

    # AML
    measurements: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "measurements"), aligned_actual_vals))
    measurement_bytes: np.ndarray = np.array(list(map(lambda measurement: measurement.size_bytes, measurements)))
    measurement_gb: np.ndarray = measurement_bytes / 1_000_000_000.0

    trends: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "trends"), aligned_actual_vals))
    trends_bytes: np.ndarray = np.array(list(map(lambda trend: trend.size_bytes, trends)))
    trends_gb: np.ndarray = trends_bytes / 1_000_000_000.0

    total_gb_aml = trends_gb + measurement_gb

    total_gb_aml = total_gb_aml - total_gb_aml[0]

    # DL
    events: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "events"), aligned_actual_vals))
    events_bytes: np.ndarray = np.array(list(map(lambda event: event.size_bytes, events)))
    events_gb: np.ndarray = events_bytes / 1_000_000_000.0
    events_gb = events_gb - events_gb[0]

    # IL

    incidents: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "incidents"), aligned_actual_vals))
    incidents_bytes: np.ndarray = np.array(list(map(lambda incident: incident.size_bytes, incidents)))
    incidents_gb: np.ndarray = incidents_bytes / 1_000_000_000.0
    incidents_gb = incidents_gb - incidents_gb[0]

    # Total
    total_opq_gb = iml_actual + total_gb_aml + events_gb + incidents_gb

    # Estimated Data

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual Laha vs Laha w/o TTL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(aligned_est_dts, aligned_est_vals)

    ax_estimated.set_title("Estimated Unbounded Laha with 15 Sensors")
    ax_estimated.set_ylabel("Size GB")

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(aligned_actual_dts, total_opq_gb)

    ax_actual.set_title("Actual Laha")
    ax_actual.set_ylabel("Size GB")

    # Estimated - Actual
    ax_diff = ax[2]
    diff = aligned_est_vals - total_opq_gb
    ax_diff.plot(aligned_actual_dts, diff)

    ax_diff.set_title("Difference (Estimated Laha - Actual Laha)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")

    fig.show()
    fig.savefig(f"{out_dir}/actual_laha_vs_unbounded_opq.png")


def plot_laha_vs_no_tll_no_iml(laha_stats: List[LahaStat], out_dir: str) -> None:
    # Estimated Data
    mu_dr_dl = 3138.2362879905268
    mu_dr_il = 234.83720746762222

    # Align the data
    first_laha_stat_timestamp_s = laha_stats[0].timestamp_s
    last_laha_stat_timestamp_s = laha_stats[-1].timestamp_s
    time_range = last_laha_stat_timestamp_s - first_laha_stat_timestamp_s
    aligned_actual_dts, aligned_actual_vals, aligned_est_dts, aligned_est_vals = align_data(
            laha_stats,
            list(range(1, time_range + 1)),
            lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s),
            lambda x: datetime.datetime.utcfromtimestamp(first_laha_stat_timestamp_s + x),
            lambda laha_stat: laha_stat,
            lambda x: ((x * 145 * 1.0 / 1.0) + (x * 365 * 1.0 / 60.0) * 15.0) + (x * mu_dr_dl) + (
                        x * mu_dr_il) / 1_000_000_000.0
    )

    # Actual Data

    # AML
    measurements: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "measurements"), aligned_actual_vals))
    measurement_bytes: np.ndarray = np.array(list(map(lambda measurement: measurement.size_bytes, measurements)))
    measurement_gb: np.ndarray = measurement_bytes / 1_000_000_000.0

    trends: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "trends"), aligned_actual_vals))
    trends_bytes: np.ndarray = np.array(list(map(lambda trend: trend.size_bytes, trends)))
    trends_gb: np.ndarray = trends_bytes / 1_000_000_000.0

    total_gb_aml = trends_gb + measurement_gb

    total_gb_aml = total_gb_aml - total_gb_aml[0]

    # DL
    events: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "events"), aligned_actual_vals))
    events_bytes: np.ndarray = np.array(list(map(lambda event: event.size_bytes, events)))
    events_gb: np.ndarray = events_bytes / 1_000_000_000.0
    events_gb = events_gb - events_gb[0]

    # IL

    incidents: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "incidents"), aligned_actual_vals))
    incidents_cnt: np.ndarray = np.array(list(map(lambda incident: incident.count, incidents)))
    incidents_bytes: np.ndarray = np.array(list(map(lambda incident: incident.size_bytes, incidents)))
    incidents_gb: np.ndarray = incidents_bytes / 1_000_000_000.0
    incidents_gb = incidents_gb - incidents_gb[0]

    # Total
    total_opq_gb = total_gb_aml + events_gb + incidents_gb

    # Estimated Data

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual Laha vs Laha w/o TTL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(aligned_est_dts, aligned_est_vals)

    ax_estimated.set_title("Estimated Unbounded Laha with 15 Sensors")
    ax_estimated.set_ylabel("Size GB")

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(aligned_actual_dts, total_opq_gb)

    ax_actual.set_title("Actual Laha")
    ax_actual.set_ylabel("Size GB")

    # Estimated - Actual
    ax_diff = ax[2]
    diff = aligned_est_vals - total_opq_gb
    ax_diff.plot(aligned_actual_dts, diff)

    ax_diff.set_title("Difference (Estimated Laha - Actual Laha)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")

    fig.show()
    fig.savefig(f"{out_dir}/actual_laha_vs_unbounded_opq.png")


def plot_iml_vs_sim(laha_stat_dts: np.ndarray,
                    sim_data_dts: np.ndarray,
                    aligned_laha_iml_total_mb: np.ndarray,
                    aligned_sim_iml_total_mb: np.ndarray,
                    out_dir: str) -> None:
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual IML vs IML w/o TTL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(sim_data_dts, aligned_sim_iml_total_mb)

    ax_estimated.set_title("Actual IML vs Simulated IML (OPQ)")
    ax_estimated.set_ylabel("Size MB")

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(laha_stat_dts, aligned_laha_iml_total_mb)

    ax_actual.set_title("Actual IML")
    ax_actual.set_ylabel("Size MB")

    # Estimated - Actual
    ax_diff = ax[2]
    ax_diff.plot(laha_stat_dts, aligned_sim_iml_total_mb - aligned_laha_iml_total_mb)

    ax_diff.set_title("Difference (Simulated IML - Actual IML)")
    ax_diff.set_ylabel("Size MB")
    ax_diff.set_xlabel("Time (UTC)")

    fig.show()
    fig.savefig(f"{out_dir}/actual_iml_vs_sim_opq.png")


def plot_aml_vs_sim(laha_stat_dts: np.ndarray,
                    sim_data_dts: np.ndarray,
                    aligned_laha_measurements_gb_zero_offset: np.ndarray,
                    aligned_laha_trends_gb_zero_offset: np.ndarray,
                    aligned_laha_aml_total_gb_zero_offset: np.ndarray,
                    aligned_sim_measurements_gb: np.ndarray,
                    aligned_sim_trends_gb: np.ndarray,
                    aligned_sim_aml_total_gb: np.ndarray,
                    out_dir: str) -> None:
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual AML vs Simulated AML (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(sim_data_dts, aligned_sim_measurements_gb, label="Measurements")
    ax_estimated.plot(sim_data_dts, aligned_sim_trends_gb, label="Trends")
    ax_estimated.plot(sim_data_dts, aligned_sim_aml_total_gb, label="Total AML", color="red")

    ax_estimated.set_title("Actual AML vs Simulated AML (OPQ)")
    ax_estimated.set_ylabel("Size GB")
    ax_estimated.legend()

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(laha_stat_dts, aligned_laha_measurements_gb_zero_offset, label="Measurements")
    ax_actual.plot(laha_stat_dts, aligned_laha_trends_gb_zero_offset, label="Trends")
    ax_actual.plot(laha_stat_dts, aligned_laha_aml_total_gb_zero_offset, label="Total AML", color="red")

    ax_actual.set_title("Actual AML")
    ax_actual.set_ylabel("Size GB")
    ax_actual.legend()

    # Estimated - Actual
    ax_diff = ax[2]
    ax_diff.plot(laha_stat_dts, aligned_sim_measurements_gb - aligned_laha_measurements_gb_zero_offset,
                 label="Difference Measurements")
    ax_diff.plot(laha_stat_dts, aligned_sim_trends_gb - aligned_laha_trends_gb_zero_offset, label="Difference Trends")
    ax_diff.plot(laha_stat_dts, aligned_sim_aml_total_gb - aligned_laha_aml_total_gb_zero_offset,
                 label="Difference Total AML", color="red")

    ax_diff.set_title("Difference (Simulated AML - Actual AML)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")
    ax_diff.legend()

    fig.show()
    fig.savefig(f"{out_dir}/actual_aml_vs_sim_opq.png")


def plot_dl_vs_sim(laha_stat_dts: np.ndarray,
                   sim_data_dts: np.ndarray,
                   aligned_laha_events_gb_offset_zero: np.ndarray,
                   aligned_sim_events_gb: np.ndarray,
                   out_dir: str) -> None:
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual DL vs Simulated DL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(sim_data_dts, aligned_sim_total_events_gb, label="Events")

    ax_estimated.set_title("Actual DL vs Simulated DL (OPQ)")
    ax_estimated.set_ylabel("Size GB")
    ax_estimated.legend()

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(laha_stat_dts, aligned_laha_events_gb_offset_zero, label="Events")

    ax_actual.set_title("Actual DL")
    ax_actual.set_ylabel("Size GB")
    ax_actual.legend()

    # Estimated - Actual
    ax_diff = ax[2]
    ax_diff.plot(laha_stat_dts, aligned_sim_events_gb - aligned_laha_events_gb_offset_zero, label="Difference Events")

    ax_diff.set_title("Difference (Simulated DL - Actual DL)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")
    ax_diff.legend()

    fig.show()
    fig.savefig(f"{out_dir}/actual_dl_vs_sim_opq.png")


def plot_il_vs_sim(laha_stat_dts: np.ndarray,
                   sim_data_dts: np.ndarray,
                   aligned_laha_incidents_gb_offset_zero: np.ndarray,
                   aligned_sim_incidents_gb: np.ndarray,
                   out_dir: str) -> None:
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual IL vs Simulated IL (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(sim_data_dts, aligned_sim_incidents_gb, label="Incidents")

    ax_estimated.set_title("Actual IL vs Simulated IL (OPQ)")
    ax_estimated.set_ylabel("Size GB")
    ax_estimated.legend()

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(laha_stat_dts, aligned_laha_incidents_gb_offset_zero, label="Incidents")

    ax_actual.set_title("Actual IL")
    ax_actual.set_ylabel("Size GB")
    ax_actual.legend()

    # Estimated - Actual
    ax_diff = ax[2]
    ax_diff.plot(laha_stat_dts, aligned_sim_incidents_gb - aligned_laha_incidents_gb_offset_zero,
                 label="Difference Incidents")

    ax_diff.set_title("Difference (Simulated IL - Actual IL)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")
    ax_diff.legend()

    fig.show()
    fig.savefig(f"{out_dir}/actual_il_vs_sim_opq.png")


def plot_laha_vs_sim(laha_stat_dts: np.ndarray,
                     sim_data_dts: np.ndarray,
                     aligned_laha_total_gb: np.ndarray,
                     aligned_sim_total_gb: np.ndarray,
                     out_dir: str) -> None:
    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Actual Laha vs Simulated Laha (OPQ)")

    # Estimated
    ax_estimated = ax[0]
    ax_estimated.plot(sim_data_dts, aligned_sim_total_gb, label="Total Simulated Laha")

    ax_estimated.set_title("Actual Laha vs Simulated Laha (OPQ)")
    ax_estimated.set_ylabel("Size GB")
    ax_estimated.legend()

    # Actual
    ax_actual = ax[1]
    ax_actual.plot(laha_stat_dts, aligned_laha_total_gb, label="Total Actual Laha")

    ax_actual.set_title("Actual Laha")
    ax_actual.set_ylabel("Size GB")
    ax_actual.legend()

    # Estimated - Actual
    ax_diff = ax[2]
    ax_diff.plot(laha_stat_dts, aligned_sim_total_gb - aligned_laha_total_gb, label="Difference Laha")

    ax_diff.set_title("Difference (Simulated Laha - Actual Laha)")
    ax_diff.set_ylabel("Size GB")
    ax_diff.set_xlabel("Time (UTC)")
    ax_diff.legend()

    fig.show()
    fig.savefig(f"{out_dir}/actual_laha_vs_sim_opq.png")


if __name__ == "__main__":
    # mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    # laha_stats: List[LahaStat] = get_laha_stats(mongo_client)
    # save_laha_stats(laha_stats, "laha_stats.pickle.db")

    print("Parsing Laha stats...", end=" ")
    laha_stats: List[LahaStat] = load_laha_stats("laha_stats.pickle.db")
    print("Done.")
    print("Parsing Sim Data...", end=" ")
    sim_data = parse_file("sim_data.txt")
    print("Done.")

    print("Aligning Laha stats and Sim Data...", end=" ")
    first_laha_stat_timestamp_s: int = laha_stats[0].timestamp_s
    aligned: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = align_data(
            laha_stats,
            sim_data,
            lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s),
            lambda data: datetime.datetime.utcfromtimestamp(first_laha_stat_timestamp_s + data.time),
            lambda laha_stat: laha_stat,
            lambda sim_data: sim_data
    )
    print("Done")

    aligned_laha_stats_dts: np.ndarray = aligned[0]
    aligned_laha_stats: np.ndarray = aligned[1]
    aligned_sim_data_dts: np.ndarray = aligned[2]
    aligned_sim_data: np.ndarray = aligned[3]

    print("Extracting features from Laha Stats...", end=" ")
    laha_stat_dts: np.ndarray = np.array(
        list(map(lambda laha_stat: datetime.datetime.utcfromtimestamp(laha_stat.timestamp_s), laha_stats)))

    # Laha IML
    aligned_laha_active_devices: np.ndarray = np.array(
        list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, aligned_laha_stats)))
    aligned_laha_iml_total_b: np.ndarray = aligned_laha_active_devices * 12_000 * 2 * 60 * 15
    aligned_laha_iml_total_mb: np.ndarray = aligned_laha_iml_total_b / 1_000_000.0
    aligned_laha_iml_total_gb: np.ndarray = aligned_laha_iml_total_b / 1_000_000_000.0

    laha_active_devices: np.ndarray = np.array(
        list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, laha_stats)))
    laha_iml_total_b: np.ndarray = laha_active_devices * 12_000 * 2 * 60 * 15
    laha_iml_total_mb: np.ndarray = laha_iml_total_b / 1_000_000.0
    laha_iml_total_gb: np.ndarray = laha_iml_total_b / 1_000_000_000.0

    # Laha AML
    aligned_laha_measurements: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "measurements"), aligned_laha_stats))
    aligned_laha_measurements_bytes: np.ndarray = np.array(
        list(map(lambda measurement: measurement.size_bytes, aligned_laha_measurements)))
    aligned_laha_measurements_gb: np.ndarray = aligned_laha_measurements_bytes / 1_000_000_000.0
    aligned_laha_measurements_gb_zero_offset: np.ndarray = aligned_laha_measurements_gb - aligned_laha_measurements_gb[
        0]

    aligned_laha_trends: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "trends"), aligned_laha_stats))
    aligned_laha_trends_bytes: np.ndarray = np.array(list(map(lambda trend: trend.size_bytes, aligned_laha_trends)))
    aligned_laha_trends_gb: np.ndarray = aligned_laha_trends_bytes / 1_000_000_000.0
    aligned_laha_trends_gb_zero_offset: np.ndarray = aligned_laha_trends_gb - aligned_laha_trends_gb[0]

    aligned_laha_aml_total_gb: np.ndarray = aligned_laha_trends_gb + aligned_laha_measurements_gb
    aligned_laha_aml_total_gb_zero_offset: np.ndarray = aligned_laha_trends_gb_zero_offset + \
                                                        aligned_laha_measurements_gb_zero_offset

    laha_measurements: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "measurements"), laha_stats))
    laha_measurements_bytes: np.ndarray = np.array(
        list(map(lambda measurement: measurement.size_bytes, laha_measurements)))
    laha_measurements_gb: np.ndarray = laha_measurements_bytes / 1_000_000_000.0
    laha_measurements_gb_zero_offset: np.ndarray = laha_measurements_gb - laha_measurements_gb[0]
    laha_measurements_cnt: np.ndarray = np.array(list(map(lambda measurement: measurement.count, laha_measurements)))
    laha_measurements_gc: np.ndarray = correct_counts(
        np.array(list(map(lambda laha_stat: laha_stat.laha_stats.gc_stats.measurements, laha_stats))))

    laha_trends: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "trends"), laha_stats))
    laha_trends_bytes: np.ndarray = np.array(list(map(lambda trend: trend.size_bytes, laha_trends)))
    laha_trends_gb: np.ndarray = laha_trends_bytes / 1_000_000_000.0
    laha_trends_gb_zero_offset: np.ndarray = laha_trends_gb - laha_trends_gb[0]
    laha_trends_cnt: np.ndarray = np.array(list(map(lambda trend: trend.count, laha_trends)))
    laha_trends_gc: np.ndarray = correct_counts(
        np.array(list(map(lambda trend: trend.laha_stats.gc_stats.trends, laha_stats))))

    laha_aml_total_cnt: np.ndarray = laha_measurements_cnt + laha_trends_cnt
    laha_aml_total_gc: np.ndarray = laha_measurements_gc + laha_trends_gc
    laha_aml_total_gb: np.ndarray = laha_trends_gb + laha_measurements_gb
    laha_aml_total_gb_zero_offset: np.ndarray = laha_trends_gb_zero_offset + laha_measurements_gb_zero_offset

    # Laha DL
    aligned_laha_events: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "events"), aligned_laha_stats))
    aligned_laha_events_bytes: np.ndarray = np.array(list(map(lambda event: event.size_bytes, aligned_laha_events)))
    aligned_laha_events_gb: np.ndarray = aligned_laha_events_bytes / 1_000_000_000.0
    aligned_laha_events_gb_offset_zero: np.ndarray = aligned_laha_events_gb - aligned_laha_events_gb[0]

    laha_events: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "events"), laha_stats))
    laha_events_bytes: np.ndarray = np.array(list(map(lambda event: event.size_bytes, laha_events)))
    laha_events_gb: np.ndarray = laha_events_bytes / 1_000_000_000.0
    laha_events_gb_offset_zero: np.ndarray = laha_events_gb - laha_events_gb[0]
    laha_events_cnt: np.ndarray = np.array(list(map(lambda event: event.count, laha_events)))
    laha_events_gc: np.ndarray = correct_counts(
            np.array(list(map(lambda laha_stat: laha_stat.laha_stats.gc_stats.events, laha_stats))))

    # Laha IL
    aligned_laha_incidents: List[LahaMetric] = list(
        map(lambda laha_stat: map_laha_metric(laha_stat, "incidents"), aligned_laha_stats))
    aligned_laha_incidents_bytes: np.ndarray = np.array(
        list(map(lambda incident: incident.size_bytes, aligned_laha_incidents)))
    aligned_laha_incidents_gb: np.ndarray = aligned_laha_incidents_bytes / 1_000_000_000.0
    aligned_laha_incidents_gb_offset_zero: np.ndarray = aligned_laha_incidents_gb - aligned_laha_incidents_gb[0]

    laha_incidents: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "incidents"), laha_stats))
    laha_incidents_bytes: np.ndarray = np.array(list(map(lambda incident: incident.size_bytes, laha_incidents)))
    laha_incidents_gb: np.ndarray = laha_incidents_bytes / 1_000_000_000.0
    laha_incidents_gb_offset_zero: np.ndarray = laha_incidents_gb - laha_incidents_gb[0]
    laha_incidents_cnt: np.ndarray = np.array(list(map(lambda incident: incident.count, laha_incidents)))
    laha_incidents_gc: np.ndarray = correct_counts(
            np.array(list(map(lambda laha_stat: laha_stat.laha_stats.gc_stats.incidents, laha_stats))))

    # Laha
    aligned_laha_total_gb: np.ndarray = aligned_laha_iml_total_gb + aligned_laha_aml_total_gb + \
                                        aligned_laha_events_gb + aligned_laha_incidents_gb
    aligned_laha_total_gb_offset_zero: np.ndarray = aligned_laha_total_gb - aligned_laha_total_gb[0]

    laha_total_gb: np.ndarray = laha_iml_total_gb + laha_aml_total_gb + laha_events_gb + laha_incidents_gb
    laha_total_gb_offset_zero: np.ndarray = laha_total_gb - laha_total_gb[0]
    print("Done")

    print("Extracting features from Sim Data...", end=" ")
    # Sim IML
    aligned_sim_iml_total_b: np.ndarray = np.array(list(map(lambda d: d.total_samples_b, aligned_sim_data)))
    aligned_sim_iml_total_mb: np.ndarray = aligned_sim_iml_total_b / 1_000_000.0 * 15
    aligned_sim_iml_total_gb: np.ndarray = aligned_sim_iml_total_b / 1_000_000_000.0 * 15

    # Sim AML
    aligned_sim_total_measurements_b = np.array(list(map(lambda d: d.total_measurements_b, aligned_sim_data)))
    aligned_sim_total_measurements_gb = aligned_sim_total_measurements_b / 1_000_000_000.0 * 15.0
    aligned_sim_total_trends_b = np.array(list(map(lambda d: d.total_trends_b, aligned_sim_data)))
    aligned_sim_total_trends_gb = aligned_sim_total_trends_b / 1_000_000_000.0 * 15.0
    aligned_sim_aml_total_gb = aligned_sim_total_measurements_gb + aligned_sim_total_trends_gb

    # Sim DL
    aligned_sim_total_events_b = np.array(list(map(lambda d: d.total_events_b, aligned_sim_data))) * 15.0
    aligned_sim_total_events_gb = aligned_sim_total_events_b / 1_000_000_000.0

    # Sim IL
    aligned_sim_total_incidents_b = np.array(list(map(lambda d: d.total_incidents_b, aligned_sim_data))) * 15.0
    aligned_sim_total_incidents_gb = aligned_sim_total_incidents_b / 1_000_000_000.0

    # Sim total
    aligned_sim_total_gb = aligned_sim_iml_total_gb + aligned_sim_aml_total_gb + aligned_sim_total_events_gb + \
                           aligned_sim_total_incidents_gb
    print("Done")

    out_dir: str = "/Users/anthony/Development/dissertation/src/figures"

    print("Making plots...")

    # Laha Plots

    # plot_iml(laha_stat_dts,
    #          laha_active_devices,
    #          laha_iml_total_mb,
    #          out_dir)

    # plot_aml(laha_stat_dts,
    #          laha_measurements_gb,
    #          laha_trends_gb,
    #          laha_aml_total_gb,
    #          laha_measurements_cnt,
    #          laha_trends_cnt,
    #          laha_aml_total_cnt,
    #          laha_measurements_gc,
    #          laha_trends_gc,
    #          laha_aml_total_gc,
    #          laha_active_devices,
    #          out_dir)

    # plot_dl(laha_stat_dts,
    #         laha_events_gb,
    #         laha_events_cnt,
    #         laha_events_gc,
    #         laha_active_devices,
    #         out_dir)

    # plot_il(laha_stat_dts,
    #         laha_incidents_gb,
    #         laha_incidents_cnt,
    #         laha_incidents_gc,
    #         laha_active_devices,
    #         out_dir)

    plot_laha(laha_stat_dts,
              laha_iml_total_gb,
              laha_aml_total_gb,
              laha_events_gb,
              laha_incidents_gb,
              laha_total_gb,
              out_dir)

    # plot_system_resources(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_iml_vs_no_tll(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_aml_vs_no_tll(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_dl_vs_no_tll(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_il_vs_no_tll(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_laha_vs_no_tll(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_laha_vs_no_tll_no_iml(laha_stats, "/Users/anthony/Development/dissertation/src/figures")

    # Simulation plots
    # plot_iml_vs_sim(aligned_laha_stats_dts,
    #                 aligned_sim_data_dts,
    #                 aligned_laha_iml_total_mb,
    #                 aligned_sim_iml_total_mb,
    #                 out_dir)
    #
    # plot_aml_vs_sim(aligned_laha_stats_dts,
    #                 aligned_sim_data_dts,
    #                 aligned_laha_measurements_gb_zero_offset,
    #                 aligned_laha_trends_gb_zero_offset,
    #                 aligned_laha_aml_total_gb_zero_offset,
    #                 aligned_sim_total_measurements_gb,
    #                 aligned_sim_total_trends_gb,
    #                 aligned_sim_aml_total_gb,
    #                 out_dir)
    #
    # plot_dl_vs_sim(aligned_laha_stats_dts,
    #                aligned_sim_data_dts,
    #                aligned_laha_events_gb_offset_zero,
    #                aligned_sim_total_events_gb,
    #                out_dir)
    #
    # plot_il_vs_sim(aligned_laha_stats_dts,
    #                aligned_sim_data_dts,
    #                aligned_laha_incidents_gb_offset_zero,
    #                aligned_sim_total_incidents_gb,
    #                out_dir)
    #
    # plot_laha_vs_sim(aligned_laha_stats_dts,
    #                  aligned_sim_data_dts,
    #                  aligned_laha_total_gb,
    #                  aligned_sim_total_gb,
    #                  out_dir)
