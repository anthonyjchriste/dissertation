from dataclasses import dataclass
import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database

DB: str = "opq"
COLL: str = "laha_stats"


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


def plot_aml(laha_stats: List[LahaStat], out_dir: str) -> None:
    timestamps_s: List[int] = list(map(lambda laha_stat: laha_stat.timestamp_s, laha_stats))
    dts: List[datetime.datetime] = list(map(datetime.datetime.utcfromtimestamp, timestamps_s))

    def map_laha_metric(laha_stat: LahaStat, name: str) -> Optional[LahaMetric]:
        for laha_metric in laha_stat.laha_stats.laha_metrics:
            if laha_metric.name == name:
                return laha_metric

        return None

    measurements: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "measurements"), laha_stats))
    measurement_cnt: np.ndarray = np.array(list(map(lambda measurement: measurement.count, measurements)))
    measurement_bytes: np.ndarray = np.array(list(map(lambda measurement: measurement.size_bytes, measurements)))
    measurement_gb: np.ndarray = measurement_bytes / 1_000_000_000.0

    trends: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "trends"), laha_stats))
    trends_cnt: np.ndarray = np.array(list(map(lambda trend: trend.count, trends)))
    trends_bytes: np.ndarray = np.array(list(map(lambda trend: trend.size_bytes, trends)))
    trends_gb: np.ndarray = trends_bytes / 1_000_000_000.0

    total_gb = trends_gb + measurement_gb
    total_cnt = trends_cnt + measurement_cnt

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Laha AML (OPQ)")

    # Size
    ax_size = ax[0]
    ax_size.plot(dts, measurement_gb, label="Measurements Size GB", color="blue")
    ax_size.plot(dts, trends_gb, label="Trends Size GB", color="green")
    ax_size.plot(dts, total_gb, label="AML Total GB", color="red")
    ax_size.set_ylabel("Size GB")
    ax_size.set_title("Actual AML Size")
    ax_size.legend(loc="upper left")

    # cnt
    ax_cnt = ax_size.twinx()
    ax_cnt.plot(dts, measurement_cnt, label="Measurements Count", color="blue", linestyle="--")
    ax_cnt.plot(dts, trends_cnt, label="Trends Count", color="green", linestyle="--")
    ax_cnt.plot(dts, total_cnt, label="Total Count", color="red", linestyle="--")
    ax_cnt.set_ylabel("Count")

    ax_cnt.legend(loc="lower left")

    # GC
    measurements_gc: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.gc_stats.measurements, laha_stats))
    trends_gc: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.gc_stats.trends, laha_stats))
    corrected_measurements_gc: np.ndarray = correct_counts(np.array(measurements_gc))
    corrected_trends_gc: np.ndarray = correct_counts(np.array(trends_gc))
    total_gc = corrected_measurements_gc + corrected_trends_gc

    ax_gc = ax[1]
    ax_gc.plot(dts[1::], corrected_measurements_gc, label="Measurements GC", color="blue")
    ax_gc.plot(dts[1::], corrected_trends_gc, label="Trends GC", color="green")
    ax_gc.plot(dts[1::], total_gc, label="Total GC", color="red")

    ax_gc.set_title("AML Garbage Collection")
    ax_gc.set_yscale("log")

    ax_gc.set_ylabel("Items Garbage Collected")
    ax_gc.legend(loc="upper left")

    # % GC
    ax_gc_p: plt.Axes = ax_gc.twinx()

    total_measurements: np.ndarray = measurement_cnt[1::] + corrected_measurements_gc
    measurements_pct: np.ndarray = corrected_measurements_gc / total_measurements * 100.0

    total_trends: np.ndarray = trends_cnt[1::] + corrected_trends_gc
    trends_pct: np.ndarray = corrected_trends_gc / total_trends * 100.0

    total_trends: np.ndarray = trends_cnt[1::] + corrected_trends_gc
    trends_pct: np.ndarray = corrected_trends_gc / total_trends * 100.0

    total: np.ndarray = total_cnt[1::] + total_gc
    total_pct: np.ndarray = total_gc / total * 100.0

    ax_gc_p.plot(dts[1::], measurements_pct, label="Percent Measurements GC", color="blue", linestyle="--")
    ax_gc_p.plot(dts[1::], trends_pct, label="Percent Trends GC", color="green", linestyle="--")
    # ax_gc_p.plot(dts[1::], total_pct, label="Percent AML GCed", color="red", linestyle="--")
    ax_gc_p.legend(loc="lower left")
    ax_gc_p.set_ylabel("Percent Garbage Collected")

    # Active devices
    active_devices: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, laha_stats))
    ax_active = ax[2]
    ax_active.plot(dts, active_devices, label="Active Devices", color="blue")
    ax_active.set_ylabel("Active OPQ Boxes")
    ax_active.set_title("Active OPQ Boxes")
    ax_active.set_xlabel("Time (UTC)")
    ax_active.legend(loc="upper left")
    # fig.show()
    fig.savefig(f"{out_dir}/actual_aml_opq.png")


def plot_dl(laha_stats: List[LahaStat], out_dir: str) -> None:
    timestamps_s: List[int] = list(map(lambda laha_stat: laha_stat.timestamp_s, laha_stats))
    dts: List[datetime.datetime] = list(map(datetime.datetime.utcfromtimestamp, timestamps_s))

    def map_laha_metric(laha_stat: LahaStat, name: str) -> Optional[LahaMetric]:
        for laha_metric in laha_stat.laha_stats.laha_metrics:
            if laha_metric.name == name:
                return laha_metric

        return None

    events: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "events"), laha_stats))
    events_cnt: np.ndarray = np.array(list(map(lambda event: event.count, events)))
    events_bytes: np.ndarray = np.array(list(map(lambda event: event.size_bytes, events)))
    events_gb: np.ndarray = events_bytes / 1_000_000_000.0

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Laha DL (OPQ)")

    # Size
    ax_size = ax[0]
    ax_size.plot(dts, events_gb, label="Events Size GB", color="blue")
    ax_size.set_ylabel("Size GB")
    ax_size.set_title("Actual DL Size")
    ax_size.legend(loc="upper left")

    # cnt
    ax_cnt = ax_size.twinx()
    ax_cnt.plot(dts, events_cnt, label="Events Count", color="blue", linestyle="--")
    ax_cnt.set_ylabel("Count")

    ax_cnt.legend(loc="lower left")

    # GC
    events_gc: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.gc_stats.events, laha_stats))
    corrected_events_gc: np.ndarray = correct_counts(np.array(events_gc))


    ax_gc = ax[1]
    ax_gc.plot(dts[1::], corrected_events_gc, label="Events GC", color="blue")

    ax_gc.set_title("DL Garbage Collection")
    ax_gc.set_yscale("log")

    ax_gc.set_ylabel("Items Garbage Collected")
    ax_gc.legend(loc="upper left")

    # % GC
    ax_gc_p: plt.Axes = ax_gc.twinx()

    total_events: np.ndarray = events_cnt[1::] + corrected_events_gc
    trends_pct: np.ndarray = corrected_events_gc / total_events * 100.0

    ax_gc_p.plot(dts[1::], trends_pct, label="Percent Events GC", color="blue", linestyle="--")

    ax_gc_p.legend(loc="lower left")
    ax_gc_p.set_ylabel("Percent Garbage Collected")

    # Active devices
    active_devices: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, laha_stats))
    ax_active = ax[2]
    ax_active.plot(dts, active_devices, label="Active Devices", color="blue")
    ax_active.set_ylabel("Active OPQ Boxes")
    ax_active.set_title("Active OPQ Boxes")
    ax_active.set_xlabel("Time (UTC)")
    ax_active.legend(loc="upper left")

    # fig.show()
    fig.savefig(f"{out_dir}/actual_dl_opq.png")


def plot_il(laha_stats: List[LahaStat], out_dir: str) -> None:
    timestamps_s: List[int] = list(map(lambda laha_stat: laha_stat.timestamp_s, laha_stats))
    dts: List[datetime.datetime] = list(map(datetime.datetime.utcfromtimestamp, timestamps_s))

    def map_laha_metric(laha_stat: LahaStat, name: str) -> Optional[LahaMetric]:
        for laha_metric in laha_stat.laha_stats.laha_metrics:
            if laha_metric.name == name:
                return laha_metric

        return None

    incidents: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "incidents"), laha_stats))
    incidents_cnt: np.ndarray = np.array(list(map(lambda incident: incident.count, incidents)))
    incidents_bytes: np.ndarray = np.array(list(map(lambda incident: incident.size_bytes, incidents)))
    incidents_gb: np.ndarray = incidents_bytes / 1_000_000_000.0

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Laha IL (OPQ)")

    # Size
    ax_size = ax[0]
    ax_size.plot(dts, incidents_gb, label="Incidents Size GB", color="blue")
    ax_size.set_ylabel("Size GB")
    ax_size.set_title("Actual IL Size")
    ax_size.legend(loc="upper left")

    # cnt
    ax_cnt = ax_size.twinx()
    ax_cnt.plot(dts, incidents_cnt, label="Incidents Count", color="blue", linestyle="--")
    ax_cnt.set_ylabel("Count")

    ax_cnt.legend(loc="lower left")

    # GC
    events_gc: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.gc_stats.incidents, laha_stats))
    corrected_events_gc: np.ndarray = correct_counts(np.array(events_gc))


    ax_gc = ax[1]
    ax_gc.plot(dts[1::], corrected_events_gc, label="Incidents GC", color="blue")

    ax_gc.set_title("IL Garbage Collection")
    ax_gc.set_yscale("log")

    ax_gc.set_ylabel("Items Garbage Collected")
    ax_gc.legend(loc="upper left")

    # % GC
    ax_gc_p: plt.Axes = ax_gc.twinx()

    total_events: np.ndarray = incidents_cnt[1::] + corrected_events_gc
    trends_pct: np.ndarray = corrected_events_gc / total_events * 100.0

    ax_gc_p.plot(dts[1::], trends_pct, label="Percent Incidents GC", color="blue", linestyle="--")

    ax_gc_p.legend(loc="lower left")
    ax_gc_p.set_ylabel("Percent Garbage Collected")

    # Active devices
    active_devices: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, laha_stats))
    ax_active = ax[2]
    ax_active.plot(dts, active_devices, label="Active Devices", color="blue")
    ax_active.set_ylabel("Active OPQ Boxes")
    ax_active.set_title("Active OPQ Boxes")
    ax_active.set_xlabel("Time (UTC)")
    ax_active.legend(loc="upper left")

    # fig.show()
    fig.savefig(f"{out_dir}/actual_il_opq.png")


def plot_iml(laha_stats: List[LahaStat], out_dir: str):
    timestamps_s: List[int] = list(map(lambda laha_stat: laha_stat.timestamp_s, laha_stats))
    dts: List[datetime.datetime] = list(map(datetime.datetime.utcfromtimestamp, timestamps_s))
    active_devices: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, laha_stats))

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), constrained_layout=True)
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    y = np.array(active_devices) * 12_000 * 2 * 60 * 15 / 1_000_000.0

    ax.plot(dts, y, color="blue", label="IML Size")
    ax.set_ylabel("Size MB")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Actual IML (OPQ)")
    ax.legend(loc="upper left")

    ax_active = ax.twinx()
    ax_active.plot(dts, active_devices, visible=False)
    ax_active.set_ylabel("Active OPQ Boxes")

    # fig.show()
    fig.savefig(f"{out_dir}/actual_iml_opq.png")

def plot_laha(laha_stats: List[LahaStat], out_dir: str):
    def map_laha_metric(laha_stat: LahaStat, name: str) -> Optional[LahaMetric]:
        for laha_metric in laha_stat.laha_stats.laha_metrics:
            if laha_metric.name == name:
                return laha_metric

        return None


    timestamps_s: List[int] = list(map(lambda laha_stat: laha_stat.timestamp_s, laha_stats))
    dts: List[datetime.datetime] = list(map(datetime.datetime.utcfromtimestamp, timestamps_s))
    active_devices: List[int] = list(map(lambda laha_stat: laha_stat.laha_stats.active_devices, laha_stats))

    # IML
    iml_gb: np.ndarray = np.array(active_devices) * 12_000 * 2 * 60 * 15 / 1_000_000_000.0

    # AML
    measurements: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "measurements"), laha_stats))
    measurement_cnt: np.ndarray = np.array(list(map(lambda measurement: measurement.count, measurements)))
    measurement_bytes: np.ndarray = np.array(list(map(lambda measurement: measurement.size_bytes, measurements)))
    measurement_gb: np.ndarray = measurement_bytes / 1_000_000_000.0

    trends: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "trends"), laha_stats))
    trends_cnt: np.ndarray = np.array(list(map(lambda trend: trend.count, trends)))
    trends_bytes: np.ndarray = np.array(list(map(lambda trend: trend.size_bytes, trends)))
    trends_gb: np.ndarray = trends_bytes / 1_000_000_000.0

    aml_total_gb = trends_gb + measurement_gb
    aml_total_cnt = trends_cnt + measurement_cnt

    # DL
    events: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "events"), laha_stats))
    events_cnt: np.ndarray = np.array(list(map(lambda event: event.count, events)))
    events_bytes: np.ndarray = np.array(list(map(lambda event: event.size_bytes, events)))
    events_gb: np.ndarray = events_bytes / 1_000_000_000.0

    # IL
    incidents: List[LahaMetric] = list(map(lambda laha_stat: map_laha_metric(laha_stat, "incidents"), laha_stats))
    incidents_cnt: np.ndarray = np.array(list(map(lambda incident: incident.count, incidents)))
    incidents_bytes: np.ndarray = np.array(list(map(lambda incident: incident.size_bytes, incidents)))
    incidents_gb: np.ndarray = incidents_bytes / 1_000_000_000.0

    # Total
    total_gb = iml_gb + aml_total_gb + events_gb + incidents_gb

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9), sharex="all", constrained_layout=True)
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    fig.suptitle("Laha IL (OPQ)")

    # Size
    size_ax = ax
    size_ax.plot(dts, iml_gb, label="IML Total")
    size_ax.plot(dts, aml_total_gb, label="AML Total")
    size_ax.plot(dts, events_gb, label="DL Total")
    size_ax.plot(dts, incidents_gb, label="IL Total")
    size_ax.plot(dts, total_gb, label="Total")

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

    # fig.show()
    fig.savefig(f"{out_dir}/actual_laha_opq.png")



if __name__ == "__main__":
    # mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    #
    # laha_stats: List[LahaStat] = get_laha_stats(mongo_client)
    #
    # # for laha_stat in laha_stats:
    # #     print(laha_stat)
    #
    # save_laha_stats(laha_stats, "laha_stats.pickle.db")

    laha_stats: List[LahaStat] = load_laha_stats("laha_stats.pickle.db")

    # print(len(laha_stats))

    # plot_iml(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_aml(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_dl(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    # plot_il(laha_stats, "/Users/anthony/Development/dissertation/src/figures")
    plot_laha(laha_stats, "/Users/anthony/Development/dissertation/src/figures")

