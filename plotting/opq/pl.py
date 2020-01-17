import datetime
from dataclasses import dataclass
from typing import Dict, List, Callable, Set, Tuple

import bson
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database
from scipy import stats

@dataclass
class SeriesSpec:
    values: List
    dt_func: Callable
    v_func: Callable


def bin_dt_by_min(dt: datetime.datetime) -> datetime.datetime:
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0, 0, tzinfo=datetime.timezone.utc)


# def bin_dt_by_day(dt: datetime.datetime) -> datetime.datetime:
#     return datetime.datetime(dt.year, dt.month, dt.day, 0, 0, 0, 0)


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
        binned_dts: List[datetime.datetime] = list(map(bin_dt_by_min, dts))
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


def sum_series(series: np.ndarray) -> np.ndarray:
    result: List[int] = [series[0]]
    for i in range(1, len(series)):
        result.append(result[i - 1] + series[i])

    return np.array(result)


def slope_intercept(slope: float, intercept: float) -> str:
    return f"y = {slope} * x + {intercept}"


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


def pl_vs_sim(aligned_dts: np.ndarray,
              aligned_actual_vals: np.ndarray,
              aligned_data: np.ndarray):

    summed_sizes: np.ndarray = sum_series(np.array(aligned_actual_vals))
    summed_sizes_mb: np.ndarray = summed_sizes / 1_000_000.0

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("OPQ: PL vs Simulated PL")

    sim_ax = ax[0]
    sim_ax.plot(aligned_dts, aligned_data / 1_000_000.0)
    sim_ax.set_title("Simulated PL")
    sim_ax.set_ylabel("Size (MB)")

    actual_ax = ax[1]
    actual_ax.plot(aligned_dts, summed_sizes_mb)
    actual_ax.set_title("Actual PL")
    actual_ax.set_ylabel("Size (MB)")

    diff_ax = ax[2]
    diff_ax.plot(aligned_dts, summed_sizes_mb - (aligned_data / 1_000_000.0))
    diff_ax.set_title("Difference (Simulated - Actual)")
    diff_ax.set_ylabel("Size (MB)")
    diff_ax.set_xlabel("Time (UTC)")

    fig.savefig("/Users/anthony/Development/dissertation/src/figures/actual_phenomena_opq_vs_sim.png")
    # fig.show()


def pl_vs_est(mongo_client: pymongo.MongoClient):
    # Query
    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["phenomena"]

    # Data
    phenomena_docs: List[Dict] = sorted(list(coll.find()), key=lambda doc: doc["start_ts_ms"])
    start_timestamps_ms: List[float] = list(map(lambda doc: doc["start_ts_ms"], phenomena_docs))
    ts_s: np.ndarray = np.array(start_timestamps_ms) / 1_000.0
    sizes_bytes: List[int] = list(map(lambda doc: len(bson.BSON.encode(doc)), phenomena_docs))
    dts: List[datetime.datetime] = list(
            map(lambda ts: datetime.datetime.utcfromtimestamp(ts / 1000.0), start_timestamps_ms))

    summed_sizes: np.ndarray = sum_series(np.array(sizes_bytes))
    summed_sizes_mb: np.ndarray = summed_sizes / 1_000_000.0

    # Est data
    dr_s = 0.22
    ts = ts_s - ts_s[0]
    vs = ts * dr_s
    vs = vs / 1_000_000.0

    # Plots
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("OPQ: Estimated PL vs Actual PL")

    est_ax = ax[0]
    est_ax.plot(dts, vs)
    est_ax.set_title("Estimated PL")
    est_ax.set_ylabel("Size (MB)")

    actual_ax = ax[1]
    actual_ax.plot(dts, summed_sizes_mb)
    actual_ax.set_title("Actual PL")
    actual_ax.set_ylabel("Size (MB)")

    diff_ax = ax[2]
    diff_ax.plot(dts, vs - summed_sizes_mb)
    diff_ax.set_title("Difference (Estimated - Actual)")

    diff_ax.set_xlabel("Time (UTC)")
    diff_ax.set_ylabel("Size (MB)")

    # fig.show()
    fig.savefig("/home/opq/Documents/anthony/dissertation/src/figures/actual_phenomena_opq_vs_est.png")


def actual_pl(mongo_client: pymongo.MongoClient):
    # Query
    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["phenomena"]

    # Data
    phenomena_docs: List[Dict] = sorted(list(coll.find()), key=lambda doc: doc["start_ts_ms"])
    start_timestamps_ms: List[float] = list(map(lambda doc: doc["start_ts_ms"], phenomena_docs))
    sizes_bytes: List[int] = list(map(lambda doc: len(bson.BSON.encode(doc)), phenomena_docs))
    dts: List[datetime.datetime] = list(
            map(lambda ts: datetime.datetime.utcfromtimestamp(ts / 1000.0), start_timestamps_ms))

    # for i, dt in enumerate(dts):
    #     print(i, dt)

    summed_sizes: np.ndarray = sum_series(np.array(sizes_bytes))
    summed_sizes_mb: np.ndarray = summed_sizes / 1_000_000.0

    xs = np.array(list(map(lambda dt: dt.timestamp(), dts[3:])))
    xs = xs - xs[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, summed_sizes_mb[3:])
    print("pl", slope_intercept(slope, intercept))

    # Plots
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.plot(dts, summed_sizes_mb, label="OPQ PL Growth")
    ax.plot(dts[3:], intercept + slope * xs, color="black", linestyle=":",
            label=f"PL Total MB LR ($m$={slope:.5f} $b$={intercept:.5f} $R^2$={r_value ** 2:.5f} $\sigma$="
                  f"{std_err:.5f})")

    ax.set_title("Actual Phenomena: OPQ")
    ax.set_ylabel("Size (MB)")
    ax.set_xlabel("Time (UTC)")
    ax.legend()

    fig.show()
    fig.savefig("/Users/anthony/Development/dissertation/src/figures/actual_phenomena_opq.png")


def main():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    # actual_pl(mongo_client)
    # pl_vs_est(mongo_client)

    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["phenomena"]

    # Data
    phenomena_docs: List[Dict] = sorted(list(coll.find()), key=lambda doc: doc["start_ts_ms"])
    sim_data: List[Data] = parse_file("/Users/anthony/scrap/sim_data_opq.txt")

    start_ts_s = phenomena_docs[0]["start_ts_ms"] / 1_000.0
    specs: List[SeriesSpec] = [
        SeriesSpec(phenomena_docs,
                   lambda doc: datetime.datetime.utcfromtimestamp(doc["start_ts_ms"] / 1_000.0),
                   lambda doc: len(bson.BSON.encode(doc))),
        SeriesSpec(sim_data,
                   lambda doc: datetime.datetime.utcfromtimestamp(start_ts_s + doc.time),
                   lambda doc: doc.total_pl_b)
    ]

    aligned_data: List[Tuple[np.ndarray, np.ndarray]] = align_data_multi(specs)
    actual_data: Tuple[np.ndarray, np.ndarray] = aligned_data[0]
    sim_data: Tuple[np.ndarray, np.ndarray] = aligned_data[1]
    aligned_dts: np.ndarray = actual_data[0]
    aligned_actual: np.ndarray = actual_data[1]
    aligned_sim: np.ndarray = sim_data[1]

    pl_vs_sim(aligned_dts, aligned_actual, aligned_sim)

if __name__ == "__main__":
    main()
