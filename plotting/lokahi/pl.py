from dataclasses import dataclass
import datetime
import os
from typing import Callable, Dict, List, Tuple, TypeVar, Optional, Set
import urllib.parse

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
    return datetime.datetime(dt.year, dt.month, dt.day, dt.hour, dt.minute, 0, 0)


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

def sum_series(series: np.ndarray) -> np.ndarray:
    result: List[int] = [series[0]]
    for i in range(1, len(series)):
        result.append(result[i - 1] + series[i])

    return np.array(result)


@dataclass
class ActualPl:
    ts_s: int
    size_bytes: int

    @staticmethod
    def from_line(line: str) -> 'ActualPl':
        s: List[str] = line.split(" ")
        i: List[int] = list(map(int, s))
        return ActualPl(*i)

    def dt(self):
        return datetime.datetime.utcfromtimestamp(self.ts_s)


data = """
1494890400 4730
1496523900 3932
1497209340 5356
1498244940 2560
1498245420 2248
1498422300 3402
1498469400 8343
1499296980 2506
1504409400 1613
1504410960 1872
1504806240 1163
1511973000 7304
1513352100 2816
1515373200 3317
1517433900 4110
1517949300 8940
1517949900 6079
1519941720 3806
1520314380 2849
1523370600 1198
1523374200 1184
1523374800 1478
1523377800 1184
1523394480 3467
1524855840 2550
1525379396 2473
1525469504 13568
1525473116 21085
1525473116 13721
1526495400 1941
1526529600 2324
1526565600 2236
1535043600 9616
1535068800 5625
1539749700 1976
1542445200 2678
1552474442 23229
1555022040 7110
1555211042 7384
1556054040 5431
1556313360 2431
1558656000 11211
1558665000 18501
1561444200 8554
1561444200 3627
1562065200 8183
1562065200 5485
1562788800 7330
1564049460 13799
1565259180 5621
1565620800 3439
1570237785 4842
1573484100 2482
1578363540 4576
"""


def format_mongo_auth_uri_ssl(user: str,
                              password: str,
                              host: str,
                              port: int,
                              cert: str) -> str:
    escaped_pass = urllib.parse.quote_plus(password)
    return f"mongodb://{user}:{escaped_pass}@{host}:{port}/?ssl=true&ssl_ca_certs={cert}&replicaSet=rs0"


def get_client(host: str, port: int, user: str, password: str, cert: str) -> pymongo.MongoClient:
    return pymongo.MongoClient(format_mongo_auth_uri_ssl(user, password, host, port, cert))


def pl_vs_sim(aligned_dts: np.ndarray,
              aligned_actual_vals: np.ndarray,
              aligned_data: np.ndarray):

    summed_sizes: np.ndarray = sum_series(np.array(aligned_actual_vals))
    summed_sizes_mb: np.ndarray = summed_sizes / 1_000.0

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Lokahi: PL vs Simulated PL")

    sim_ax = ax[0]
    sim_ax.plot(aligned_dts, aligned_data / 1_000.0)
    sim_ax.set_title("Simulated PL")
    sim_ax.set_ylabel("Size (kB)")

    actual_ax = ax[1]
    actual_ax.plot(aligned_dts, summed_sizes_mb)
    actual_ax.set_title("Actual PL")
    actual_ax.set_ylabel("Size (kB)")

    diff_ax = ax[2]
    diff_ax.plot(aligned_dts, summed_sizes_mb - (aligned_data / 1_000.0))
    diff_ax.set_title("Difference (Simulated - Actual)")
    diff_ax.set_ylabel("Size (kB)")
    diff_ax.set_xlabel("Time (UTC)")

    # fig.savefig("/home/opq/Documents/anthony/dissertation/src/figures/actual_phenomena_lokahi_vs_sim.png")
    fig.show()


def pl_vs_est(pls: List[ActualPl]):
    dts: List[datetime.datetime] = list(map(ActualPl.dt, pls))
    size_bytes: List[int] = list(map(lambda pl: pl.size_bytes, pls))
    summed_bytes: np.ndarray = sum_series(np.array(size_bytes))
    summed_kb = summed_bytes / 1_000.0

    xs: np.ndarray = np.array(list(map(lambda pl: pl.ts_s, pls)))
    xs = xs - xs[0]
    vs = xs * 0.01 / 1_000.0

    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all", sharey="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle("Lokahi: Actual PL vs Estimated PL")

    est_ax = ax[0]
    est_ax.plot(dts, vs)
    est_ax.set_title("Estimated PL")
    est_ax.set_ylabel("Size (kB)")

    actual_ax = ax[1]
    actual_ax.plot(dts, summed_kb)
    actual_ax.set_title("Actual PL")
    actual_ax.set_ylabel("Size (kB)")

    diff_ax = ax[2]
    diff_ax.plot(dts, vs - summed_kb)
    diff_ax.set_title("Difference (Estimated - Actual)")
    diff_ax.set_ylabel("Size (kB)")
    diff_ax.set_xlabel("Time (UTC)")

    # fig.show()
    fig.savefig("/home/opq/Documents/anthony/dissertation/src/figures/lokahi_pl_actual_vs_est.png")


def slope_intercept(slope: float, intercept: float) -> str:
    return f"y = {slope} * x + {intercept}"


def actual_pl(pls: List[ActualPl]) -> None:
    dts: List[datetime.datetime] = list(map(ActualPl.dt, pls))
    size_bytes: List[int] = list(map(lambda pl: pl.size_bytes, pls))
    summed_bytes: np.ndarray = sum_series(np.array(size_bytes))
    summed_kb = summed_bytes / 1_000.0

    xs = np.array(list(map(lambda dt: dt.timestamp(), dts)))
    xs = xs - xs[0]
    slope, intercept, r_value, p_value, std_err = stats.linregress(xs, summed_kb)
    print("pl", slope_intercept(slope, intercept))

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax
    ax.plot(dts, summed_kb, label="PL Growth")
    ax.plot(dts, intercept + slope * xs, color="black", linestyle=":",
            label=f"PL Total MB LR ($m$={slope:.5f} $b$={intercept:.5f} $R^2$={r_value ** 2:.5f} $\sigma$="
                  f"{std_err:.5f})")

    ax.set_title("Lokahi: Actual PL Growth")
    ax.set_ylabel("Size (kB)")
    ax.set_xlabel("Time (UTC)")
    ax.legend()

    fig.show()
    fig.savefig("/home/opq/Documents/anthony/dissertation/src/figures/lokahi_pl_actual.png")


def main():
    # host = os.getenv("REDVOX_MONGO_HOST")
    # port = int(os.getenv("REDVOX_MONGO_PORT"))
    # user = os.getenv("REDVOX_MONGO_USER")
    # passw = os.getenv("REDVOX_MONGO_PASS")
    #
    # mongo_client: pymongo.MongoClient = get_client(host, port, user, passw, "/home/ec2-user/rds-combined-ca-bundle.pem")
    # db: pymongo.database.Database = mongo_client["redvox"]
    # coll: pymongo.collection.Collection = db["WebBasedReport"]
    # web_based_report_docs: List[dict] = sorted(list(coll.find({"isPublic": True})), key=lambda doc: doc["startTimestampS"])
    #
    # for doc in web_based_report_docs:
    #     start_ts_s: int = doc["startTimestampS"]
    #     size_bytes: int = len(bson.BSON.encode(doc))
    #     print(f"{start_ts_s} {size_bytes}")
    actual_data_lines: List[str] = data.splitlines()[1:]
    actual_pls: List[ActualPl] = list(map(ActualPl.from_line, actual_data_lines))[50:]
    # print(actual_pls)

    # actual_pl(actual_pls)
    # pl_vs_est(actual_pls)

    sim_data: List[Data] = parse_file("/home/opq/scrap/sim_data_80_lokahi.txt")[50:]
    start_ts_s = actual_pls[0].ts_s
    specs: List[SeriesSpec] = [
        SeriesSpec(actual_pls,
                   lambda doc: doc.dt(),
                   lambda doc: doc.size_bytes),
        SeriesSpec(sim_data,
                   lambda doc: datetime.datetime.utcfromtimestamp(start_ts_s + 3600),
                   lambda doc: doc.total_pl_b)
    ]

    aligned_data: List[Tuple[np.ndarray, np.ndarray]] = align_data_multi(specs)
    actual_data: Tuple[np.ndarray, np.ndarray] = aligned_data[0]
    sim_data: Tuple[np.ndarray, np.ndarray] = aligned_data[1]
    aligned_dts: np.ndarray = actual_data[0]
    aligned_actual: np.ndarray = actual_data[1]
    aligned_sim: np.ndarray = sim_data[1]

    print(aligned_dts)
    print(aligned_actual)
    print(aligned_sim)

    pl_vs_sim(aligned_dts, aligned_actual, aligned_sim)


if __name__ == "__main__":
    main()
