import datetime
import os.path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database
import scipy.stats as stats

import util
import util.align_data as align
import util.io as io


def plot_frequency(opq_start_ts_s: int,
                   opq_end_ts_s: int,
                   opq_box_id: str,
                   ground_truth_root: str,
                   uhm_sensor: str,
                   mongo_client: pymongo.MongoClient,
                   out_dir: str) -> str:
    ground_truth_path: str = f"{ground_truth_root}/{uhm_sensor}/Frequency"
    uhm_data_points: List[io.DataPoint] = io.parse_file(ground_truth_path)

    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["trends"]
    query: Dict = {
        "box_id": opq_box_id,
        "timestamp_ms": {"$gte": opq_start_ts_s * 1000,
                         "$lte": opq_end_ts_s * 1000},
        "frequency": {"$exists": True}
    }

    projection: Dict[str, bool] = {
        "_id": False,
        "box_id": True,
        "timestamp_ms": True,
        "frequency": True,
    }

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    opq_trend_docs: List[Dict] = list(cursor)
    opq_trends: List[io.Trend] = list(map(io.Trend.from_doc, opq_trend_docs))

    aligned_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = align.align_data_by_min(
            opq_trends,
            uhm_data_points,
            lambda trend: datetime.datetime.utcfromtimestamp(trend.timestamp_ms / 1000.0),
            lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
            lambda trend: trend.frequency.average,
            lambda data_point: data_point.avg_v
    )

    aligned_opq_dts: np.ndarray = aligned_data[0]
    aligned_opq_vs: np.ndarray = aligned_data[1]
    aligned_uhm_dts: np.ndarray = aligned_data[2]
    aligned_uhm_vs: np.ndarray = aligned_data[3]

    diffs: np.ndarray = aligned_uhm_vs - aligned_opq_vs
    mean_diff: float = diffs.mean()
    mean_stddev: float = diffs.std()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    fig.suptitle(
            f"Frequency Comparison "
            f"({aligned_opq_dts[0].strftime('%m-%d')} to "
            f"{aligned_opq_dts[-1].strftime('%m-%d')})"
            f"\n{opq_box_id} vs {uhm_sensor}"

            # f"\n$\mu$={mean_diff:.4f} $\sigma$={mean_stddev:.4f}"
    )

    n, bins, patches = ax.hist(diffs, bins=250, density=True)

    x = np.linspace(diffs.min(), diffs.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mean_diff, mean_stddev), label=f"$\mu$={mean_diff:.4f} $\sigma$={mean_stddev:.4f}")

    print(f"{opq_box_id} & {uhm_sensor} & {mean_diff:.4f} & {mean_stddev:.4f}")

    ax.set_xlabel("Frequency Difference Hz (UHM - OPQ)")
    ax.set_ylabel("% Density")
    ax.legend(loc="upper left")

    # fig.show()
    out_path = f"{out_dir}/f_hist_{opq_box_id}_{uhm_sensor}.png"
    # print(util.latex_figure_source(out_path))
    fig.savefig(f"{out_dir}/f_hist_{opq_box_id}_{uhm_sensor}.png", bbox_inches='tight')
    return out_path


def compare_frequencies(opq_start_ts_s: int,
                        opq_end_ts_s: int,
                        ground_truth_root: str,
                        mongo_client: pymongo.MongoClient,
                        out_dir: str) -> None:
    paths: List[str] = []
    with open(os.path.join(out_dir, "f_latex.txt"), "w") as fout:
        for opq_box, uhm_meters in util.opq_box_to_uhm_meters.items():
            for uhm_meter in uhm_meters:
                try:
                    # print(f"plot_frequency {opq_box} {uhm_meter}")
                    path = plot_frequency(opq_start_ts_s,
                                          opq_end_ts_s,
                                          opq_box,
                                          ground_truth_root,
                                          uhm_meter,
                                          mongo_client,
                                          out_dir)
                    paths.append(path)
                    # fout.write(util.latex_figure_source(path))
                except Exception as e:
                    print(e, "...ignoring...")

    util.latex_figure_table_source(paths, 3, 2)


def plot_frequency_incidents(opq_start_ts_s: int,
                             opq_end_ts_s: int,
                             opq_box_id: str,
                             ground_truth_root: str,
                             uhm_sensor: str,
                             mongo_client: pymongo.MongoClient,
                             out_dir: str) -> str:
    f_types: List[str] = ["FREQUENCY_INTERRUPTION", "FREQUENCY_SAG", "FREQUENCY_SWELL"]

    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["incidents"]
    query: Dict = {
        "box_id": opq_box_id,
        "start_timestamp_ms": {"$gte": opq_start_ts_s * 1000},
        "end_timestamp_ms": {"$lte": opq_end_ts_s * 1000},
        "classifications": {"$in": f_types}
    }

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=io.Incident.projection())
    incidents: List[io.Incident] = list(map(io.Incident.from_doc, list(cursor)))

    ground_truth_path: str = f"{ground_truth_root}/{uhm_sensor}/Frequency"
    uhm_data_points: List[io.DataPoint] = io.parse_file(ground_truth_path)

    uhm_dts: np.ndarray = np.array(list(map(lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s), uhm_data_points)))
    uhm_vals_max: np.ndarray = np.array(list(map(lambda data_point: data_point.max_v, uhm_data_points)))
    uhm_vals_min: np.ndarray = np.array(list(map(lambda data_point: data_point.min_v, uhm_data_points)))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.plot(uhm_dts, uhm_vals_min, label="UHM Min. Frequency", color="blue")
    ax.plot(uhm_dts, uhm_vals_max, label="UHM Max. Frequency", color="red")

    freq_threshold_low = 60.0 - (60.0 * .0016)
    freq_threshold_high = 60.0 + (60.0 * .0016)

    ax.plot(uhm_dts, [freq_threshold_low for _ in uhm_dts], linestyle="--", color="blue")
    ax.plot(uhm_dts, [freq_threshold_high for _ in uhm_dts], linestyle="--", color="red")

    incident_sags: List[io.Incident] = list(filter(lambda incident: "FREQUENCY_SAG" in incident.classifications, incidents))
    incident_sag_dts: np.ndarray = np.array(list(map(lambda incident: datetime.datetime.utcfromtimestamp(incident.start_timestamp_ms / 1000.0), incident_sags)))
    incident_sag_vals: np.ndarray = np.array(list(map(lambda incident: 60 - incident.deviation_from_nominal, incident_sags)))
    ax.scatter(incident_sag_dts, incident_sag_vals, label="OPQ Frequency Sags", color="blue", s=1)

    incident_swells: List[io.Incident] = list(filter(lambda incident: "FREQUENCY_SWELL" in incident.classifications, incidents))
    incident_swell_dts: np.ndarray = np.array(list(map(lambda incident: datetime.datetime.utcfromtimestamp(incident.start_timestamp_ms / 1000.0), incident_swells)))
    incident_swell_vals: np.ndarray = np.array(list(map(lambda incident: 60 - incident.deviation_from_nominal, incident_swells)))
    ax.scatter(incident_swell_dts, incident_swell_vals, label="OPQ Frequency Swells", color="red", s=1)

    ax.legend()
    fig.show()

    return ""


def compare_frequency_incidents(opq_start_ts_s: int,
                                opq_end_ts_s: int,
                                ground_truth_root: str,
                                mongo_client: pymongo.MongoClient,
                                out_dir: str):
    paths: List[str] = []

    for opq_box, uhm_meters in util.opq_box_to_uhm_meters.items():
        for uhm_meter in uhm_meters:
            try:
                # print(f"plot_frequency_incident {opq_box} {uhm_meter}")
                path = plot_frequency_incidents(opq_start_ts_s,
                                                opq_end_ts_s,
                                                opq_box,
                                                ground_truth_root,
                                                uhm_meter,
                                                mongo_client,
                                                out_dir)
                paths.append(path)
            except Exception as e:
                print(e, "...ignoring...")
    util.latex_figure_table_source(paths, 3, 2)
