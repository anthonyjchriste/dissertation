import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database
import scipy.stats as stats

import util
import util.align_data as align
import util.io as io

VOLTAGE_TYPES = ["VAB", "VBC", "VCA"]


def plot_voltage(opq_start_ts_s: int,
                 opq_end_ts_s: int,
                 opq_box_id: str,
                 ground_truth_root: str,
                 uhm_sensor: str,
                 mongo_client: pymongo.MongoClient,
                 out_dir: str) -> None:
    ground_truth_path_vab: str = f"{ground_truth_root}/{uhm_sensor}/VAB"
    uhm_data_points_vab: List[io.DataPoint] = io.parse_file(ground_truth_path_vab)

    ground_truth_path_vbc: str = f"{ground_truth_root}/{uhm_sensor}/VBC"
    uhm_data_points_vbc: List[io.DataPoint] = io.parse_file(ground_truth_path_vbc)

    ground_truth_path_vca: str = f"{ground_truth_root}/{uhm_sensor}/VCA"
    uhm_data_points_vca: List[io.DataPoint] = io.parse_file(ground_truth_path_vca)

    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["trends"]
    query: Dict = {
        "box_id": opq_box_id,
        "timestamp_ms": {"$gte": opq_start_ts_s * 1000,
                         "$lte": opq_end_ts_s * 1000},
        "voltage": {"$exists": True}
    }

    projection: Dict[str, bool] = {
        "_id": False,
        "box_id": True,
        "timestamp_ms": True,
        "voltage": True,
    }

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    opq_trend_docs: List[Dict] = list(cursor)
    opq_trends: List[io.Trend] = list(map(io.Trend.from_doc, opq_trend_docs))

    series: List[align.SeriesSpec] = [
        align.SeriesSpec(opq_trends,
                         lambda opq_trend: datetime.datetime.utcfromtimestamp(opq_trend.timestamp_ms / 1000.0),
                         lambda opq_trend: opq_trend.voltage.average),
        align.SeriesSpec(uhm_data_points_vab,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.avg_v),
        align.SeriesSpec(uhm_data_points_vbc,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.avg_v),
        align.SeriesSpec(uhm_data_points_vca,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.avg_v)
    ]

    aligned_data: List[Tuple[np.ndarray, np.ndarray]] = align.align_data_multi(series)

    trends_data: Tuple[np.ndarray, np.ndarray] = aligned_data[0]
    vab_data: Tuple[np.ndarray, np.ndarray] = aligned_data[1]
    vbc_data: Tuple[np.ndarray, np.ndarray] = aligned_data[2]
    vca_data: Tuple[np.ndarray, np.ndarray] = aligned_data[3]

    trend_dts: np.ndarray = trends_data[0]
    trend_vals: np.ndarray = trends_data[1]
    vab_dts: np.ndarray = vab_data[0]
    vab_vals: np.ndarray = vab_data[1]
    vbc_dts: np.ndarray = vbc_data[0]
    vbc_vals: np.ndarray = vbc_data[1]
    vca_dts: np.ndarray = vca_data[0]
    vca_vals: np.ndarray = vca_data[1]

    eq_left: float = (1.0 / (np.sqrt(3) * 3.9985))
    sq_sum = np.square(vab_vals) + np.square(vbc_vals) + np.square(vca_vals)
    vrms_vals: np.ndarray = eq_left * np.sqrt(sq_sum)

    print(len(vrms_vals), type(vrms_vals), vrms_vals[0])
    print(len(trend_vals), type(trend_vals), trend_vals[0])

    diffs: np.ndarray = vrms_vals - trend_vals





    mean_diff: float = diffs.mean()
    mean_stddev: float = diffs.std()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    fig.suptitle(
            f"RMS Voltage Ground Truth Comparison: {opq_box_id} vs {uhm_sensor} "
            f"{trend_dts[0].strftime('%Y-%m-%d')} to "
            f"{trend_dts[-1].strftime('%Y-%m-%d')}"
            f"\n$\mu$={mean_diff:.4f} $\sigma$={mean_stddev:.4f}"
    )

    n, bins, patches = ax.hist(diffs, bins=250, density=True)

    x = np.linspace(diffs.min(), diffs.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mean_diff, mean_stddev))

    ax.set_xlabel("RMS Difference V (UHM - OPQ)")
    ax.set_ylabel("% Density")

    fig.show()
    # fig.savefig(f"{out_dir}/thd_hist_{opq_box_id}_{uhm_sensor}.png")


def compare_vrms(opq_start_ts_s: int,
                 opq_end_ts_s: int,
                 ground_truth_root: str,
                 mongo_client: pymongo.MongoClient,
                 out_dir: str) -> None:
    for opq_box, uhm_meters in util.opq_box_to_uhm_meters.items():
        for uhm_meter in uhm_meters:
            try:
                print(f"plot_voltages {opq_box} {uhm_meter}")
                plot_voltage(opq_start_ts_s,
                             opq_end_ts_s,
                             opq_box,
                             ground_truth_root,
                             uhm_meter,
                             mongo_client,
                             out_dir)
            except Exception as e:
                print(e, "...ignoring...")
