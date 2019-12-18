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
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    fig.suptitle(
            f"Frequency Ground Truth Comparison\n{opq_box_id} vs {uhm_sensor}"
            f"\n{aligned_opq_dts[0].strftime('%Y-%m-%d')} to "
            f"{aligned_opq_dts[-1].strftime('%Y-%m-%d')}"
            f"\n$\mu$={mean_diff:.4f} $\sigma$={mean_stddev:.4f}"
    )

    n, bins, patches = ax.hist(diffs, bins=250, density=True)

    x = np.linspace(diffs.min(), diffs.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mean_diff, mean_stddev))

    ax.set_xlabel("Frequency Difference Hz (UHM - OPQ)")
    ax.set_ylabel("% Density")

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
    with open(os.path.join(out_dir, "f_latex.txt"), "w") as fout:
        for opq_box, uhm_meters in util.opq_box_to_uhm_meters.items():
            for uhm_meter in uhm_meters:
                try:
                    print(f"plot_frequency {opq_box} {uhm_meter}")
                    path = plot_frequency(opq_start_ts_s,
                                          opq_end_ts_s,
                                          opq_box,
                                          ground_truth_root,
                                          uhm_meter,
                                          mongo_client,
                                          out_dir)
                    fout.write(util.latex_figure_source(path))
                except Exception as e:
                    print(e, "...ignoring...")
