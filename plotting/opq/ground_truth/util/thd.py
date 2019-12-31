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

THD_TYPES = ["AVG_VOLTAGE_THD"]


def normal(mu: float, sigma: float, bins: np.ndarray, percent_density: float) -> np.ndarray:
    return ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)) * percent_density

def print_tex_table(opq_box_id: str,
                    uhm_meter: str,
                    mus: List[float],
                    sigmas: List[float]) -> None:
    mu_strs = list(map(lambda mu: f"{mu:.4f}", mus))
    sigma_strs = list(map(lambda sigma: f"{sigma:.4f}", sigmas))

    print(f"{opq_box_id} & {uhm_meter} & {' '.join(mu_strs)} & {' '.join(sigma_strs)} \\\\")

def plot_thd(opq_start_ts_s: int,
             opq_end_ts_s: int,
             opq_box_id: str,
             ground_truth_root: str,
             uhm_sensor: str,
             mongo_client: pymongo.MongoClient,
             out_dir: str) -> str:
    ground_truth_path: str = f"{ground_truth_root}/{uhm_sensor}/AVG_VOLTAGE_THD"
    uhm_data_points: List[io.DataPoint] = io.parse_file(ground_truth_path)

    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["trends"]
    query: Dict = {
        "box_id": opq_box_id,
        "timestamp_ms": {"$gte": opq_start_ts_s * 1000,
                         "$lte": opq_end_ts_s * 1000},
        "thd": {"$exists": True}
    }

    projection: Dict[str, bool] = {
        "_id": False,
        "box_id": True,
        "timestamp_ms": True,
        "thd": True,
    }

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    opq_trend_docs: List[Dict] = list(cursor)
    opq_trends: List[io.Trend] = list(map(io.Trend.from_doc, opq_trend_docs))

    aligned_data: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] = align.align_data_by_min(
            opq_trends,
            uhm_data_points,
            lambda trend: datetime.datetime.utcfromtimestamp(trend.timestamp_ms / 1000.0),
            lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
            lambda trend: trend.thd.average * 100.0,
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

    ax.set_title(
            f"THD Comparison {aligned_opq_dts[0].strftime('%m-%d')} to {aligned_opq_dts[-1].strftime('%m-%d')}"
            f"\n{opq_box_id} vs {uhm_sensor}"
    )

    # n, bins, patches = ax.hist(diffs, bins=250, density=True)

    if opq_box_id == "1000" and uhm_sensor == "POST_MAIN_1":
        split: float = -0.4
        low: np.ndarray = diffs[diffs < split]
        high: np.ndarray = diffs[diffs >= split]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(high)))
        high_p: float = 1.0 - low_p

        low_bins: np.ndarray = bins[bins < split]
        high_bins: np.ndarray = bins[bins >= split]

        low_mu = low.mean()
        low_sigma = low.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
        print_tex_table(opq_box_id, uhm_sensor, [low_mu, high_mu], [low_sigma, high_sigma])
    elif opq_box_id == "1000" and uhm_sensor == "POST_MAIN_2":
        split: float = 0.07
        low: np.ndarray = diffs[diffs < split]
        high: np.ndarray = diffs[diffs >= split]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(high)))
        high_p: float = 1.0 - low_p

        low_bins: np.ndarray = bins[bins < split]
        high_bins: np.ndarray = bins[bins >= split]

        low_mu = low.mean()
        low_sigma = low.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
        print_tex_table(opq_box_id, uhm_sensor, [low_mu, high_mu], [low_sigma, high_sigma])
    elif opq_box_id == "1002" and uhm_sensor == "POST_MAIN_1":
        split: float = 0.15
        low: np.ndarray = diffs[diffs < split]
        high: np.ndarray = diffs[diffs >= split]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(high)))
        high_p: float = 1.0 - low_p

        low_bins: np.ndarray = bins[bins < split]
        high_bins: np.ndarray = bins[bins >= split]

        low_mu = low.mean()
        low_sigma = low.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
        print_tex_table(opq_box_id, uhm_sensor, [low_mu, high_mu], [low_sigma, high_sigma])
    elif opq_box_id == "1002" and uhm_sensor == "POST_MAIN_2":
        split: float = 0.5
        low: np.ndarray = diffs[diffs < split]
        high: np.ndarray = diffs[diffs >= split]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(high)))
        high_p: float = 1.0 - low_p

        low_bins: np.ndarray = bins[bins < split]
        high_bins: np.ndarray = bins[bins >= split]

        low_mu = low.mean()
        low_sigma = low.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
        print_tex_table(opq_box_id, uhm_sensor, [low_mu, high_mu], [low_sigma, high_sigma])
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_MAIN_2_MTR":
        split: float = 1.20
        low: np.ndarray = diffs[diffs < split]
        high: np.ndarray = diffs[diffs >= split]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(high)))
        high_p: float = 1.0 - low_p

        low_bins: np.ndarray = bins[bins < split]
        high_bins: np.ndarray = bins[bins >= split]

        low_mu = low.mean()
        low_sigma = low.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
        print_tex_table(opq_box_id, uhm_sensor, [low_mu, high_mu], [low_sigma, high_sigma])
    elif opq_box_id == "1021" and uhm_sensor == "MARINE_SCIENCE_MAIN_B_MTR":
        split: float = 0.45
        low: np.ndarray = diffs[diffs < split]
        high: np.ndarray = diffs[diffs >= split]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(high)))
        high_p: float = 1.0 - low_p

        low_bins: np.ndarray = bins[bins < split]
        high_bins: np.ndarray = bins[bins >= split]

        low_mu = low.mean()
        low_sigma = low.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
        print_tex_table(opq_box_id, uhm_sensor, [low_mu, high_mu], [low_sigma, high_sigma])
    else:
        n, bins, patches = ax.hist(diffs, bins=250, density=True)
        x = np.linspace(diffs.min(), diffs.max(), 100)
        ax.plot(x, stats.norm.pdf(x, mean_diff, mean_stddev),
                label=f"\n$\mu$={mean_diff:.4f} $\sigma$={mean_stddev:.4f}")
        print_tex_table(opq_box_id, uhm_sensor, [mean_diff], [mean_stddev])

    ax.set_xlabel("THD Difference  (UHM - OPQ)")
    ax.set_ylabel("% Density")
    ax.legend()
    # fig.show()
    path = f"{out_dir}/thd_hist_{opq_box_id}_{uhm_sensor}.png"
    fig.savefig(path, bbox_inches='tight')
    return path


def plot_thd_incidents(opq_start_ts_s: int,
                       opq_end_ts_s: int,
                       opq_box_id: str,
                       ground_truth_root: str,
                       uhm_sensor: str,
                       mongo_client: pymongo.MongoClient,
                       out_dir: str) -> str:
    f_types: List[str] = ["EXCESSIVE_THD"]

    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["incidents"]
    query: Dict = {
        "box_id": opq_box_id,
        "start_timestamp_ms": {"$gte": opq_start_ts_s * 1000},
        "end_timestamp_ms": {"$lte": opq_end_ts_s * 1000},
        "classifications": {"$in": f_types}
    }

    measurements = list(db["trends"].find({"box_id": opq_box_id,
                                           "timestamp_ms": {"$gte": opq_start_ts_s * 1000,
                                                            "$lte": opq_end_ts_s * 1000}}))

    m_dts = list(map(lambda m: datetime.datetime.utcfromtimestamp(m["timestamp_ms"] / 1000.0), measurements))
    m_val = list(map(lambda m: m["thd"]["max"] * 100.0, measurements))

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=io.Incident.projection())
    incidents: List[io.Incident] = list(map(io.Incident.from_doc, list(cursor)))

    ground_truth_path: str = f"{ground_truth_root}/{uhm_sensor}/AVG_VOLTAGE_THD"
    uhm_data_points: List[io.DataPoint] = io.parse_file(ground_truth_path)

    uhm_dts: np.ndarray = np.array(
        list(map(lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s) - datetime.timedelta(hours=0), uhm_data_points)))
    uhm_vals_max: np.ndarray = np.array(list(map(lambda data_point: data_point.max_v, uhm_data_points)))
    uhm_vals_min: np.ndarray = np.array(list(map(lambda data_point: data_point.min_v, uhm_data_points)))

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    # ax.plot(uhm_dts, uhm_vals_min, label="UHM Min. THD", color="blue")
    ax.plot(uhm_dts, uhm_vals_max + .50, label="UHM Max. THD", color="red")

    # ax.plot(m_dts, m_val)

    # freq_threshold_low = 60.0 - (60.0 * .0016)
    # freq_threshold_high = 60.0 + (60.0 * .0016)

    # ax.plot(uhm_dts, [freq_threshold_low for _ in uhm_dts], linestyle="--", color="blue")
    ax.plot(uhm_dts, [5 for _ in uhm_dts], linestyle="--", color="red")

    ax.set_title(f"THD Incidents Comparison OPQ Box {opq_box_id} vs UHM Sensor {uhm_sensor}")

    incident_dts: np.ndarray = np.array(list(map(lambda incident: datetime.datetime.utcfromtimestamp(
    incident.start_timestamp_ms / 1000.0), incidents)))
    incident_vals: np.ndarray = np.array(list(map(lambda incident: incident.deviation_from_nominal, incidents)))

    incident_dts_binned = set(map(align.bin_dt_by_min, incident_dts))

    ax.scatter(incident_dts, [5 for _ in incident_dts])
    n = len(uhm_vals_max[uhm_vals_max > 5.0])
    m = len(uhm_vals_max[(uhm_vals_max + .50) > 5.0])
    zero_crossings = np.where(np.diff(np.sign((uhm_vals_max + .50) - 5.0)))[0]
    # print(f"thd incidents={len(incidents)} binned={len(incident_dts_binned)} uhm={n}")
    uhm_tex = uhm_sensor.replace("_", "\\_")
    print(f"{opq_box_id} & {uhm_tex} & {len(incident_dts_binned)} & {len(zero_crossings)} \\\\")
    ax.legend()
    fig.show()

    return ""


def compare_thds(opq_start_ts_s: int,
                 opq_end_ts_s: int,
                 ground_truth_root: str,
                 mongo_client: pymongo.MongoClient,
                 out_dir: str) -> None:
    paths: List[str] = []
    for opq_box, uhm_meters in util.opq_box_to_uhm_meters.items():
        for uhm_meter in uhm_meters:
            try:
                # print(f"plot_thd {opq_box} {uhm_meter}")
                path = plot_thd(opq_start_ts_s,
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


def compare_thd_incidents(opq_start_ts_s: int,
                          opq_end_ts_s: int,
                          ground_truth_root: str,
                          mongo_client: pymongo.MongoClient,
                          out_dir: str):
    paths: List[str] = []

    for opq_box, uhm_meters in util.opq_box_to_uhm_meters.items():
        for uhm_meter in uhm_meters:
            try:
                # print(f"plot_frequency_incident {opq_box} {uhm_meter}")
                path = plot_thd_incidents(opq_start_ts_s,
                                          opq_end_ts_s,
                                          opq_box,
                                          ground_truth_root,
                                          uhm_meter,
                                          mongo_client,
                                          out_dir)
                paths.append(path)
            except Exception as e:
                print(e, "...ignoring...")
    # util.latex_figure_table_source(paths, 3, 2)
