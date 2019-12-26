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


def normal(mu: float, sigma: float, bins: np.ndarray, percent_density: float) -> np.ndarray:
    return ((1 / (np.sqrt(2 * np.pi) * sigma)) *
            np.exp(-0.5 * (1 / sigma * (bins - mu)) ** 2)) * percent_density


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

    diffs: np.ndarray = vrms_vals - trend_vals

    mean_diff: float = diffs.mean()
    mean_stddev: float = diffs.std()

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig: plt.Figure = fig
    ax: plt.Axes = ax

    ax.set_title(
            f"Voltage Comparison {trend_dts[0].strftime('%m-%d')} to {trend_dts[-1].strftime('%m-%d')}"
            f"\n{opq_box_id} vs {uhm_sensor}"
    )

    if opq_box_id == "1002" and uhm_sensor == "POST_MAIN_1":
        split: float = -2.25
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
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_MAIN_1_MTR":
        split_low: float = -2.2
        split_high: float = -1.65
        low: np.ndarray = diffs[diffs < split_low]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_MAIN_2_MTR":
        split_low: float = -2.0
        split_high: float = -1.40
        low: np.ndarray = diffs[diffs < split_low]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_MCC_AC1_MTR":
        split_low: float = -2.45
        split_high: float = -1.7
        low: np.ndarray = diffs[diffs < split_low]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_MCC_AC2_MTR":
        split_low: float = -2.35
        split_high: float = -1.70
        low: np.ndarray = diffs[diffs < split_low]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_CH_1_MTR":
        split_low: float = -2.5
        split_high: float = -1.90
        low: np.ndarray = diffs[diffs < split_low]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_CH_2_MTR":
        split_low: float = -2.7
        split_high: float = -2.1
        low: np.ndarray = diffs[diffs < split_low]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1001" and uhm_sensor == "HAMILTON_LIB_PH_III_CH_3_MTR":
        split_low: float = -2.3
        split_high: float = -1.75
        low: np.ndarray = diffs[diffs < split_low]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1021" and uhm_sensor == "MARINE_SCIENCE_MAIN_A_MTR":
        split_low: float = -1.3
        split_high: float = -0.6
        low: np.ndarray = diffs[np.logical_and(diffs < split_low, diffs > -1.75)]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1021" and uhm_sensor == "MARINE_SCIENCE_MAIN_B_MTR":
        split_low: float = 1.1
        split_high: float = 1.75
        low: np.ndarray = diffs[np.logical_and(diffs < split_low, diffs > .75)]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    elif opq_box_id == "1021" and uhm_sensor == "MARINE_SCIENCE_MCC_MTR":
        split_low: float = -1.5
        split_high: float = -0.85
        low: np.ndarray = diffs[np.logical_and(diffs < split_low, diffs > -2)]
        mid: np.ndarray = diffs[np.logical_and(diffs >= split_low, diffs < split_high)]
        high: np.ndarray = diffs[diffs >= split_high]

        n, bins, patches = ax.hist(diffs, bins=250, density=True)

        low_p: float = float(len(low)) / float((len(low) + len(mid) + len(high)))
        mid_p: float = float(len(mid)) / float((len(low) + len(mid) + len(high)))
        high_p: float = float(len(high)) / float((len(low) + len(mid) + len(high)))

        low_bins: np.ndarray = bins[bins < split_low]
        mid_bins: np.ndarray = bins[np.logical_and(bins >= split_low, bins < split_high)]
        high_bins: np.ndarray = bins[bins >= split_high]

        low_mu = low.mean()
        low_sigma = low.std()

        mid_mu = mid.mean()
        mid_sigma = mid.std()

        high_mu = high.mean()
        high_sigma = high.std()

        y_low = normal(low_mu, low_sigma, low_bins, low_p)
        y_mid = normal(mid_mu, mid_sigma, mid_bins, mid_p)
        y_high = normal(high_mu, high_sigma, high_bins, high_p)

        ax.plot(low_bins, y_low, label=f"\n$\mu$={low_mu:.4f} $\sigma$={low_sigma:.4f}")
        ax.plot(mid_bins, y_mid, label=f"\n$\mu$={mid_mu:.4f} $\sigma$={mid_sigma:.4f}")
        ax.plot(high_bins, y_high, label=f"\n$\mu$={high_mu:.4f} $\sigma$={high_sigma:.4f}")
    else:
        n, bins, patches = ax.hist(diffs, bins=400, density=True)
        ax.plot(bins, normal(mean_diff, mean_stddev, bins, 1.0), label=f"\n$\mu$={mean_diff:.4f} $\sigma$={mean_stddev:.4f}")

    ax.set_xlabel("RMS Difference V (UHM - OPQ)")
    ax.set_ylabel("% Density")

    ax.legend()

    # fig.show()
    fig.savefig(f"{out_dir}/thd_hist_{opq_box_id}_{uhm_sensor}.png", bbox_inches='tight')


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
