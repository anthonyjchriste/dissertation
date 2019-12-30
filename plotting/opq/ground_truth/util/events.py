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


def events(opq_start_ts_s: int,
           opq_end_ts_s: int,
           opq_box_id: str,
           ground_truth_root: str,
           uhm_sensor: str,
           mongo_client: pymongo.MongoClient,
           out_dir: str) -> None:
    # if opq_box_id != "1021" and uhm_sensor != "MARINE_SCIENCE_MCC_MTR":
    #     return


    ground_truth_path_f: str = f"{ground_truth_root}/{uhm_sensor}/Frequency"
    uhm_data_points_f: List[io.DataPoint] = io.parse_file(ground_truth_path_f)

    ground_truth_path_thd: str = f"{ground_truth_root}/{uhm_sensor}/AVG_VOLTAGE_THD"
    uhm_data_points_thd: List[io.DataPoint] = io.parse_file(ground_truth_path_thd)

    ground_truth_path_vab: str = f"{ground_truth_root}/{uhm_sensor}/VAB"
    uhm_data_points_vab: List[io.DataPoint] = io.parse_file(ground_truth_path_vab)

    ground_truth_path_vbc: str = f"{ground_truth_root}/{uhm_sensor}/VBC"
    uhm_data_points_vbc: List[io.DataPoint] = io.parse_file(ground_truth_path_vbc)

    ground_truth_path_vca: str = f"{ground_truth_root}/{uhm_sensor}/VCA"
    uhm_data_points_vca: List[io.DataPoint] = io.parse_file(ground_truth_path_vca)

    series: List[align.SeriesSpec] = [
        align.SeriesSpec(uhm_data_points_f,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.min_v),
        align.SeriesSpec(uhm_data_points_f,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.max_v),
        align.SeriesSpec(uhm_data_points_thd,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.min_v),
        align.SeriesSpec(uhm_data_points_thd,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.max_v),
        align.SeriesSpec(uhm_data_points_vab,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.min_v),
        align.SeriesSpec(uhm_data_points_vab,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.max_v),
        align.SeriesSpec(uhm_data_points_vbc,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.min_v),
        align.SeriesSpec(uhm_data_points_vbc,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.max_v),
        align.SeriesSpec(uhm_data_points_vca,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.min_v),
        align.SeriesSpec(uhm_data_points_vca,
                         lambda data_point: datetime.datetime.utcfromtimestamp(data_point.ts_s),
                         lambda data_point: data_point.max_v),
    ]

    aligned_data: List[Tuple[np.ndarray, np.ndarray]] = align.align_data_multi(series)

    f_min_data: Tuple[np.ndarray, np.ndarray] = aligned_data[0]
    f_max_data: Tuple[np.ndarray, np.ndarray] = aligned_data[1]
    thd_min_data: Tuple[np.ndarray, np.ndarray] = aligned_data[2]
    thd_max_data: Tuple[np.ndarray, np.ndarray] = aligned_data[3]
    vab_min_data: Tuple[np.ndarray, np.ndarray] = aligned_data[4]
    vab_max_data: Tuple[np.ndarray, np.ndarray] = aligned_data[5]
    vbc_min_data: Tuple[np.ndarray, np.ndarray] = aligned_data[6]
    vbc_max_data: Tuple[np.ndarray, np.ndarray] = aligned_data[7]
    vca_min_data: Tuple[np.ndarray, np.ndarray] = aligned_data[8]
    vca_max_data: Tuple[np.ndarray, np.ndarray] = aligned_data[9]

    dts: np.ndarray = f_min_data[0]
    f_mins: np.ndarray = f_min_data[1]
    f_maxes: np.ndarray = f_max_data[1]
    thd_mins: np.ndarray = thd_min_data[1]
    thd_maxes: np.ndarray = thd_max_data[1]
    vab_mins: np.ndarray = vab_min_data[1]
    vab_maxes: np.ndarray = vab_max_data[1]
    vbc_mins: np.ndarray = vbc_min_data[1]
    vbc_maxes: np.ndarray = vbc_max_data[1]
    vca_mins: np.ndarray = vca_min_data[1]
    vca_maxes: np.ndarray = vca_max_data[1]

    # thd_maxes += 0.7


    eq_left: float = (1.0 / (np.sqrt(3) * 3.9985))
    sq_sum_min = np.square(vab_mins) + np.square(vbc_mins) + np.square(vca_mins)
    vrms_vals_min: np.ndarray = eq_left * np.sqrt(sq_sum_min)
    sq_sum_max = np.square(vab_maxes) + np.square(vbc_maxes) + np.square(vca_maxes)
    vrms_vals_max: np.ndarray = eq_left * np.sqrt(sq_sum_max)

    # vrms_max_adj = vrms_vals_max + 1.7
    # vrms_min_adj = vrms_vals_min + 1.7

    # OPQ data

    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["events"]

    query: Dict = {
        "target_event_start_timestamp_ms": {"$gte": opq_start_ts_s * 1000,
                                            "$lte": opq_end_ts_s * 1000},
        "$and": [{"boxes_triggered": {"$size": 1}},
                 {"boxes_triggered": opq_box_id},
                 {"boxes_received": {"$size": 1}},
                 {"boxes_received": opq_box_id}],
    }

    projection: Dict[str, bool] = {
        "_id": False,
        "target_event_start_timestamp_ms": True,
        "boxes_triggered": True,
        "boxes_received": True,
    }

    events_cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    event_docs: List[Dict] = list(events_cursor)

    event_dts_all = list(map(lambda doc: datetime.datetime.utcfromtimestamp(doc["target_event_start_timestamp_ms"] / 1000.0), event_docs))
    binned_event_dts = set(map(align.bin_dt_by_min, event_dts_all))
    event_dts = list(binned_event_dts)
    print(f"event_dts_all={len(event_dts_all)} num_event_dts={len(event_dts)}")

    # Plot
    fig, ax = plt.subplots(3, 1, figsize=(16, 9), sharex="all")
    fig: plt.Figure = fig
    ax: List[plt.Axes] = ax

    fig.suptitle(
            f"Grount Truth "
            f"({dts[0].strftime('%m-%d')} to "
            f"{dts[-1].strftime('%m-%d')})"
            f"\n{opq_box_id} vs {uhm_sensor}"
    )

    # Frequency
    f_min_thresh: float = 60.0 - (60.0 * 0.0016)
    f_max_thresh: float = 60.0 + (60.0 * 0.0016)
    f_ax: plt.Axes = ax[0]
    f_ax.plot(dts, f_mins, label="Min. Frequency")
    f_ax.plot(dts, f_maxes, label="Max Frequency")
    f_ax.plot(dts, [f_min_thresh for _ in dts], label="Min Voltage Threshold", linestyle="--")
    f_ax.plot(dts, [f_max_thresh for _ in dts], label="Max Voltage Threshold", linestyle="--")
    f_ax.set_ylabel("Hz")

    f_ax.set_title("Frequency")

    num_f_min = len(f_mins[f_mins <= f_min_thresh])
    num_f_max = len(f_maxes[f_maxes >= f_max_thresh])
    print(f"{opq_box_id} {uhm_sensor} f_above_max={num_f_max} f_below_min={num_f_min} f_total={num_f_min + num_f_max}")

    # f_ax.scatter(event_dts, [60.0 for _ in event_dts], color="red")

    f_ax.legend()

    # Voltage
    v_min_thresh: float = 120.0 - (120.0 * 0.025)
    v_max_thresh: float = 120.0 + (120.0 * 0.025)
    v_ax: plt.Axes = ax[1]
    v_ax.plot(dts, vrms_vals_min, label="Min. Voltage")
    v_ax.plot(dts, vrms_vals_max, label="Max Voltage")
    # v_ax.plot(dts, vrms_max_adj, label="Adjusted", linestyle=":", color="red")
    v_ax.plot(dts, [v_min_thresh for _ in dts], label="Min Voltage Threshold", linestyle="--")
    v_ax.plot(dts, [v_max_thresh for _ in dts], label="Max Voltage Threshold", linestyle="--")
    v_ax.set_title("Voltage")
    v_ax.set_ylabel("RMS")

    num_v_min = len(vrms_vals_min[vrms_vals_min <= v_min_thresh])
    num_v_max = len(vrms_vals_max[vrms_vals_max >= v_max_thresh])
    print(f"{opq_box_id} {uhm_sensor} v_above_max={num_v_max} v_below_min={num_v_min} v_total={num_v_min + num_v_max}")

    # v_ax.scatter(event_dts, [120.0 for _ in event_dts], color="red")

    v_ax.legend()

    # THD
    thd_ax: plt.Axes = ax[2]
    thd_ax.plot(dts, thd_mins, label="Min. THD")
    thd_ax.plot(dts, thd_maxes, label="Max THD")
    thd_ax.plot(dts, [3.0 for _ in dts], label="THD Threshold", linestyle="--")
    thd_ax.set_title("THD")
    thd_ax.set_ylabel("% THD")
    thd_ax.set_xlabel("Time (UTC)")

    thd_above = len(thd_maxes[thd_maxes > 3.0])
    zero_crossings = np.where(np.diff(np.sign(thd_maxes - 3.0)))[0]
    print(f"THD above={thd_above} zeroes={len(zero_crossings)}")

    thd_ax.legend()

    # fig.show()
    fig.savefig(f"{out_dir}/gt_all_{opq_box_id}_{uhm_sensor}.png", bbox_inches='tight')

    # fig2, ax2 = plt.subplots(1, 1, figsize=(16, 9))
    # ax2.plot(dts, thd_maxes + .7, label="Adjusted Max THD")
    # ax2.plot(dts, [3.0 for _ in dts], label="THD Threshold", linestyle="--")
    # # ax2.plot(dts, vrms_min_adj, label="Adjusted Min. Voltage")
    # # ax2.plot(dts, vrms_max_adj, label="Adjusted Max Voltage")
    # # ax2.plot(dts, [v_min_thresh for _ in dts], label="Min Voltage Threshold", linestyle="--")
    # # ax2.plot(dts, [v_max_thresh for _ in dts], label="Max Voltage Threshold", linestyle="--")
    # ax2.set_ylabel("% THD")
    # ax2.set_xlabel("Time (UTC)")
    #
    # ax2.set_title(
    #         f"Grount Truth Adjusted THD "
    #         f"({dts[0].strftime('%m-%d')} to "
    #         f"{dts[-1].strftime('%m-%d')})"
    #         f"\n{opq_box_id} vs {uhm_sensor}"
    # )
    #
    # # zero_crossings = np.where(np.diff(np.sign(vrms_max_adj - v_max_thresh)))[0]
    # # print(f"adj vmax zeros={len(zero_crossings)}")
    #
    # ax2.legend()
    #
    # # fig2.show()
    # fig2.savefig(f"{out_dir}/gt_adj_{opq_box_id}_{uhm_sensor}.png", bbox_inches='tight')


def compare_events(opq_start_ts_s: int,
                   opq_end_ts_s: int,
                   ground_truth_root: str,
                   mongo_client: pymongo.MongoClient,
                   out_dir: str):
    paths: List[str] = []

    for opq_box, uhm_meters in util.opq_box_to_uhm_meters.items():
        for uhm_meter in uhm_meters:
            try:
                print(f"plot_event {opq_box} {uhm_meter}")
                path = events(opq_start_ts_s,
                                          opq_end_ts_s,
                                          opq_box,
                                          ground_truth_root,
                                          uhm_meter,
                                          mongo_client,
                                          out_dir)
                paths.append(path)
            except Exception as e:
                print(e, "...ignoring...")
