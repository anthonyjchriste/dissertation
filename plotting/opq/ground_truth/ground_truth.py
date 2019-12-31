import datetime
import glob
import os.path
from typing import List

import numpy as np
import pymongo

import util.frequency as frequency
import util.io as io
import util.thd as thd
import util.voltage as voltage
import util.events as events


def find_ground_truth_data_range(ground_truth_root: str):
    all_files: List[str] = glob.glob(os.path.join(ground_truth_root, "*", "*"), recursive=True)
    files: List[str] = list(filter(os.path.isfile, all_files))

    data_points: List[io.DataPoint] = []

    for file in files:
        file_data_points: List[io.DataPoint] = io.parse_file(file)
        data_points.extend(file_data_points)

    timestamps = np.array(list(map(lambda data_point: data_point.ts_s, data_points)))
    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()

    print(f"min_ts {min_timestamp} {datetime.datetime.utcfromtimestamp(min_timestamp)}")
    print(f"min_ts {max_timestamp} {datetime.datetime.utcfromtimestamp(max_timestamp)}")


def main():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    start_ts_s: int = 1574977260
    end_ts_s: int = 1577664000
    gt_root: str = "/Users/anthony/scrap/ground_truth_data2"
    # gt_root: str = "/Users/anthony/Desktop/forAnthony/data"
    out_dir: str = "/Users/anthony/Development/dissertation/src/figures"

    # find_ground_truth_data_range(gt_root)

    # frequency.compare_frequencies(start_ts_s, end_ts_s, gt_root, mongo_client, out_dir)
    # thd.compare_thds(start_ts_s, end_ts_s, gt_root, mongo_client, out_dir)
    # voltage.compare_vrms(start_ts_s, end_ts_s, gt_root, mongo_client, out_dir)

    frequency.compare_frequency_incidents(start_ts_s, end_ts_s, gt_root, mongo_client, out_dir)
    # voltage.compare_vrms_incidents(start_ts_s, end_ts_s, gt_root, mongo_client, out_dir)
    # thd.compare_thd_incidents(start_ts_s, end_ts_s, gt_root, mongo_client, out_dir)
    # thd.plot_thd_incidents(start_ts_s, end_ts_s, "1001", gt_root, "HAMILTON_LIB_PH_III_CH_1_MTR", mongo_client,
    # out_dir)

    # events.compare_events(start_ts_s, end_ts_s, gt_root, mongo_client, out_dir)


if __name__ == "__main__":
    main()
