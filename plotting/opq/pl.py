import datetime
from typing import Dict, List

import bson
import matplotlib.pyplot as plt
import numpy as np
import pymongo
import pymongo.database
from scipy import stats


def sum_series(series: np.ndarray) -> np.ndarray:
    result: List[int] = [series[0]]
    for i in range(1, len(series)):
        result.append(result[i - 1] + series[i])

    return np.array(result)


def slope_intercept(slope: float, intercept: float) -> str:
    return f"y = {slope} * x + {intercept}"


def pl_vs_sim():
    pass


def pl_vs_est():
    pass


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
            label=f"PL Total MB LR ($m$={slope:.5f} $b$={intercept:.5f} $R^2$={r_value ** 2:.5f} $\sigma$={std_err:.5f})")

    ax.set_title("Actual Phenomena: OPQ")
    ax.set_ylabel("Size (MB)")
    ax.set_xlabel("Time (UTC)")
    ax.legend()

    fig.show()
    fig.savefig("/Users/anthony/Development/dissertation/src/figures/actual_phenomena_opq.png")


def main():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    actual_pl(mongo_client)


if __name__ == "__main__":
    main()
