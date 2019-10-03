import numpy as np
import pymongo

if __name__ == "__main__":
    mongo_client = pymongo.MongoClient()
    db = mongo_client.opq

    # Boxes per event
    events = db.events.find(projection={"target_event_start_timestamp_ms": True})
    ts_ms = []
    diffs = []

    for event in events:
        s = event["target_event_start_timestamp_ms"]
        ts_ms.append(s)

    for i in range(1, len(ts_ms)):
        p = ts_ms[i - 1]
        n = ts_ms[i]
        if n - p == 0:
            continue
        diffs.append(1.0 / ((n - p) / 1000.0))

    num_boxes = np.array(diffs)
    print("mean", num_boxes.mean())
    print("stddev", num_boxes.std())