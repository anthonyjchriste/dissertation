import numpy as np
import pymongo

if __name__ == "__main__":
    mongo_client = pymongo.MongoClient()
    db = mongo_client.opq

    # Events
    events =  db.events.find({"target_event_start_timestamp_ms": {"$gt": 0}},
                             projection={"target_event_start_timestamp_ms": True,
                                         "target_event_end_timestamp_ms": True,
                                         "boxes_received": True})

    total_events = 0
    event_lens = []
    boxes_received = []
    ts_ms = []
    diffs = []

    for event in events:
        event_lens.append((event["target_event_end_timestamp_ms"] - event["target_event_start_timestamp_ms"]) / 1000.0)
        boxes_received.append(len(event["boxes_received"]))
        ts_ms.append(event["target_event_end_timestamp_ms"])
        total_events += 1

    for i in range(1, len(ts_ms)):
        p = ts_ms[i - 1]
        n = ts_ms[i]
        if n - p == 0:
            continue
        diffs.append(1.0 / ((n - p) / 1000.0))

    event_lens = np.array(event_lens)
    boxes_received = np.array(boxes_received)
    diffs = np.array(diffs)
    print("total events", total_events)
    print("mean/std event len", event_lens.mean(), event_lens.std())
    print("mean/std boxes recv", boxes_received.mean(), boxes_received.std())
    print("mean/std event rate", diffs.mean(), diffs.std())

    # Incidents
    incidents = db.incidents.find(projection={"start_timestamp_ms": True,
                                              "end_timestamp_ms": True,
                                              "classifications": True})
    incident_lens = []
    ts_ms = []
    diffs = []
    total_incidents = 0

    for incident in incidents:
        le = (incident["end_timestamp_ms"] - incident["start_timestamp_ms"]) / 1000.0
        if le < 0 or "OUTAGE" in incident["classifications"]:
            continue

        incident_lens.append(le)
        ts_ms.append(incident["start_timestamp_ms"])
        total_incidents += 1

    for i in range(1, len(ts_ms)):
        p = ts_ms[i - 1]
        n = ts_ms[i]
        if n - p == 0:
            continue
        diffs.append(1.0 / ((n - p) / 1000.0))

    incident_lens = np.array(incident_lens)
    diffs = np.array(diffs)
    print("total incidents",total_incidents )
    print("mean/std incident len", incident_lens.mean(), incident_lens.std())
    print("mean/std incident rate", diffs.mean(), diffs.std())