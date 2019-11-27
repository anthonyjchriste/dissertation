import numpy as np
import pymongo
import pymongo.database
import sys

if __name__ == "__main__":
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client.opq

    # Measurements
    stats = db.command("collstats", "measurements")
    print(f"measurements mean size={stats['avgObjSize']}")

    # Trends
    stats = db.command("collstats", "trends")
    print(f"trends mean size={stats['avgObjSize']}")

    sys.exit(0)

    # Events
    events = db.events.find({"target_event_start_timestamp_ms": {"$gt": 0},
                             "target_event_end_timestamp_ms": {"$gt": 0}},
                            projection={"target_event_start_timestamp_ms": True,
                                        "target_event_end_timestamp_ms": True,
                                        "boxes_received": True})
    events_bytes = []
    timestamps = []

    for event in events:
        timestamps.append(event["target_event_start_timestamp_ms"])
        timestamps.append(event["target_event_end_timestamp_ms"])
        range_s = (event["target_event_end_timestamp_ms"] - event["target_event_start_timestamp_ms"]) / 1000.0
        event_bytes = range_s * 12_000 * 2 * len(event["boxes_received"])
        events_bytes.append(event_bytes)

    total_bytes = sum(events_bytes)
    total_s = (max(timestamps) - min(timestamps)) / 1000.0
    mean_dr = total_bytes / total_s
    sigma_dr = np.sqrt((1.0/total_s) * sum([(x - mean_dr)**2 for x in events_bytes]))

    print("event total_bytes", total_bytes)
    print("event total_s", total_s)
    print("event mean_dr", mean_dr)
    print("event sigma_dr", sigma_dr)

    incidents = db.incidents.find(projection={"start_timestamp_ms": True,
                                              "end_timestamp_ms": True,
                                              "classifications": True})
    incidents_bytes = []
    for incident in incidents:
        le = (incident["end_timestamp_ms"] - incident["start_timestamp_ms"]) / 1000.0
        if le < 0 or "OUTAGE" in incident["classifications"]:
            continue
        incidents_bytes.append(le * 12_000 * 2)

    total_bytes = sum(incidents_bytes)
    mean_ir = total_bytes / total_s
    sigma_ir = np.sqrt((1.0/total_s) * sum([(x - mean_ir)**2 for x in incidents_bytes]))

    print("incident total_bytes", total_bytes)
    print("incident mean_dr", mean_ir)
    print("incident sigma_dr", sigma_ir)
