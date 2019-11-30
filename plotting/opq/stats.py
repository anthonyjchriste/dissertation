from typing import Dict, List

import numpy as np
import pymongo
import pymongo.database


def event_stats(mongo_client: pymongo.MongoClient):
    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["events"]

    query = {}
    projection = {"_id": False,
                  "boxes_received": True,
                  "target_event_start_timestamp_ms": True,
                  "target_event_end_timestamp_ms": True}

    events: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    events: List[Dict] = list(events)

    min_ts_ms: int = min(list(map(lambda doc: doc["target_event_start_timestamp_ms"], events)))
    max_ts_ms: int = max(list(map(lambda doc: doc["target_event_end_timestamp_ms"], events)))

    durations_ms: np.ndarray = np.array(list(map(lambda doc: doc["target_event_end_timestamp_ms"] - doc["target_event_start_timestamp_ms"], events)))
    durations_s: np.ndarray = durations_ms / 1000.0
    boxes_received: np.ndarray = np.array(list(map(lambda doc: len(doc["boxes_received"]), events)))
    data_stored: np.ndarray = 12_000.0 * 2 * durations_ms * boxes_received

    total_duration_ms: int = max_ts_ms - min_ts_ms
    total_duration_s: float = total_duration_ms / 1000.0

    data_duration_s: float = durations_s.sum()
    mean_data_duration_s: float = durations_s.mean()
    std_data_duration_s: float = durations_s.std()

    total_data = data_stored.sum()
    mean_data = data_stored.mean()
    std_data = data_stored.std()
    data_per_second = total_data / total_duration_s
    events_per_second = len(events) / total_duration_s
    seconds_per_week: float = 604800.0
    seconds_per_month: float = seconds_per_week * 4

    data_per_month = (total_data * seconds_per_month) / total_duration_s

    print(f"total events {len(events)}")
    print(f"mean boxes received {boxes_received.mean()}, {boxes_received.std()}")
    print(f"total duration s {total_duration_s}")
    print(f"total data duration s {data_duration_s}")
    print(f"percent data duration {data_duration_s / total_duration_s * 100.0}")
    print(f"mean data duration s, std {mean_data_duration_s}, {std_data_duration_s}")
    print(f"total data {total_data}")
    print(f"mean data, std {mean_data}, {std_data}")
    print(f"mean data per second {data_per_second}")
    print(f"mean events per second {events_per_second}")
    print(f"mean data per month {data_per_month}")

def incident_stats(mongo_client: pymongo.MongoClient):
    db: pymongo.database.Database = mongo_client["opq"]
    coll: pymongo.collection.Collection = db["incidents"]

    query = {"start_timestamp_ms": {"$gt": 0},
             "end_timestamp_ms": {"$gt": 0}}
    projection = {"_id": False,
                  "start_timestamp_ms": True,
                  "end_timestamp_ms": True,
                  "classifications": True}

    incidents: pymongo.cursor.Cursor = coll.find(query, projection=projection)
    incidents: List[Dict] = list(incidents)

    min_ts_ms: int = min(list(map(lambda doc: doc["start_timestamp_ms"], incidents)))
    max_ts_ms: int = max(list(map(lambda doc: doc["end_timestamp_ms"], incidents)))

    durations_ms: np.ndarray = np.array(list(map(lambda doc: doc["end_timestamp_ms"] - doc["start_timestamp_ms"], incidents)))
    durations_s: np.ndarray = durations_ms / 1000.0
    data_stored: np.ndarray = 12_000.0 * 2 * durations_s

    total_duration_ms: int = max_ts_ms - min_ts_ms
    total_duration_s: float = total_duration_ms / 1000.0

    data_duration_s: float = durations_s.sum()
    mean_data_duration_s: float = durations_s.mean()
    std_data_duration_s: float = durations_s.std()

    total_data = data_stored.sum()
    mean_data = data_stored.mean()
    std_data = data_stored.std()
    data_per_second = total_data / total_duration_s
    events_per_second = len(incidents) / total_duration_s
    seconds_per_week: float = 604800.0
    seconds_per_month: float = seconds_per_week * 4
    seconds_per_year: float = seconds_per_month * 12
    data_per_year = (total_data * seconds_per_year) / total_duration_s

    print(f"total incidents {len(incidents)}")
    print(f"total duration s {total_duration_s}")
    print(f"total data duration s {data_duration_s}")
    print(f"percent data duration {data_duration_s / total_duration_s * 100.0}")
    print(f"mean data duration s, std {mean_data_duration_s}, {std_data_duration_s}")
    print(f"total data {total_data}")
    print(f"mean data, std {mean_data}, {std_data}")
    print(f"mean data per second {data_per_second}")
    print(f"mean events per second {events_per_second}")
    print(f"mean data per year {data_per_year}")



if __name__ == "__main__":
    mongo_client = pymongo.MongoClient()
    event_stats(mongo_client)
    print()
    incident_stats(mongo_client)

