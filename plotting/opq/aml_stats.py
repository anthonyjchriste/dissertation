from typing import *

import numpy as np
import pymongo
import pymongo.database

if __name__ == "__main__":
    mongo_client = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client["opq"]
    events_coll: pymongo.collection.Collection = db["events"]
    box_events_coll: pymongo.collection.Collection = db["box_events"]
    incidents_coll: pymongo.collection.Collection = db["incidents"]

    # Statistics for events
    events_query = {}
    events_projection = {"_id": False,
                         "event_id": True}

    events: pymongo.cursor.Cursor = events_coll.find(events_query, projection=events_projection)
    events: List[Dict] = list(events)
    event_ids: List[int] = list(map(lambda event: event["event_id"], events))

    box_events_query = {"event_id": {"$in": event_ids}}
    box_events_projection: Dict[str, bool] = {"_id": False,
                                              "event_start_timestamp_ms": True,
                                              "event_end_timestamp_ms": True,
                                              "event_id": True}

    box_events: pymongo.cursor.Cursor = box_events_coll.find(box_events_query, projection=box_events_projection)
    box_events: List[Dict] = list(box_events)

    event_min_ts_ms: int = min(list(map(lambda doc: doc["event_start_timestamp_ms"], box_events)))
    event_max_ts_ms: int = max(list(map(lambda doc: doc["event_end_timestamp_ms"], box_events)))
    event_min_ts_s: float = event_min_ts_ms / 1000.0
    event_max_ts_s: float = event_max_ts_ms / 1000.0

    event_durations_ms: np.ndarray = np.array(
            list(map(lambda doc: doc["event_end_timestamp_ms"] - doc["event_start_timestamp_ms"], box_events)))
    event_durations_s: np.ndarray = event_durations_ms / 1000.0

    event_total_duration_ms: int = event_max_ts_ms - event_min_ts_ms
    event_total_duration_s: float = event_total_duration_ms / 1000.0

    event_data_duration_s: float = event_durations_s.sum()
    event_mean_data_duration_s: float = event_durations_s.mean()
    events_per_second = len(box_events) / event_total_duration_s

    event_data_stored: np.ndarray = 12_000.0 * 2 * event_durations_s
    event_total_data = event_data_stored.sum()
    event_mean_data = event_data_stored.mean()
    event_data_per_second = event_total_data / event_total_duration_s

    incident_query = {"event_id": {"$gte": 0},
                      "start_timestamp_ms": {"$gte": 0},
                      "end_timestamp_ms": {"$gte": 0}}

    incident_projection = {"_id": False,
                           "start_timestamp_ms": True,
                           "end_timestamp_ms": True,
                           "event_id": True}

    incidents: List[Dict] = list(incidents_coll.find(incident_query, projection=incident_projection))
    incident_event_ids: Set[int] = set(map(lambda incident: incident["event_id"], incidents))

    total_events_to_incidents: int = 0
    for event in events:
        if event["event_id"] in incident_event_ids:
            total_events_to_incidents += 1

    print(f"total events {len(events)}")
    print(f"total duration s {event_total_duration_s}")
    print(f"total data duration s {event_data_duration_s}")
    print(f"percent data duration {event_data_duration_s / event_total_duration_s}")
    print(f"mean data duration s, std {event_mean_data_duration_s}")
    print(f"total data {event_total_data}")
    print(f"mean data, std {event_mean_data}")
    print(f"mean data per second {event_data_per_second}")
    print(f"mean events per second {events_per_second}")
    print(f"percent events to incident {total_events_to_incidents / float(len(events))}")

    incident_min_ts_ms: int = min(list(map(lambda doc: doc["start_timestamp_ms"], incidents)))
    incident_max_ts_ms: int = max(list(map(lambda doc: doc["end_timestamp_ms"], incidents)))
    incident_min_ts_s: float = incident_min_ts_ms / 1000.0
    incident_max_ts_s: float = incident_max_ts_ms / 1000.0

    incident_durations_ms: np.ndarray = np.array(
            list(map(lambda doc: doc["end_timestamp_ms"] - doc["start_timestamp_ms"], incidents)))
    incident_durations_s: np.ndarray = incident_durations_ms / 1000.0

    incident_total_duration_ms: int = incident_max_ts_ms - incident_min_ts_ms
    incident_total_duration_s: float = incident_total_duration_ms / 1000.0

    incident_data_duration_s: float = incident_durations_s.sum()
    incident_mean_data_duration_s: float = incident_durations_s.mean()
    incidents_per_second = len(incidents) / incident_total_duration_s

    incident_data_stored: np.ndarray = 12_000.0 * 2 * incident_durations_s
    incident_total_data = incident_data_stored.sum()
    incident_mean_data = incident_data_stored.mean()
    incident_data_per_second = incident_total_data / incident_total_duration_s
    print()
    print(f"total incidents {len(incidents)}")
    print(f"total duration s {event_total_duration_s}")
    print(f"total data duration s {event_data_duration_s}")
    print(f"percent data duration {event_data_duration_s / event_total_duration_s}")
    print(f"mean data duration s, std {event_mean_data_duration_s}")
    print(f"total data {event_total_data}")
    print(f"mean data, std {event_mean_data}")
    print(f"mean data per second {event_data_per_second}")
    print(f"mean incidents per second {events_per_second}")
