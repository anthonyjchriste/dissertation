from collections import defaultdict
from typing import Dict, List

import pymongo
import pymongo.database


def global_events_summary():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client["opq"]
    events_coll: pymongo.collection.Collection = db["events"]

    query: Dict = {
        "target_event_start_timestamp_ms": {"$gt": 1569888000000}
    }

    projection: Dict[str, bool] = {"_id": False,
                                   "target_event_start_timestamp_ms": True,
                                   "boxes_received": True}

    events_cursor: pymongo.cursor.Cursor = events_coll.find(query, projection=projection)
    event_docs: List[Dict] = list(events_cursor)

    num_boxes_received_to_num_events: Dict[int, int] = defaultdict(lambda: 0)
    for event_doc in event_docs:
        num_boxes_received: int = len(event_doc["boxes_received"])
        num_boxes_received_to_num_events[num_boxes_received] += 1

    total_incidents = []
    for k, v in num_boxes_received_to_num_events.items():
        total_incidents.append(v)
        print(f"{v} & {k} \\")

    print(sum(total_incidents))




def incidents_summary():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client["opq"]
    incidents_coll: pymongo.collection.Collection = db["incidents"]

    query: Dict = {
        "start_timestamp_ms": {"$gt": 1569888000000}
    }

    projection: Dict[str, bool] = {"_id": False,
                                   "start_timestamp_ms": True,
                                   "classifications": True,
                                   "incident_id": True}

    incidents_cursor: pymongo.cursor.Cursor = incidents_coll.find(query, projection=projection)
    incident_docs: List[Dict] = list(incidents_cursor)

    incident_classification_to_cnt: Dict[str, int] = defaultdict(lambda: 0)

    for incident_doc in incident_docs:
        classification: str = incident_doc["classifications"][0]
        incident_classification_to_cnt[classification] += 1

        if classification == "VOLTAGE_INTERRUPTION":
            print(incident_doc["incident_id"])

        # if classification == "FREQUENCY_SWELL":
        #     incident_id: int = incident_doc["incident_id"]
        #     print(f"plot_voltage_incident({incident_id}, '.', mongo_client)")

    for k, v in incident_classification_to_cnt.items():
        print(f"{k} & {v} & {v / 90.0:.2f}")


if __name__ == "__main__":
    incidents_summary()
    # global_events_summary()