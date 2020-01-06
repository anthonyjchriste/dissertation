from collections import defaultdict
import datetime
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


def periodic_phenomena_summary():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client["opq"]
    incidents_coll: pymongo.collection.Collection = db["phenomena"]

    query: Dict = {
        "phenomena_type.type": "periodic"
    }

    # projection: Dict[str, bool] = {"_id": False,
    #                                "start_timestamp_ms": True,
    #                                "classifications": True,
    #                                "incident_id": True}

    phenomena_cursor: pymongo.cursor.Cursor = incidents_coll.find(query)
    phenomea_docs: List[Dict] = list(phenomena_cursor)

    for phenomena_doc in phenomea_docs:
        # print(phenomena_doc)
        affected_opq_box: str = phenomena_doc['affected_opq_boxes'][0]
        mean_period: float = phenomena_doc['phenomena_type']['period_s']
        std: float = phenomena_doc['phenomena_type']['std_s']
        periods: int = len(phenomena_doc['phenomena_type']['period_timestamps'])
        peaks: int = phenomena_doc['phenomena_type']['peaks']
        related_incidents: int = len(phenomena_doc['related_incident_ids'])
        related_events: int = len(phenomena_doc['related_event_ids'])
        deviation_from_mean_values: List[float] = phenomena_doc['phenomena_type']['deviation_from_mean_values']
        mean_deviations: float = sum(deviation_from_mean_values) / len(deviation_from_mean_values)
        start_dt = datetime.datetime.utcfromtimestamp(phenomena_doc["start_ts_ms"] / 1000.0)
        end_dt = datetime.datetime.utcfromtimestamp(phenomena_doc["end_ts_ms"] / 1000.0)
        print(f"opq_box={affected_opq_box} period={mean_period} std={std} peaks={peaks} ts={periods} reids={related_events} riids={related_incidents} mean_deviation={mean_deviations} start={start_dt} end={end_dt}")


if __name__ == "__main__":
    # incidents_summary()
    # global_events_summary()
    periodic_phenomena_summary()
