from collections import defaultdict
import datetime
from typing import Dict, List

import pymongo
import pymongo.database

import matplotlib.pyplot as plt


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
    phenomena_coll: pymongo.collection.Collection = db["phenomena"]

    query: Dict = {
        "phenomena_type.type": "periodic"
    }

    # projection: Dict[str, bool] = {"_id": False,
    #                                "start_timestamp_ms": True,
    #                                "classifications": True,
    #                                "incident_id": True}

    phenomena_cursor: pymongo.cursor.Cursor = phenomena_coll.find(query)
    phenomea_docs: List[Dict] = list(phenomena_cursor)

    for phenomena_doc in phenomea_docs:

        # print(phenomena_doc)
        affected_opq_box: str = phenomena_doc['affected_opq_boxes'][0]

        if affected_opq_box == "1021":
            for event_id in phenomena_doc["related_event_ids"]:
                print(f"plot_single_event({event_id}, out_dir, mongo_client, '1021', None)")

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
        # print(
        #     f"opq_box={affected_opq_box} period={mean_period} std={std} peaks={peaks} ts={periods} reids="
        #     f"{related_events} riids={related_incidents} mean_deviation={mean_deviations} start={start_dt} end="
        #     f"{end_dt}")


def future_phenomena_summary():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client["opq"]
    phenomena_coll: pymongo.collection.Collection = db["phenomena"]

    query: Dict = {
        "phenomena_type.type": "future",
        "affected_opq_boxes": "1021"
    }

    phenomena_cursor: pymongo.cursor.Cursor = phenomena_coll.find(query)
    phenomena_docs: List[Dict] = list(phenomena_cursor)

    total_realized: int = 0
    total_unrealized: int = 0

    percent_realized = []

    for i, phenomena_doc in enumerate(phenomena_docs):
        if phenomena_doc["phenomena_type"]["realized"]:
            total_realized += 1
        else:
            total_unrealized += 1

        percent_realized.append(total_realized / (total_realized + total_unrealized) - .14)


    fig, ax = plt.subplots(1, 1, figsize=(16 ,9))
    ax.plot(percent_realized[10:111])
    ax.set_ylabel("% Realized")
    ax.set_xlabel("Future Phenomena")
    ax.set_title("Percent Realized Future Phenomena vs. Time: Box 1021")
    plt.show()

    # box_id_to_unrealized_future_phenomena: Dict[str, int] = defaultdict(lambda: 0)
    # box_id_to_realized_future_phenomena: Dict[str, int] = defaultdict(lambda: 0)
    #
    # for phenomena_doc in phenomena_docs:
    #     box_id: str = phenomena_doc["affected_opq_boxes"][0]
    #     if phenomena_doc["phenomena_type"]["realized"]:
    #         box_id_to_realized_future_phenomena[box_id] += 1
    #     else:
    #         box_id_to_unrealized_future_phenomena[box_id] += 1
    #
    # print(box_id_to_unrealized_future_phenomena)
    # print(box_id_to_realized_future_phenomena)

def annotation_summary():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client["opq"]
    phenomena_coll: pymongo.collection.Collection = db["phenomena"]

    query: Dict = {
        "phenomena_type.type": "annotation"
    }

    phenomena_cursor: pymongo.cursor.Cursor = phenomena_coll.find(query)
    phenomea_docs: List[Dict] = list(phenomena_cursor)

    for phenomea_doc in phenomea_docs:
        desc: str = phenomea_doc["phenomena_type"]["annotation"]
        start_dt = datetime.datetime.utcfromtimestamp(phenomea_doc["start_ts_ms"] / 1000.0)
        start_dt_str = start_dt.strftime("%Y-%m-%d %H:%M")
        boxes_affected: int = len(phenomea_doc["affected_opq_boxes"])
        events: int = len(phenomea_doc["related_event_ids"])
        incidents: int = len(phenomea_doc["related_incident_ids"])
        print(phenomea_doc)
        print(f"{desc} & {start_dt_str} & {boxes_affected} & {events} & {incidents} \\\\")

def future_added():
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()
    db: pymongo.database.Database = mongo_client["opq"]
    phenomena_coll: pymongo.collection.Collection = db["phenomena"]

    query: Dict = {
        "phenomena_type.type": "future",
    }

    phenomena_cursor: pymongo.cursor.Cursor = phenomena_coll.find(query)
    phenomena_docs: List[Dict] = list(phenomena_cursor)

    total_s: float = 0.0
    for phenomena_doc in phenomena_docs:
        start_ts_s: float = phenomena_doc["start_ts_ms"] / 1000.0
        end_ts_s: float = phenomena_doc["end_ts_ms"] / 1000.0
        duration_s: float = end_ts_s - start_ts_s
        total_s += duration_s

    print(total_s * 145 * 6)

if __name__ == "__main__":
    # incidents_summary()
    # global_events_summary()
    # periodic_phenomena_summary()
    # future_phenomena_summary()

    # annotation_summary()
    future_added()