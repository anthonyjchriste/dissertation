import os
from typing import Dict, List, Optional, TypeVar
import urllib.parse

import numpy as np
import pymongo
import pymongo.database

T = TypeVar("T")


def doc_get(doc: Dict, key: str, default: Optional[T] = None) -> Optional[T]:
    if key in doc:
        return doc[key]

    return default


class RedvoxReport:
    def __init__(self,
                 start_timestamp_s: int,
                 end_timestamp_s: int,
                 web_based_products: List[str],
                 report_id: str,
                 is_public: bool,
                 is_private: bool,
                 receivers: int):
        self.start_timestamp_s = start_timestamp_s
        self.end_timestamp_s = end_timestamp_s
        self.web_based_products = web_based_products
        self.report_id = report_id
        self.is_public = is_public
        self.is_private = is_private
        self.receivers = receivers

    @staticmethod
    def from_doc(doc: Dict) -> 'RedvoxReport':
        return RedvoxReport(doc_get(doc, "startTimestampS"),
                            doc_get(doc, "endTimestampS"),
                            list(map(lambda web_based_product: web_based_product["productLink"],
                                     doc["webBasedProducts"])),
                            doc_get(doc, "reportId"),
                            doc_get(doc, "isPublic"),
                            doc_get(doc, "isPrivate"),
                            doc_get(doc, "receivers"))

    def __str__(self):
        return f"{self.start_timestamp_s} {self.end_timestamp_s} {self.report_id} {self.is_public} {self.is_private} " \
               f"{self.web_based_products}"


def format_mongo_auth_uri_ssl(user: str,
                              password: str,
                              host: str,
                              port: int,
                              cert: str) -> str:
    escaped_pass = urllib.parse.quote_plus(password)
    return f"mongodb://{user}:{escaped_pass}@{host}:{port}/?ssl=true&ssl_ca_certs={cert}&replicaSet=rs0"


def get_client(host: str, port: int, user: str, password: str, cert: str) -> pymongo.MongoClient:
    return pymongo.MongoClient(format_mongo_auth_uri_ssl(user, password, host, port, cert))


def get_reports(mongo_client: pymongo.MongoClient) -> List[RedvoxReport]:
    db: pymongo.database.Database = mongo_client["redvox"]
    reports_coll: pymongo.collection.Collection = db["WebBasedReport"]

    query = {}
    projection = {"_id": False,
                  "startTimestampS": True,
                  "endTimestampS": True,
                  "webBasedProducts": True,
                  "reportId": True,
                  "isPublic": True,
                  "isPrivate": True,
                  "receivers": True}

    return list(map(RedvoxReport.from_doc, list(reports_coll.find(query, projection=projection))))


def get_stats(reports: List[RedvoxReport]) -> None:
    incidents = list(filter(lambda report: report.is_public or (report.receivers is not None and len(report.receivers) > 0), reports))
    events = list(filter(lambda report: not report.is_public, reports))
    incident_durations = np.array(list(map(lambda report: report.end_timestamp_s - report.start_timestamp_s, incidents)))
    events_durations = np.array(list(map(lambda report: report.end_timestamp_s - report.start_timestamp_s, events)))

    print(f"Total events: {len(events)}")
    print(f"Event durations sum: {events_durations.sum()}")
    print(f"Event durations mean: {events_durations.mean()}")
    print(f"Event durations std: {events_durations.std()}")
    print(f"Total incidents: {len(incidents)}")
    print(f"Incident durations sum: {incident_durations.sum()}")
    print(f"Incident durations mean: {incident_durations.mean()}")
    print(f"Incident durations std: {incident_durations.std()}")


if __name__ == "__main__":
    host = os.getenv("REDVOX_MONGO_HOST")
    port = int(os.getenv("REDVOX_MONGO_PORT"))
    user = os.getenv("REDVOX_MONGO_USER")
    passw = os.getenv("REDVOX_MONGO_PASS")
    mongo_client: pymongo.MongoClient = get_client(host, port, user, passw, "/home/ec2-user/rds-combined-ca-bundle.pem")
    reports = get_reports(mongo_client)
    get_stats(reports)

