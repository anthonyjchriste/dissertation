import os
from typing import Dict, List, Optional, TypeVar
import urllib.parse

import boto3
import botocore
import numpy as np
import pymongo
import pymongo.database

T = TypeVar("T")


def doc_get(doc: Dict, key: str, default: Optional[T] = None) -> Optional[T]:
    if key in doc:
        return doc[key]

    return default


def get_sample_rate(mongo_client: pymongo.MongoClient, id_uuid: str, start_ts_s: int, end_ts_s: int) -> float:
    db: pymongo.database.Database = mongo_client["redvox"]
    coll: pymongo.collection.Collection = db["RedvoxPacketApi900"]
    s = id_uuid.split(":")
    redvox_id = s[0]
    redvox_uuid = s[1]
    query = {"appFileStartTimestampMachine": {"$gte": start_ts_s * 1_000_000.0,
                                              "$lte": end_ts_s * 1_000_000.0},
             "redvoxId": redvox_id,
             "redvoxUuid": redvox_uuid}
    projection = {"_id": False,
                  "appFileStartTimestampMachine": True,
                  "redvoxId": True,
                  "redvoxUuid": True,
                  "evenlySampledChannels": True}

    packet = coll.find_one(query, projection=projection)
    return packet["evenlySampledChannels"][0]["sampleRateHz"] if packet is not None else 0

class RedvoxReport:
    def __init__(self,
                 start_timestamp_s: int,
                 end_timestamp_s: int,
                 web_based_products: List[str],
                 report_id: str,
                 is_public: bool,
                 is_private: bool,
                 receivers: int,
                 devices: List[str],
                 mongo_client: pymongo.MongoClient,
                 s3_client):
        self.start_timestamp_s = start_timestamp_s
        self.end_timestamp_s = end_timestamp_s
        self.web_based_products = web_based_products
        self.report_id = report_id
        self.is_public = is_public
        self.is_private = is_private
        self.receivers = receivers
        self.devices = devices
        self.estimated_data_bytes = self.__data_bytes(mongo_client, s3_client)
        self.product_bytes = self.__product_bytes(s3_client)
        self.total_bytes = self.estimated_data_bytes + self.product_bytes
        self.is_estimated = False

    def duration(self) -> int:
        return self.end_timestamp_s - self.start_timestamp_s

    def __data_bytes(self, mongo_client: pymongo.MongoClient, s3_client) -> int:
        try:
            k = f"report_data/{self.report_id}.zip"
            res = s3_client.head_object(Bucket="rdvxdata", Key=k)
            c = res["ContentLength"]
            # print(k, c)
            return c
        except Exception as e:
            # print("Associated data not found, falling back to estimation")
            return self.__estimated_data_bytes(mongo_client)

    def __estimated_data_bytes(self, mongo_client: pymongo.MongoClient) -> int:
        self.is_estimated = True
        total_bytes = 0
        bytes_per_sample = 4
        for id_uuid in self.devices:
            total_bytes += get_sample_rate(mongo_client, id_uuid, self.start_timestamp_s, self.end_timestamp_s) * bytes_per_sample * self.duration()

        return total_bytes

    def __product_bytes(self, s3_client) -> int:
        total_bytes = 0
        for product in self.web_based_products:
            try:
                k = f"reports/{product}"
                res = s3_client.head_object(Bucket="rdvxproducts", Key=k)
                c = res["ContentLength"]
                total_bytes += c
                # print(k, c)
            except Exception as e:
                print(product, e)

        return total_bytes

    @staticmethod
    def from_doc(doc: Dict, mongo_client: pymongo.MongoClient, s3_client) -> 'RedvoxReport':
        return RedvoxReport(doc_get(doc, "startTimestampS"),
                            doc_get(doc, "endTimestampS"),
                            list(map(lambda web_based_product: web_based_product["productLink"],
                                     doc["webBasedProducts"])),
                            doc_get(doc, "reportId"),
                            doc_get(doc, "isPublic"),
                            doc_get(doc, "isPrivate"),
                            doc_get(doc, "receivers"),
                            list(map(lambda device: f"{device['redvoxId']}:{device['redvoxUuid']}", doc["devices"])),
                            mongo_client,
                            s3_client)

    def __str__(self):
        return f"{self.start_timestamp_s} {self.end_timestamp_s} {self.report_id} {self.is_public} {self.is_private} " \
               f"{self.web_based_products} {self.receivers} {self.devices} {self.estimated_data_bytes}"


def format_mongo_auth_uri_ssl(user: str,
                              password: str,
                              host: str,
                              port: int,
                              cert: str) -> str:
    escaped_pass = urllib.parse.quote_plus(password)
    return f"mongodb://{user}:{escaped_pass}@{host}:{port}/?ssl=true&ssl_ca_certs={cert}&replicaSet=rs0"


def get_client(host: str, port: int, user: str, password: str, cert: str) -> pymongo.MongoClient:
    return pymongo.MongoClient(format_mongo_auth_uri_ssl(user, password, host, port, cert))


def get_reports(mongo_client: pymongo.MongoClient,
                s3_client) -> List[RedvoxReport]:
    db: pymongo.database.Database = mongo_client["redvox"]
    reports_coll: pymongo.collection.Collection = db["WebBasedReport"]

    query = {"devices.isApi900": True,
             "devices": {"$exists": True}}
    projection = {"_id": False,
                  "startTimestampS": True,
                  "endTimestampS": True,
                  "webBasedProducts": True,
                  "reportId": True,
                  "isPublic": True,
                  "isPrivate": True,
                  "receivers": True,
                  "devices": True}

    return list(map(lambda doc: RedvoxReport.from_doc(doc, mongo_client, s3_client), list(reports_coll.find(query, projection=projection))))


def get_stats(reports: List[RedvoxReport]) -> None:
    reports = list(filter(lambda report: report.devices is not None and len(report.devices) > 0, reports))
    incidents = list(filter(lambda report: report.is_public or (report.receivers is not None and len(report.receivers) > 0), reports))
    events = list(filter(lambda report: not report.is_public, reports))
    incident_durations = np.array(list(map(lambda report: report.end_timestamp_s - report.start_timestamp_s, incidents)))
    events_durations = np.array(list(map(lambda report: report.end_timestamp_s - report.start_timestamp_s, events)))
    incident_bytes = np.array(list(map(lambda report: report.total_bytes, incidents)))
    event_bytes = np.array(list(map(lambda report: report.total_bytes, events)))

    emin_t = min(list(map(lambda report: report.start_timestamp_s, events)))
    emax_t = max(list(map(lambda report: report.end_timestamp_s, events)))
    e_d = emax_t - emin_t
    e_dr = event_bytes.sum() / float(e_d)
    e_dr_sem = event_bytes.std() / np.sqrt(e_d)

    imin_t = min(list(map(lambda report: report.start_timestamp_s, incidents)))
    imax_t = max(list(map(lambda report: report.end_timestamp_s, incidents)))
    i_d = imax_t - imin_t
    i_dr = incident_bytes.sum() / float(i_d)
    i_dr_sem = incident_bytes.std() / np.sqrt(i_d)

    print(f"Total events: {len(events)}")
    print(f"Event durations sum: {events_durations.sum()}")
    print(f"Event durations mean: {events_durations.mean()}")
    print(f"Event durations std: {events_durations.std()}")
    print(f"Event bytes sum: {event_bytes.sum()}")
    print(f"Event bytes mean: {event_bytes.mean()}")
    print(f"Event bytes std: {event_bytes.std()}")
    print(f"Event bytes sem: {event_bytes.std() / np.sqrt(len(event_bytes))}")
    print(f"Event DR/s: {e_dr}")
    print(f"Event DR/s sem: {e_dr_sem}")

    seconds_per_week: float = 604800.0
    seconds_per_month: float = seconds_per_week * 4
    seconds_per_year: float = seconds_per_month * 12
    event_data_month: float = (event_bytes.sum() * seconds_per_month) / (emax_t - emin_t)
    print(f"Event data per month: {event_data_month}")

    print()
    print(f"Total incidents: {len(incidents)}")
    print(f"Incident durations sum: {incident_durations.sum()}")
    print(f"Incident durations mean: {incident_durations.mean()}")
    print(f"Incident durations std: {incident_durations.std()}")
    print(f"Incident bytes sum: {incident_bytes.sum()}")
    print(f"Incident bytes mean: {incident_bytes.mean()}")
    print(f"Incident bytes std: {incident_bytes.std()}")
    print(f"Incident bytes sem: {incident_bytes.std() / np.sqrt(len(incident_bytes))}")
    print(f"Incident DR/s: {i_dr}")
    print(f"Incident DR/s sem: {i_dr_sem}")

    incident_data_year: float = (incident_bytes.sum() * seconds_per_year) / (imax_t - imin_t)
    print(f"Incident data per month: {incident_data_year}")


if __name__ == "__main__":
    host = os.getenv("REDVOX_MONGO_HOST")
    port = int(os.getenv("REDVOX_MONGO_PORT"))
    user = os.getenv("REDVOX_MONGO_USER")
    passw = os.getenv("REDVOX_MONGO_PASS")
    mongo_client: pymongo.MongoClient = get_client(host, port, user, passw, "/home/ec2-user/rds-combined-ca-bundle.pem")
    s3_client = boto3.client("s3")
    reports = get_reports(mongo_client, s3_client)
    # for report in reports:
    #     print(report.estimated_data_bytes, report.product_bytes)
    get_stats(reports)



