from dataclasses import dataclass
import datetime
import os
from typing import Callable, Dict, List, Tuple, TypeVar, Optional
import urllib.parse

import bson
import pymongo
import pymongo.database


def format_mongo_auth_uri_ssl(user: str,
                              password: str,
                              host: str,
                              port: int,
                              cert: str) -> str:
    escaped_pass = urllib.parse.quote_plus(password)
    return f"mongodb://{user}:{escaped_pass}@{host}:{port}/?ssl=true&ssl_ca_certs={cert}&replicaSet=rs0"


def get_client(host: str, port: int, user: str, password: str, cert: str) -> pymongo.MongoClient:
    return pymongo.MongoClient(format_mongo_auth_uri_ssl(user, password, host, port, cert))


def pl_vs_sim():
    pass


def pl_vs_est():
    pass


def actual_pl():
    pass


def main():
    host = os.getenv("REDVOX_MONGO_HOST")
    port = int(os.getenv("REDVOX_MONGO_PORT"))
    user = os.getenv("REDVOX_MONGO_USER")
    passw = os.getenv("REDVOX_MONGO_PASS")

    mongo_client: pymongo.MongoClient = get_client(host, port, user, passw, "/home/ec2-user/rds-combined-ca-bundle.pem")
    db: pymongo.database.Database = mongo_client["redvox"]
    coll: pymongo.collection.Collection = db["WebBasedReport"]
    web_based_report_docs: List[dict] = sorted(list(coll.find({"isPublic": True})), key=lambda doc: doc["startTimestampS"])

    for doc in web_based_report_docs:
        start_ts_s: int = doc["startTimestampS"]
        size_bytes: int = len(bson.BSON.encode(doc))
        print(f"{start_ts_s} {size_bytes}")


if __name__ == "__main__":
    main()
