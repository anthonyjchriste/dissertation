# from dataclasses import dataclass
import datetime
import os
from typing import Callable, Dict, List, Tuple, TypeVar, Optional
import urllib.parse

import bson
import pymongo
import pymongo.database

data = """
1494890400 4730
1496523900 3932
1497209340 5356
1498244940 2560
1498245420 2248
1498422300 3402
1498469400 8343
1499296980 2506
1504409400 1613
1504410960 1872
1504806240 1163
1511973000 7304
1513352100 2816
1515373200 3317
1517433900 4110
1517949300 8940
1517949900 6079
1519941720 3806
1520314380 2849
1523370600 1198
1523374200 1184
1523374800 1478
1523377800 1184
1523394480 3467
1524855840 2550
1525379396 2473
1525469504 13568
1525473116 21085
1525473116 13721
1526495400 1941
1526529600 2324
1526565600 2236
1535043600 9616
1535068800 5625
1539749700 1976
1542445200 2678
1552474442 23229
1555022040 7110
1555211042 7384
1556054040 5431
1556313360 2431
1558656000 11211
1558665000 18501
1561444200 8554
1561444200 3627
1562065200 8183
1562065200 5485
1562788800 7330
1564049460 13799
1565259180 5621
1565620800 3439
1570237785 4842
1573484100 2482
1578363540 4576
"""

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
