import collections
import datetime
import os
import os.path
from typing import Dict, Set
import urllib.parse

import pymongo
import pymongo.database


class DailyMetrics:
    def __init__(self,
                 devices_80hz: Set[str],
                 devices_800hz: Set[str],
                 devices_8000hz: Set[str],
                 total_80hz_packets: int,
                 total_800hz_packets: int,
                 total_8000hz_packets: int):

        self.devices_80hz: Set[str] = devices_80hz
        self.devices_800hz: Set[str] = devices_800hz
        self.devices_8000hz: Set[str] = devices_8000hz
        self.total_80hz_packets: int = total_80hz_packets
        self.total_800hz_packets: int = total_800hz_packets
        self.total_8000hz_packets: int = total_8000hz_packets

    @staticmethod
    def new() -> 'DailyMetrics':
        return DailyMetrics(set(), set(), set(), 0, 0, 0)

    def total_devices_80hz(self) -> int:
        return len(self.devices_80hz)

    def total_devices_800hz(self) -> int:
        return len(self.devices_800hz)

    def total_devices_8000hz(self) -> int:
        return len(self.devices_8000hz)

    def total_devices(self) -> int:
        return self.total_devices_80hz() + self.total_devices_800hz() + self.total_devices_8000hz()

    def total_packets(self) -> int:
        return self.total_80hz_packets + self.total_800hz_packets + self.total_8000hz_packets

    def total_data_bytes_80hz(self) -> float:
        return self.total_80hz_packets * 4096 * 4

    def total_data_bytes_800hz(self) -> float:
        return self.total_800hz_packets * 32768 * 4

    def total_data_bytes_8000hz(self) -> float:
        return self.total_8000hz_packets * 262144 * 4

    def total_data_bytes(self) -> float:
        return self.total_data_bytes_80hz() + self.total_devices_800hz() + self.total_devices_8000hz()


def format_mongo_auth_uri_ssl(user: str,
                              password: str,
                              host: str,
                              port: int,
                              cert: str) -> str:
    escaped_pass = urllib.parse.quote_plus(password)
    return f"mongodb://{user}:{escaped_pass}@{host}:{port}/?ssl=true&ssl_ca_certs={cert}&replicaSet=rs0"


def get_client(host: str, port: int, user: str, password: str, cert: str) -> pymongo.MongoClient:
    return pymongo.MongoClient(format_mongo_auth_uri_ssl(user, password, host, port, cert))


def bin_timestamp_us(ts_us: int) -> str:
    dt: datetime.datetime = datetime.datetime.utcfromtimestamp(ts_us / 1000.0 / 1000.0)
    return dt.strftime("%Y-%m-%d")


def scrape_api900_packets(mongo_client: pymongo.MongoClient,
                          out_dir: str):
    db: pymongo.database.Database = mongo_client["redvox"]
    coll: pymongo.collection.Collection = db["RedvoxPacketApi900"]

    query: Dict = {
        "appFileStartTimestampMachine": {"$gte": 1569888000000000,
                                         "$lte": 1576368000000000}
    }

    projection: Dict[str, bool] = {
        "_id": False,
        "redvoxId": True,
        "redvoxUuid": True,
        "appFileStartTimestampMachine": True,
        "evenlySampledChannels.sampleRateHz": True
    }

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection)

    daily_data: Dict[str, DailyMetrics] = collections.defaultdict(lambda: DailyMetrics.new())

    already_seen_binned_ts: Set[str] = set()

    for doc in cursor:
        app_file_start_timestamp_machine: int = doc["appFileStartTimestampMachine"]
        redvox_id: str = doc["redvoxId"]
        redvox_uuid: str = doc["redvoxUuid"]
        redvox_id_uuid: str = f"{redvox_id}:{redvox_uuid}"
        binned_ts: str = bin_timestamp_us(app_file_start_timestamp_machine)

        sample_rate_hz: int = doc["evenlySampledChannels"][0]["sampleRateHz"]

        daily_metrics: DailyMetrics = daily_data[binned_ts]

        if sample_rate_hz == 80:
            daily_metrics.devices_80hz.add(redvox_id_uuid)
            daily_metrics.total_80hz_packets += 1
        elif sample_rate_hz == 800:
            daily_metrics.devices_800hz.add(redvox_id_uuid)
            daily_metrics.total_800hz_packets += 1
        elif sample_rate_hz == 8000:
            daily_metrics.devices_8000hz.add(redvox_id_uuid)
            daily_metrics.total_8000hz_packets += 1
        else:
            print(f"Unknown sample rate hz: {sample_rate_hz}")

        if binned_ts not in already_seen_binned_ts:
            already_seen_binned_ts.add(binned_ts)
            print(binned_ts)

    out_file: str = "metrics.txt"
    out_path: str = os.path.join(out_dir, out_file)
    with open(out_path, "w") as fout:
        for binned_ts, daily_metrics in daily_data.items():
            line: str = f"{binned_ts} " \
                        f"{daily_metrics.total_80hz_packets} " \
                        f"{daily_metrics.total_800hz_packets} " \
                        f"{daily_metrics.total_8000hz_packets} " \
                        f"{daily_metrics.total_packets()} " \
                        f"{daily_metrics.total_data_bytes_80hz()} " \
                        f"{daily_metrics.total_data_bytes_800hz()} " \
                        f"{daily_metrics.total_data_bytes_8000hz()} " \
                        f"{daily_metrics.total_data_bytes()} " \
                        f"{daily_metrics.total_devices_80hz()} " \
                        f"{daily_metrics.total_devices_800hz()} " \
                        f"{daily_metrics.total_devices_8000hz()} " \
                        f"{daily_metrics.total_devices()}\n"

            fout.write(line)





if __name__ == "__main__":
    host: str = os.getenv("REDVOX_MONGO_HOST")
    port: int = int(os.getenv("REDVOX_MONGO_PORT"))
    user: str = os.getenv("REDVOX_MONGO_USER")
    passw: str = os.getenv("REDVOX_MONGO_PASS")
    mongo_client: pymongo.MongoClient = get_client(host, port, user, passw, "/home/ec2-user/rds-combined-ca-bundle.pem")

    scrape_api900_packets(mongo_client, "/home/ec2-user/packets")
