import datetime
import os
from dataclasses import dataclass
from typing import Dict, List
import urllib.parse

import fastkml
import pymongo
import pymongo.database
import shapely
import shapely.geometry


@dataclass
class HistoricalDevice:
    last_active_ts: int
    id_uuid: str
    lat: float
    lng: float

    @staticmethod
    def from_line(line: str) -> 'HistoricalDevice':
        split_line = line.strip().split(" ")
        last_active_ts: int = int(split_line[0])
        id_uuid: str = split_line[1]
        lat: float = float(split_line[2])
        lng: float = float(split_line[3])

        return HistoricalDevice(last_active_ts, id_uuid, lat, lng)


def parse_historical_devices(path: str) -> List[HistoricalDevice]:
    with open(path, "r") as fin:
        lines: List[str] = fin.readlines()
        historical_devices: List[HistoricalDevice] = list(map(HistoricalDevice.from_line, lines))
        return sorted(historical_devices, key=lambda historical_device: historical_device.last_active_ts)


def format_mongo_auth_uri_ssl(user: str,
                              password: str,
                              host: str,
                              port: int,
                              cert: str) -> str:
    escaped_pass = urllib.parse.quote_plus(password)
    return f"mongodb://{user}:{escaped_pass}@{host}:{port}/?ssl=true&ssl_ca_certs={cert}&replicaSet=rs0"


def get_client(host: str, port: int, user: str, password: str, cert: str) -> pymongo.MongoClient:
    return pymongo.MongoClient(format_mongo_auth_uri_ssl(user, password, host, port, cert))


def get_historical_devices(mongo_client: pymongo.MongoClient) -> None:
    db: pymongo.database.Database = mongo_client["redvox"]
    coll: pymongo.collection.Collection = db["HistoricalDevice"]
    dt_start_window: datetime.datetime = datetime.datetime(2019, 10, 1, 0, 0, 0, 0)

    query: Dict = {
        "lastActive": {"$gte": dt_start_window}
    }

    projection: Dict = {
        "_id": False,
        "deviceId": True,
        "uuid": True,
        "lastActive": True,
        "lastLocation.coordinates": True
    }

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection)

    with open("historical_devices.txt", "w") as fout:
        for doc in cursor:
            device_id: str = str(doc["deviceId"])
            uuid: str = str(doc["uuid"])
            id_uuid: str = f"{device_id}:{uuid}"
            last_update: datetime.datetime = doc["lastActive"]
            last_update_ts: int = round(last_update.timestamp())
            coords: List[float] = doc["lastLocation"]["coordinates"]
            lat: float = coords[0]
            lng: float = coords[1]
            line: str = f"{last_update_ts} {id_uuid} {lat} {lng}\n"

            fout.write(line)


def make_kml(historical_devices: List[HistoricalDevice]) -> None:
    kml: fastkml.kml.KML = fastkml.kml.KML()
    ns: str = '{http://www.opengis.net/kml/2.2}'

    document: fastkml.kml.Document = fastkml.kml.Document(ns)
    kml.append(document)

    folder: fastkml.kml.Folder = fastkml.kml.Folder(ns)

    document.append(folder)

    for historical_device in historical_devices:
        placemark: fastkml.kml.Placemark = fastkml.kml.Placemark(ns)
        placemark.geometry = shapely.geometry.Point(historical_device.lat, historical_device.lng)
        # placemark.name = historical_device.id_uuid
        folder.append(placemark)

    with open("lokahi_historical_devices.kml", "w") as fout:
        fout.write(kml.to_string(prettyprint=True))



def main():
    # host = os.getenv("REDVOX_MONGO_HOST")
    # port = int(os.getenv("REDVOX_MONGO_PORT"))
    # user = os.getenv("REDVOX_MONGO_USER")
    # passw = os.getenv("REDVOX_MONGO_PASS")
    # mongo_client: pymongo.MongoClient = get_client(host, port, user, passw,
    # "/home/ec2-user/rds-combined-ca-bundle.pem")
    # parse_historical_devices(mongo_client)
    historical_devices: List[HistoricalDevice] = parse_historical_devices("historical_devices.txt")
    make_kml(historical_devices)


if __name__ == "__main__":
    main()
