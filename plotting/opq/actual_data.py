from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import pymongo
import pymongo.database


DB: str = "opq"
COLL: str = "laha_stats"


@dataclass
class PluginStat:
    name: str
    messages_received: int
    messages_published: int
    bytes_received: int
    bytes_published: int

    @staticmethod
    def from_doc(name: str, doc: Dict[str, int]) -> 'PluginStat':
        return PluginStat(
                name,
                doc["messages_received"],
                doc["messages_published"],
                doc["bytes_received"],
                doc["bytes_published"]
        )


@dataclass
class SystemStats:
    cpu_load_percent: float
    memory_use_bytes: float
    disk_use_bytes: float

    @staticmethod
    def from_doc(doc: Dict[str, Union[float, int]]) -> 'SystemStats':
        return SystemStats(
                doc["cpu_load_percent"],
                doc["memory_use_bytes"],
                doc["disk_use_bytes"]
        )



@dataclass
class LahaMetric:
    name: str
    ttl: int
    count: int
    size_bytes: int

    @staticmethod
    def from_doc(name: str, doc: Dict[str, int]) -> 'LahaMetric':
        return LahaMetric(
                name,
                doc["ttl"],
                doc["count"],
                doc["size_bytes"]
        )


@dataclass
class LahaStat:
    timestamp_s: int
    plugin_stats: List[PluginStat]
    system_stats: SystemStats
    laha_stats: List[LahaMetric]

    @staticmethod
    def from_dict(doc: Dict) -> 'LahaStat':
        plugin_stats: List[PluginStat] = []

        dict_plugin_stats: Dict[str, Dict[str, int]] = doc["plugin_stats"]

        for plugin_name, plugin_dict in dict_plugin_stats.items():
            plugin_stats.append(PluginStat.from_doc(plugin_name, plugin_dict))


        dict_system_stats: Dict[str, Union[float, int]] = doc["system_stats"]
        system_stats: SystemStats = SystemStats.from_doc(dict_system_stats)

        laha_stats: List[LahaMetric] = []
        dict_laha_stats: Dict[str, Dict[str, Dict[str, int]]] = doc["laha_stats"]

        for laha_name, data_dict in dict_laha_stats.items():
            for data_name, data_dict_inner in data_dict.items():
                laha_stats.append(LahaMetric.from_doc(data_name, data_dict_inner))

        return LahaStat(doc["timestamp_s"],
                        plugin_stats,
                        system_stats,
                        laha_stats)


def get_laha_stats(mongo_client: pymongo.MongoClient) -> List[LahaStat]:
    db: pymongo.database.Database = mongo_client[DB]
    coll: pymongo.collection.Collection = db[COLL]

    query = {}
    projection = {"_id": False}

    cursor: pymongo.cursor.Cursor = coll.find(query, projection=projection).limit(1000)
    docs: List[Dict] = list(cursor)

    return list(map(LahaStat.from_dict, docs))


if __name__ == "__main__":
    mongo_client: pymongo.MongoClient = pymongo.MongoClient()

    laha_stats: List[LahaStat] = get_laha_stats(mongo_client)

    for laha_stat in laha_stats:
        print(laha_stat)
