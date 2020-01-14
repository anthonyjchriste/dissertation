from typing import List
import os


def find_incident_directories(base_dir: str) -> List[str]:
    return [f.path for f in os.scandir(base_dir) if f.is_dir()]


