from typing import List

import io


def main(base_dir: str):
    incident_dirs: List[str] = io.find_incident_directories(base_dir)
    print(len(incident_dirs))


if __name__ == "__main__":
    main("/home/opq/scrap/incident_data")
