from typing import List, Tuple

import lxml
import fastkml
import shapely
import shapely.geometry


def main():
    box_locations: List[Tuple[float, float]] = [
        (-157.816237, 21.297438),
        (-157.816173, 21.300332),
        (-157.816305, 21.297663),
        (-157.816034748669, 21.29974948387028),
        (-157.819234, 21.296042),
        (-157.823122, 21.29805),
        (-157.822819, 21.298046),
        (-157.8137681609306, 21.30386147625208),
        (-157.815817, 21.298351),
        (-157.816104, 21.297011),
        (-157.8156900462205, 21.29789461271471),
        (-157.8154874938278, 21.30163608338939),
        (-157.817361, 21.296328),
        (-157.816451, 21.29886),
        (-157.815225, 21.299282)
    ]

    kml: fastkml.kml.KML = fastkml.kml.KML()
    ns: str = '{http://www.opengis.net/kml/2.2}'

    document: fastkml.kml.Document = fastkml.kml.Document(ns)
    kml.append(document)

    folder: fastkml.kml.Folder = fastkml.kml.Folder(ns)

    document.append(folder)

    for lat_lng in box_locations:
        placemark: fastkml.kml.Placemark = fastkml.kml.Placemark(ns)
        placemark.geometry = shapely.geometry.Point(lat_lng[0], lat_lng[1])
        # placemark.name = historical_device.id_uuid
        folder.append(placemark)

    with open("opq_devices.kml", "w") as fout:
        fout.write(kml.to_string(prettyprint=True))


if __name__ == "__main__":
    main()
