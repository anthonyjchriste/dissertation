from typing import List, Dict

import fastkml


box_to_location: Dict[str, str] = {
    "1000": "POST 1",
    "1001": "Hamilton",
    "1002": "POST 2",
    "1003": "LAVA Lab",
    "1005": "Parking Structure Ph II",
    "1006": "Frog 1",
    "1007": "Frog 2",
    "1008": "Mile's Office",
    "1009": "Watanabe",
    "1010": "Holmes",
    "1021": "Marine Science Building",
    "1022": "Ag. Engineering",
    "1023": "Law Library",
    "1024": "IT Building",
    "1025": "Kennedy Theater"
}

def main():
    with open("doc.kml", "rb") as fin:
        kml: fastkml.kml.KML = fastkml.kml.KML()
        kml.from_string(fin.read())


        features = list(kml.features())
        document: fastkml.kml.Document = features[0]
        folder: fastkml.kml.Folder = list(document.features())[0]
        placemarks: List[fastkml.kml.Placemark] = list(folder.features())

        for placemark in placemarks:
            name: str = placemark.name
            box_id: str = name.split("-")[1].strip()
            geometry = placemark.geometry
            coords = geometry.coords[0]
            lat: float = coords[0]
            lng: float = coords[1]
            loc: str = box_to_location[box_id]
            print(f"{box_id} & {loc} & {lat} & {lng} \\\\")





if __name__ == "__main__":
    main()
