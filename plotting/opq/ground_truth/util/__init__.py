from typing import Dict, List
import os.path

opq_box_to_uhm_meters: Dict[str, List[str]] = {
    "1000": ["POST_MAIN_1",
             "POST_MAIN_2"],
    "1001": ["HAMILTON_LIB_PH_III_CH_1_MTR",
             "HAMILTON_LIB_PH_III_CH_2_MTR",
             "HAMILTON_LIB_PH_III_CH_3_MTR",
             "HAMILTON_LIB_PH_III_MAIN_1_MTR",
             "HAMILTON_LIB_PH_III_MAIN_2_MTR",
             "HAMILTON_LIB_PH_III_MCC_AC1_MTR",
             "HAMILTON_LIB_PH_III_MCC_AC2_MTR"],
    "1002": ["POST_MAIN_1",
             "POST_MAIN_2"],
    "1003": ["KELLER_HALL_MAIN_MTR"],
    "1005": [],
    "1006": [],
    "1007": [],
    "1008": [],
    "1009": [],
    "1010": [],
    "1021": ["MARINE_SCIENCE_MAIN_A_MTR",
             "MARINE_SCIENCE_MAIN_B_MTR",
             "MARINE_SCIENCE_MCC_MTR"],
    "1022": ["AG_ENGINEERING_MAIN_MTR",
             "AG_ENGINEERING_MCC_MTR"],
    "1023": ["LAW_LIB_MAIN_MTR"],
    "1024": [],
    "1025": ["KENNEDY_THEATRE_MAIN_MTR"]
}

uhm_meter_to_opq_box: Dict[str, str] = {}
for opq_box, uhm_meters in opq_box_to_uhm_meters.items():
    for uhm_meter in uhm_meters:
        uhm_meter_to_opq_box[uhm_meter] = opq_box


def latex_figure_source(path: str) -> str:
    file_name: str = os.path.split(path)[-1]
    file_label: str = file_name.split(".")[0]
    return """
Figure~\\ref{fig:FILE_LABEL}    
    
\\begin{figure}[H]
    \\centering
    \\includegraphics[width=\\linewidth]{figures/FILE_NAME}
    \\caption{}
    \\label{fig:FILE_LABEL}
\\end{figure}
""".replace("FILE_NAME", file_name).replace("FILE_LABEL", file_label)

def latex_sub_float(path: str, width: float) -> str:
    file_name: str = os.path.split(path)[-1]
    file_label: str = file_name.split(".")[0].replace("_", "\\_")

    return "\\subfloat[FILE_LABEL]{\\includegraphics[width = WIDTH\\linewidth]{figures/FILE_NAME}}" \
        .replace("FILE_NAME", file_name) \
        .replace("WIDTH", str(width)) \
        .replace("FILE_LABEL", file_label)


def latex_figure_table_source(paths: List[str], max_rows: int, max_cols: int) -> None:
    sub_float_width = 1.0 / max_cols
    sub_floats: List[str] = list(map(lambda path: latex_sub_float(path, sub_float_width), paths))

    offset: int = max_rows * max_cols
    while len(sub_floats) > 0:
        figure_floats: List[str] = sub_floats[:offset]
        sub_floats = sub_floats[offset:]

        figure_floats_str: str = ""
        for i, figure_float in enumerate(figure_floats):
            if i == len(figure_floats) - 1:
                figure_floats_str += f"\t\t{figure_floats[i]} \\\\"
            elif i % 2 == 1:
                figure_floats_str += f"\t\t{figure_floats[i]} \\\\ \n"
            else:
                figure_floats_str += f"\t\t{figure_floats[i]} & \n"

        columns = "c" * max_cols

        figure_src: str = """
\\begin{figure}
    \\begin{tabular}{COLUMNS}
SUB_FLOATS
    \\end{tabular}
\\end{figure}
        """.replace("COLUMNS", columns).replace("SUB_FLOATS", figure_floats_str)

        print(figure_src)



if __name__ == "__main__":
    print(latex_figure_source("/home/test/foo.png"))
