import numpy as np

def percent_diff(larger: float, smaller: float) -> None:
    p_of = 100.0 - (smaller / larger * 100.0)
    print(f"{larger - smaller}/-{p_of}%")

if __name__ == "__main__":
    percent_diff(286957, 8255)
