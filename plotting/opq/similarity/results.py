def res(incident_type: str,
        cluster_id: int,
        num_incidents: int,
        mean_duration_ms: float,
        sr_hz: float,
        sample_size_bytes: int,
        mean_incident_size_bytes: int) -> None:
    size_bytes: float = num_incidents * ((mean_duration_ms / 1_000.0) * sr_hz * sample_size_bytes + mean_incident_size_bytes)
    size_gb: float = size_bytes / 1_000_000_000.0
    print(f"{incident_type} {cluster_id} & {size_gb:.2f}")

def main():
    res("FSag", 1, 116769, 98.91, 12000, 2, 365)
    res("FSag", 6, 35957, 809.18, 12000, 2, 365)
    res("FSwell", 7, 78577, 138.63, 12000, 2, 365)
    res("FSwell", 0, 95488, 158.29, 12000, 2, 365)


if __name__ == "__main__":
    main()