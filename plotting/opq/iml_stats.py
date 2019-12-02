from functools import total_ordering
from typing import List
import bisect
import random

SECONDS_IN_YEAR = 86400 * 28


@total_ordering
class Measurement:
    def __init__(self, ts: int, ttl: int, is_event: bool = False):
        self.ts = ts
        self.ttl = ttl
        self.is_event = is_event

    def __eq__(self, other):
        return self.ttl == other.ttl

    def __ne__(self, other):
        return self.ttl != other.ttl

    def __lt__(self, other):
        return self.ttl < other.ttl

    def __repr__(self):
        return f"{self.ts} {self.ttl}"


class Storage:
    def __init__(self):
        self.measurements = []
        self.time = 0

    def add(self, measurement: Measurement):
        self.measurements.append(measurement)
        if self.time % 60 == 0:
            self.gc()
        self.time += 1

    def gc(self):
        from_idx = bisect.bisect_right(self.measurements, Measurement(0, self.time))
        self.measurements = self.measurements[from_idx:]

    def adjust_measurements_ttl(self, prev_measurements: int, ttl: int):
        for measurement in self.measurements[-prev_measurements:]:
            measurement.ttl = ttl
            measurement.is_event = True
        self.measurements.sort()


# def measurements_generator():
#     time = 1
#     ttl = 86400
#     def generate_measurement() -> Measurement:
#         return Measurement(time, time + ttl)

if __name__ == "__main__":
    storage = Storage()
    measurement_ttl = 86400
    event_ttl = measurement_ttl * 7 * 2
    percent_measurements_to_event: float = .11
    mean_event_duration_s = 12
    percent_measurements_to_incident: float = .11
    mean_incident_duration_s = 1

    is_event = False
    event_measurements = 0
    for i in range(SECONDS_IN_YEAR):
        measurement = Measurement(i, i + measurement_ttl)
        storage.add(measurement)

        if random.random() < 0.00009221688:
            storage.adjust_measurements_ttl(mean_event_duration_s, i + event_ttl)
            event_measurements += mean_event_duration_s

        if i % 100000 == 0:
            events_cnt = len(list(filter(lambda measurement: measurement.is_event, storage.measurements)))
            measurements_cnt = len(storage.measurements) - events_cnt
            print(i, len(storage.measurements), measurements_cnt, events_cnt, events_cnt / float(measurements_cnt) * 100.0)

    print(event_measurements / SECONDS_IN_YEAR)
    # ms = [
    #     Measurement(1, 5),
    #     Measurement(2, 6),
    #     Measurement(3, 7),
    #     Measurement(4, 8),
    #     Measurement(5, 9),
    # ]
    #
    # i = bisect.bisect_right(ms, Measurement(2, 0))
    # ttl = ms[i:]
    # print(i, ttl)
