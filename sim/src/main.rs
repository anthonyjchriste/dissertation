use rand;
use rand::Rng;
use std::cmp::Ordering;
use test::bench::iter;

const SECONDS_PER_MINUTE: usize = 60;
const SECONDS_PER_FIFTEEN_MINUTES: usize = SECONDS_PER_MINUTE * 15;
const SECONDS_PER_HOUR: usize = SECONDS_PER_MINUTE * 60;
const SECONDS_PER_DAY: usize = SECONDS_PER_HOUR * 24;
const SECONDS_PER_WEEK: usize = SECONDS_PER_DAY * 7;
const SECONDS_PER_TWO_WEEKS: usize = SECONDS_PER_WEEK * 2;
const SECONDS_PER_MONTH: usize = SECONDS_PER_DAY * 30;
const SECONDS_PER_YEAR: usize = SECONDS_PER_MONTH * 12;

const MEASUREMENT_TTL: usize = SECONDS_PER_DAY;
const TRENDS_TTL: usize = SECONDS_PER_TWO_WEEKS;
const EVENTS_TTL: usize = SECONDS_PER_MONTH;
const INCIDENTS_TTL: usize = SECONDS_PER_YEAR;

struct Measurement {
    pub ts: usize,
    pub ttl: usize,
    pub is_event: bool,
}

impl Measurement {
    pub fn new(ts: usize, ttl: usize) -> Measurement {
        Measurement {
            ts,
            ttl,
            is_event: false,
        }
    }

    pub fn from_ttl(ttl: usize) -> Measurement {
        Measurement::new(0, ttl)
    }
}

impl PartialOrd for Measurement {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.ttl.partial_cmp(&other.ttl)
    }
}

impl PartialEq for Measurement {
    fn eq(&self, other: &Self) -> bool {
        self.ttl == other.ttl
    }
}

impl Ord for Measurement {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ttl.cmp(&other.ttl)
    }
}

impl Eq for Measurement {}

struct Storage {
    pub measurements: Vec<Measurement>,
    pub time: usize,
}

impl Storage {
    pub fn new() -> Storage {
        Storage {
            measurements: vec![],
            time: 0,
        }
    }

    pub fn add(&mut self, measurement: Measurement) {
        self.measurements.push(measurement);
        if self.time % 60 == 0 {
            self.gc()
        }
        self.time += 1
    }

    pub fn adjust_measurements_ttl_for_event(&mut self, prev_measurements: usize, ttl: usize) {
        for i in 0..prev_measurements {
            let last_idx = self.measurements.len() - 1;
            self.measurements[last_idx - i].ttl = ttl;
            self.measurements[last_idx - i].is_event = true;
        }
        self.measurements.sort();
    }

    pub fn gc(&mut self) {
        //        match self
        //            .measurements
        //            .binary_search(&Measurement::from_ttl(self.time))
        //        {
        //            Ok(idx) => self.measurements.drain(0..=idx),
        //            Err(idx) => self.measurements.drain(0..idx),
        //        };
        self.measurements = self
            .measurements
            .iter()
            .to_owned()
            .filter(|measurement| self.time > measurement.ttl)
            .collect();
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let mut storage = Storage::new();

    for i in 0..SECONDS_PER_YEAR {
        let measurement = Measurement::new(i, i + MEASUREMENT_TTL);
        storage.add(measurement);

        if rng.gen_range(0.0, 1.0) < 0.00009221688 {
            storage.adjust_measurements_ttl_for_event(12, i + EVENTS_TTL)
        }

        if i % 100_000 == 0 {
            let event_measurements = storage
                .measurements
                .iter()
                .filter(|measurement| measurement.is_event)
                .count();
            let non_event_measurements = storage.measurements.len() - event_measurements;
            println!(
                "time={} #meas={} #meas_no_event={} #meas_event={} %meas_event={}",
                i,
                storage.measurements.len(),
                non_event_measurements,
                event_measurements,
                event_measurements as f64 / non_event_measurements as f64 * 100.0
            )
        }
    }
}
