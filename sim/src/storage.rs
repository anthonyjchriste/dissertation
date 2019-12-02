use std::cmp::Ordering;

const GC_INTERVAL: usize = 600;

fn find_index<T: Ord>(v: &Vec<T>, val: &T) -> usize {
    match v.binary_search(val) {
        Ok(idx) => idx + 1,
        Err(idx) => idx,
    }
}

pub struct Measurement {
    pub ts: usize,
    pub ttl: usize,
    pub is_event: bool,
    pub is_incident: bool,
}

impl Measurement {
    pub fn new(ts: usize, ttl: usize) -> Measurement {
        Measurement {
            ts,
            ttl,
            is_event: false,
            is_incident: false,
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

pub struct Storage {
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
        //        let insertion_idx = find_index(&self.measurements, &Measurement::from_ttl(self.time));
        self.measurements.push(measurement);
        if self.time % GC_INTERVAL == 0 {
            self.gc()
        }
        self.time += 1
    }

    pub fn adjust_measurements_ttl_for_event(
        &mut self,
        start_ts: usize,
        end_ts: usize,
        ttl: usize,
    ) {
        for measurement in &mut self.measurements {
            if measurement.ts >= start_ts && measurement.ts <= end_ts {
                measurement.ttl = ttl;
                measurement.is_event = true;
            }
        }
        self.measurements.sort();
    }

    pub fn gc(&mut self) {
        self.measurements.sort();
        match self
            .measurements
            .binary_search(&Measurement::from_ttl(self.time))
        {
            Ok(idx) => self.measurements.drain(0..=idx),
            Err(idx) => self.measurements.drain(0..idx),
        };
    }
}
