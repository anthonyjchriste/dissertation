use std::cmp::Ordering;

const SECONDS_PER_MINUTE: usize = 60;
const SECONDS_PER_FIFTEEN_MINUTES: usize = SECONDS_PER_MINUTE * 15;
const SECONDS_PER_HOUR: usize = SECONDS_PER_MINUTE * 60;
const SECONDS_PER_DAY: usize = SECONDS_PER_HOUR * 24;
const SECONDS_PER_WEEK: usize = SECONDS_PER_DAY * 7;
const SECONDS_PER_TWO_WEEKS: usize = SECONDS_PER_WEEK * 2;
const SECONDS_PER_MONTH: usize = SECONDS_PER_DAY * 30;
const SECONDS_PER_YEAR: usize = SECONDS_PER_MONTH * 12;

struct Measurement {
    pub ts: usize,
    pub ttl: usize,
    pub is_event: bool
}

impl Measurement {
    pub fn new(ts: usize, ttl: usize) -> Measurement {
        Measurement {
            ts,
            ttl,
            is_event: false
        }
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

impl Eq for Measurement { }

struct Storage {
    pub measurements: Vec<Measurement>,
    pub time: uszie
}

impl Storage {
    pub fn new() -> Storage {
        
    }
}

fn main() {
    println!("Hello, world!");
}
