pub const SAMPLES_PER_SECOND: usize = 12_000;
pub const BYTES_PER_SAMPLE: usize = 2;

pub const SECONDS_PER_MINUTE: usize = 60;
pub const MINUTES_PER_HOUR: usize = 60;
pub const HOURS_PER_DAY: usize = 24;
pub const DAYS_PER_WEEK: usize = 7;
pub const DAYS_PER_MONTH: usize = 30;
pub const MONTHS_PER_YEAR: usize = 12;

pub const SECONDS_PER_FIFTEEN_MINUTES: usize = SECONDS_PER_MINUTE * 15;
pub const SECONDS_PER_HOUR: usize = SECONDS_PER_MINUTE * MINUTES_PER_HOUR;
pub const SECONDS_PER_DAY: usize = SECONDS_PER_HOUR * HOURS_PER_DAY;
pub const SECONDS_PER_WEEK: usize = SECONDS_PER_DAY * DAYS_PER_WEEK;
pub const SECONDS_PER_TWO_WEEKS: usize = SECONDS_PER_WEEK * 2;
pub const SECONDS_PER_MONTH: usize = SECONDS_PER_DAY * DAYS_PER_MONTH;
pub const SECONDS_PER_YEAR: usize = SECONDS_PER_MONTH * MONTHS_PER_YEAR;

pub const DEFAULT_MEASUREMENT_TTL: usize = SECONDS_PER_DAY;
pub const DEFAULT_TRENDS_TTL: usize = SECONDS_PER_TWO_WEEKS;
pub const DEFAULT_EVENTS_TTL: usize = SECONDS_PER_MONTH;
pub const DEFAULT_INCIDENTS_TTL: usize = SECONDS_PER_YEAR;
pub const DEFAULT_PHENOMENA_TTL: usize = SECONDS_PER_YEAR * 2;

pub const ESTIMATED_EVENTS_PER_SECOND: f64 = 0.00013571616456523863;
pub const ESTIMATED_INCIDENTS_PER_SECOND: f64 = 0.018394470933077035;
pub const ESTIMATED_PERCENT_DATA_DURATION: f64 = 0.0018576987367277473;
pub const ESTIMATED_PERCENT_EVENT_TO_INCIDENT: f64 = 0.32236036457116146;
pub const ESTIMATED_EVENT_LEN_S: usize = 14;
pub const ESTIMATED_INCIDENT_LEN_S: f64 = 0.5317527500704905;

pub const ESTIMATED_BYTES_PER_META_SAMPLE: usize = SAMPLES_PER_SECOND * BYTES_PER_SAMPLE;
pub const ESTIMATED_BYTES_PER_MEASUREMENT: usize = 145;
pub const ESTIMATED_BYTES_PER_TREND: usize = 365;
pub const ESTIMATED_BYTES_PER_EVENT: usize =
    ESTIMATED_EVENT_LEN_S * SAMPLES_PER_SECOND * BYTES_PER_SAMPLE;
pub const ESTIMATED_BYTES_PER_INCIDENT: usize =
    (ESTIMATED_INCIDENT_LEN_S * SAMPLES_PER_SECOND as f64 * BYTES_PER_SAMPLE as f64) as usize;

pub const DEFAULT_GC_INTERVAL: usize = 600;
