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

pub const ESTIMATED_EVENTS_PER_SECOND: f64 = 0.00009221688;
pub const ESTIMATED_PERCENT_DATA_DURATION: f64 = 0.0011184297762990303;
pub const ESTIMATED_PERCENT_EVENT_TO_INCIDENT: f64 = 0.4438284568272211;
pub const ESTIMATED_EVENT_LEN_S: usize = 12;

pub const ESTIMATED_BYTES_PER_MEASUREMENT: usize = 145;
pub const ESTIMATED_BYTES_PER_TREND: usize = 365;
