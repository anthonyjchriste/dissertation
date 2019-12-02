use crate::storage::Measurement;
use rand;
use rand::Rng;

pub mod storage;

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

const MEAN_EVENTS_PER_SECOND: f64 = 0.00009221688;
const MEAN_EVENT_LEN_S: usize = 12;

const BYTES_PER_MEASUREMENT: usize = 145;
const BYTES_PER_TREND: usize = 365;

fn main() {
    let mut rng = rand::thread_rng();
    let mut storage = storage::Storage::new();
    let mut total_event_measurements: usize = 0;

    for i in 0..SECONDS_PER_YEAR {
        //        let measurement = storage::Measurement::new(i, i + MEASUREMENT_TTL);
        //        storage.add(measurement);
        //
        //        if rng.gen_range(0.0, 1.0) < MEAN_EVENTS_PER_SECOND {
        //            storage.adjust_measurements_ttl_for_event(i - MEAN_EVENT_LEN_S, i, i + EVENTS_TTL);
        //            total_event_measurements += MEAN_EVENT_LEN_S;
        //        }

        // Probability of this measurement belonging to an event
        if rng.gen_range(0.0, 1.0) < 0.0011 {
            // Measurement belongs to an event
            // Probability of this measurement also belonging to an incident
            let mut measurement = Measurement::new(i, i + EVENTS_TTL);
            measurement.is_event = true;
            storage.add(measurement);
            total_event_measurements += 1;
        } else {
            // Measurement does not belong to an event
            storage.add(Measurement::new(i, i + MEASUREMENT_TTL))
        }

        if i % 100_000 == 0 {
            let event_measurements = storage
                .measurements
                .iter()
                .filter(|measurement| measurement.is_event)
                .count();
            let stored_measurements = storage.measurements.len();
            let non_event_measurements = stored_measurements - event_measurements;

            println!(
                "time={} #meas={}({:.*} MB) #meas_no_event={}({:.*} MB) #meas_event={}({:.*} MB) %meas_event={}",
                i,
                stored_measurements,
                2,
                (stored_measurements * BYTES_PER_MEASUREMENT) as f64 / 1_000_000.0,
                non_event_measurements,
                2,
                (non_event_measurements * BYTES_PER_MEASUREMENT) as f64 / 1_000_000.0,
                event_measurements,
                2,
                (event_measurements * BYTES_PER_MEASUREMENT) as f64 / 1_000_000.0,
                event_measurements as f64 / non_event_measurements as f64 * 100.0,
            )
        }
    }

    let percent_event_measurements =
        total_event_measurements as f64 / SECONDS_PER_YEAR as f64 * 100.0;
    let percent_non_event_measurements = 100.0 - percent_event_measurements;
    println!(
        "event_measurements={}({:.*}%) total_measurements={}({:.*}%)",
        total_event_measurements,
        2,
        percent_event_measurements,
        SECONDS_PER_YEAR,
        2,
        percent_non_event_measurements
    )
}
