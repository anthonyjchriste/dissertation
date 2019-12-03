use crate::storage::StorageItem;
use rand;
use rand::prelude::ThreadRng;
use rand::Rng;

pub mod config;
pub mod constants;
pub mod storage;

#[inline]
fn percent_chance(chance: f64, rng: &mut ThreadRng) -> bool {
    rng.gen_range(0.0, 1.0) < chance
}

fn run_sim(conf: &config::Config) {
    let mut rng = rand::thread_rng();
    let mut storage = storage::Storage::new();

    let mut total_measurements: usize = 0;
    let mut total_orphaned_measurements: usize = 0;
    let mut total_event_measurements: usize = 0;
    let mut total_incident_measurements: usize = 0;
    let mut total_storage_items: usize = 0;

    let mut storage_items_per_tick: Vec<storage::StorageItem> = vec![];

    for i in 0..conf.ticks {
        storage_items_per_tick.clear();
        for _ in 0..conf.num_sensors {
            //         Probability of this measurement belonging to an event
            if percent_chance(conf.percent_event_duration, &mut rng) {
                // Measurement belongs to an event
                // Probability of this measurement also belonging to an incident
                if percent_chance(conf.percent_event_to_incident, &mut rng) {
                    let measurement = storage::StorageItem::new_measurement(
                        i,
                        i + conf.incidents_ttl,
                        None,
                        Some(true),
                    );
                    storage_items_per_tick.push(measurement);
                    total_incident_measurements += 1;
                    total_measurements += 1;
                    total_storage_items += 1;
                } else {
                    let measurement = storage::StorageItem::new_measurement(
                        i,
                        i + conf.events_ttl,
                        Some(true),
                        None,
                    );
                    storage_items_per_tick.push(measurement);
                    total_event_measurements += 1;
                    total_measurements += 1;
                    total_storage_items += 1;
                }
            } else {
                // Measurement does not belong to an event
                let measurement =
                    storage::StorageItem::new_measurement(i, i + conf.measurements_ttl, None, None);
                storage_items_per_tick.push(measurement);
                total_orphaned_measurements += 1;
                total_measurements += 1;
                total_storage_items += 1;
            }
        }

        storage.add_many(&mut storage_items_per_tick);

        //        storage.add(Measurement::new(i, i + conf.measurements_ttl));
        if i % 100_000 == 0 {
            let event_measurements = storage
                .storage_items
                .iter()
                .filter(|measurement| measurement.is_event)
                .count();
            let incident_measurements = storage
                .storage_items
                .iter()
                .filter(|measurement| measurement.is_incident)
                .count();
            let stored_measurements = storage.storage_items.len();
            let non_event_measurements =
                stored_measurements - event_measurements - incident_measurements;

            println!(
                "time={} #meas_total={} ({:.*}MB) #meas_orphan={} ({:.*}% {:.*}MB) #meas_event={} ({:.*}% {:.*}MB) #meas_incident={} ({:.*}% {:.*}MB)",
                i,
                stored_measurements,
                2, (stored_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT) as f64 / 1_000_000.0,
                non_event_measurements,
                2, non_event_measurements as f64 / stored_measurements as f64 * 100.0,
                2, (non_event_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT) as f64 / 1_000_000.0,
                event_measurements,
                2, event_measurements as f64 / stored_measurements as f64 * 100.0,
                2, (event_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT) as f64 / 1_000_000.0,
                incident_measurements,
                2, incident_measurements as f64 / stored_measurements as f64 * 100.0,
                2, (incident_measurements * constants::ESTIMATED_BYTES_PER_MEASUREMENT) as f64 / 1_000_000.0)
        }
    }

    let percent_orphaned_measurements =
        total_orphaned_measurements as f64 / total_measurements as f64 * 100.0;
    let percent_event_measurements =
        total_event_measurements as f64 / total_measurements as f64 * 100.0;
    let percent_incident_measurements =
        total_incident_measurements as f64 / total_measurements as f64 * 100.0;
    let percent_non_event_measurements =
        100.0 - percent_event_measurements - percent_incident_measurements;
    println!(
        "total_measurements={} (100%) orphaned_measurements={} ({:.*}%) event_measurements={} ({:.*}%) incident_measurements={} ({:.*}%)",
        total_measurements,
        total_orphaned_measurements,
        2, percent_orphaned_measurements,
        total_event_measurements,
        2, percent_event_measurements,
        total_incident_measurements,
        2, percent_incident_measurements
    )
}

fn main() {
    let conf = config::Config {
        ticks: constants::SECONDS_PER_YEAR * 3,
        percent_event_duration: constants::ESTIMATED_PERCENT_DATA_DURATION,
        percent_event_to_incident: constants::ESTIMATED_PERCENT_EVENT_TO_INCIDENT,
        mean_event_len: constants::ESTIMATED_EVENT_LEN_S,
        measurements_ttl: constants::DEFAULT_MEASUREMENT_TTL,
        trends_ttl: constants::DEFAULT_TRENDS_TTL,
        events_ttl: constants::DEFAULT_EVENTS_TTL,
        incidents_ttl: constants::DEFAULT_INCIDENTS_TTL,
        phenomena_ttl: constants::DEFAULT_PHENOMENA_TTL,
        num_sensors: 1,
    };
    run_sim(&conf);
}
