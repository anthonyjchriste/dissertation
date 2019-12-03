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
    let mut total_trends: usize = 0;
    let mut total_orphaned_trends: usize = 0;
    let mut total_event_trends: usize = 0;
    let mut total_incident_trends: usize = 0;
    let mut total_storage_items: usize = 0;

    let mut storage_items_per_tick: Vec<storage::StorageItem> = vec![];

    for i in 0..conf.ticks {
        storage_items_per_tick.clear();
        for _ in 0..conf.num_sensors {
            //         Probability of this measurement belonging to an event
            let make_trend = i % 60 == 0;
            //            let make_trend = false;
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
                    if make_trend {
                        let trend = storage::StorageItem::new_trend(
                            i,
                            i + conf.incidents_ttl,
                            None,
                            Some(true),
                        );
                        storage_items_per_tick.push(trend);
                        total_incident_trends += 1;
                        total_trends += 1;
                        total_storage_items += 1;
                    }
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
                    if make_trend {
                        let trend = storage::StorageItem::new_trend(
                            i,
                            i + conf.events_ttl,
                            Some(true),
                            None,
                        );
                        storage_items_per_tick.push(trend);
                        total_event_trends += 1;
                        total_trends += 1;
                        total_storage_items += 1;
                    }
                }
            } else {
                // Measurement does not belong to an event
                let measurement =
                    storage::StorageItem::new_measurement(i, i + conf.measurements_ttl, None, None);
                storage_items_per_tick.push(measurement);
                total_orphaned_measurements += 1;
                total_measurements += 1;
                total_storage_items += 1;
                if make_trend {
                    let trend = storage::StorageItem::new_trend(i, i + conf.trends_ttl, None, None);
                    storage_items_per_tick.push(trend);
                    total_orphaned_trends += 1;
                    total_trends += 1;
                    total_storage_items += 1;
                }
            }
        }

        storage.add_many(&mut storage_items_per_tick, i);

        if i % 100_000 == 0 {
            let measurement_stats = storage.stat_storage_items(
                Some(storage::StorageType::Measurement(145)),
                None,
                None,
            );
            let measurement_orphaned_stats = storage.stat_storage_items(
                Some(storage::StorageType::Measurement(145)),
                Some(false),
                Some(false),
            );
            let measurement_event_stats = storage.stat_storage_items(
                Some(storage::StorageType::Measurement(145)),
                Some(true),
                Some(false),
            );
            let measurement_incident_stats = storage.stat_storage_items(
                Some(storage::StorageType::Measurement(145)),
                Some(false),
                Some(true),
            );
            let trends_stats =
                storage.stat_storage_items(Some(storage::StorageType::Trend(365)), None, None);
            let trends_orphaned_stats = storage.stat_storage_items(
                Some(storage::StorageType::Trend(365)),
                Some(false),
                Some(false),
            );
            let trends_event_stats = storage.stat_storage_items(
                Some(storage::StorageType::Trend(365)),
                Some(true),
                Some(false),
            );
            let trends_incident_stats = storage.stat_storage_items(
                Some(storage::StorageType::Trend(365)),
                Some(false),
                Some(true),
            );
            println!(
                "time={}/{} ({:.*}%) m_t={} ({}) m_o={} ({} {}) m_e={} ({} {}) m_i={} ({} {}) t_t={} ({}) t_o={} ({} {}) t_e={} ({} {}) t_i={} ({} {})",
                i,
                conf.ticks,
                2, i as f64 / conf.ticks as f64 * 100.0,
                measurement_stats.items,
                measurement_stats.fmt_size_mb(),
                measurement_orphaned_stats.items,
                measurement_orphaned_stats.fmt_percent(),
                measurement_orphaned_stats.fmt_size_mb(),
                measurement_event_stats.items,
                measurement_event_stats.fmt_percent(),
                measurement_event_stats.fmt_size_mb(),
                measurement_incident_stats.items,
                measurement_incident_stats.fmt_percent(),
                measurement_incident_stats.fmt_size_mb(),
                trends_stats.items,
                trends_stats.fmt_size_mb(),
                trends_orphaned_stats.items,
                trends_orphaned_stats.fmt_percent(),
                trends_orphaned_stats.fmt_size_mb(),
                trends_event_stats.items,
                trends_event_stats.fmt_percent(),
                trends_event_stats.fmt_size_mb(),
                trends_incident_stats.items,
                trends_incident_stats.fmt_percent(),
                trends_incident_stats.fmt_size_mb(),
            );
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
    );

    let percent_orphaned_trends = total_orphaned_trends as f64 / total_trends as f64 * 100.0;
    let percent_event_trends = total_event_trends as f64 / total_trends as f64 * 100.0;
    let percent_incident_trends = total_incident_trends as f64 / total_trends as f64 * 100.0;
    let percent_non_event_trends = 100.0 - percent_event_trends - percent_incident_trends;

    println!(
        "total_trends={} (100%) orphaned_trends={} ({:.*}%) event_trends={} ({:.*}%) incident_trends={} ({:.*}%)",
        total_trends,
        total_orphaned_trends,
        2, percent_orphaned_trends,
        total_event_trends,
        2, percent_event_trends,
        total_incident_trends,
        2, percent_incident_trends
    );

    println!("total_storage_items={}", total_storage_items);
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
