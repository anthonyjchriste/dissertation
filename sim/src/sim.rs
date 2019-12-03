use crate::config::Config;
use crate::storage::{Storage, StorageItem};
use crate::{config, storage};
use rand;
use rand::prelude::ThreadRng;
use rand::Rng;

#[inline]
fn percent_chance(chance: f64, rng: &mut ThreadRng) -> bool {
    rng.gen_range(0.0, 1.0) < chance
}

pub struct Simulation {
    rng: ThreadRng,
    storage: Storage,
    conf: Config,

    total_storage_items: usize,
    total_samples: usize,
    total_orphaned_samples: usize,
    total_event_samples: usize,
    total_incident_samples: usize,
    total_phenomena_samples: usize,
    total_measurements: usize,
    total_orphaned_measurements: usize,
    total_event_measurements: usize,
    total_phenomena_measurements: usize,
    total_incident_measurements: usize,
    total_trends: usize,
    total_orphaned_trends: usize,
    total_event_trends: usize,
    total_incident_trends: usize,
    total_phenomena_trends: usize,
    total_events: usize,
    total_orphaned_events: usize,
    total_incident_events: usize,
    total_phenomena_events: usize,
    total_incidents: usize,
    total_orphaned_incidents: usize,
    total_phenomena_incidents: usize,
}

impl Simulation {
    pub fn new(conf: Config) -> Simulation {
        Simulation {
            rng: rand::thread_rng(),
            storage: Storage::new(),
            conf,
            total_storage_items: 0,
            total_samples: 0,
            total_orphaned_samples: 0,
            total_event_samples: 0,
            total_incident_samples: 0,
            total_phenomena_samples: 0,
            total_measurements: 0,
            total_orphaned_measurements: 0,
            total_event_measurements: 0,
            total_phenomena_measurements: 0,
            total_incident_measurements: 0,
            total_trends: 0,
            total_orphaned_trends: 0,
            total_event_trends: 0,
            total_incident_trends: 0,
            total_phenomena_trends: 0,
            total_events: 0,
            total_orphaned_events: 0,
            total_incident_events: 0,
            total_phenomena_events: 0,
            total_incidents: 0,
            total_orphaned_incidents: 0,
            total_phenomena_incidents: 0,
        }
    }

    fn make_sample(&mut self, time: usize, is_event: bool, is_incident: bool) -> StorageItem {
        self.total_storage_items += 1;
        self.total_samples += 1;
        let ttl = if !is_event && !is_incident {
            self.total_orphaned_samples += 1;
            time + self.conf.samples_ttl
        } else if is_event {
            self.total_event_samples += 1;
            time + self.conf.events_ttl
        } else {
            self.total_incident_samples += 1;
            time + self.conf.incidents_ttl
        };

        let is_event = if is_event { Some(is_event) } else { None };

        let is_incident = if is_incident { Some(is_incident) } else { None };

        storage::StorageItem::new_measurement(time, ttl, is_event, is_incident)
    }

    fn make_measurement(&mut self, time: usize, is_event: bool, is_incident: bool) -> StorageItem {
        self.total_storage_items += 1;
        self.total_measurements += 1;
        let ttl = if !is_event && !is_incident {
            self.total_orphaned_measurements += 1;
            time + self.conf.measurements_ttl
        } else if is_event {
            self.total_event_measurements += 1;
            time + self.conf.events_ttl
        } else {
            self.total_incident_measurements += 1;
            time + self.conf.incidents_ttl
        };

        let is_event = if is_event { Some(is_event) } else { None };

        let is_incident = if is_incident { Some(is_incident) } else { None };

        storage::StorageItem::new_measurement(time, ttl, is_event, is_incident)
    }

    fn make_trend(&mut self, time: usize, is_event: bool, is_incident: bool) -> StorageItem {
        self.total_storage_items += 1;
        self.total_trends += 1;
        let ttl = if !is_event && !is_incident {
            self.total_orphaned_trends += 1;
            time + self.conf.trends_ttl
        } else if is_event {
            self.total_event_trends += 1;
            time + self.conf.events_ttl
        } else {
            self.total_incident_trends += 1;
            time + self.conf.incidents_ttl
        };

        let is_event = if is_event { Some(is_event) } else { None };

        let is_incident = if is_incident { Some(is_incident) } else { None };

        storage::StorageItem::new_trend(time, ttl, is_event, is_incident)
    }

    fn display_info(&mut self, time: usize) {
        let measurement_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(145)),
            None,
            None,
        );
        let measurement_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(145)),
            Some(false),
            Some(false),
        );
        let measurement_event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(145)),
            Some(true),
            Some(false),
        );
        let measurement_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Measurement(145)),
            Some(false),
            Some(true),
        );
        let trends_stats =
            self.storage
                .stat_storage_items(Some(storage::StorageType::Trend(365)), None, None);
        let trends_orphaned_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(365)),
            Some(false),
            Some(false),
        );
        let trends_event_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(365)),
            Some(true),
            Some(false),
        );
        let trends_incident_stats = self.storage.stat_storage_items(
            Some(storage::StorageType::Trend(365)),
            Some(false),
            Some(true),
        );
        println!(
            "[{}/{} {:.*}%] m_t={} {} m_o={} {} {} m_e={} {} {} m_i={} {} {} t_t={} {} t_o={} {} {} t_e={} {} {} t_i={} {} {}",
            time,
            self.conf.ticks,
            1, time as f64 / self.conf.ticks as f64 * 100.0,
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

    fn display_summary(&self) {
        let percent_orphaned_measurements =
            self.total_orphaned_measurements as f64 / self.total_measurements as f64 * 100.0;
        let percent_event_measurements =
            self.total_event_measurements as f64 / self.total_measurements as f64 * 100.0;
        let percent_incident_measurements =
            self.total_incident_measurements as f64 / self.total_measurements as f64 * 100.0;

        println!(
            "total_measurements={} (100%) orphaned_measurements={} ({:.*}%) event_measurements={} ({:.*}%) incident_measurements={} ({:.*}%)",
            self.total_measurements,
            self.total_orphaned_measurements,
            2, percent_orphaned_measurements,
            self.total_event_measurements,
            2, percent_event_measurements,
            self.total_incident_measurements,
            2, percent_incident_measurements
        );

        let percent_orphaned_trends =
            self.total_orphaned_trends as f64 / self.total_trends as f64 * 100.0;
        let percent_event_trends =
            self.total_event_trends as f64 / self.total_trends as f64 * 100.0;
        let percent_incident_trends =
            self.total_incident_trends as f64 / self.total_trends as f64 * 100.0;
        let percent_non_event_trends = 100.0 - percent_event_trends - percent_incident_trends;

        println!(
            "total_trends={} (100%) orphaned_trends={} ({:.*}%) event_trends={} ({:.*}%) incident_trends={} ({:.*}%)",
            self.total_trends,
            self.total_orphaned_trends,
            2, percent_orphaned_trends,
            self.total_event_trends,
            2, percent_event_trends,
            self.total_incident_trends,
            2, percent_incident_trends
        );

        println!("total_storage_items={}", self.total_storage_items);
    }

    pub fn run_simulation(&mut self) {
        let mut storage_items_per_tick: Vec<storage::StorageItem> = vec![];

        for i in 0..self.conf.ticks {
            storage_items_per_tick.clear();
            for _ in 0..self.conf.num_sensors {
                let make_trend = i % 60 == 0;
                // Probability of belonging to an event
                if percent_chance(self.conf.percent_event_duration, &mut self.rng) {
                    // Probability of also belonging to an incident
                    if percent_chance(self.conf.percent_event_to_incident, &mut self.rng) {
                        // Store measurement saved by incident
                        storage_items_per_tick.push(self.make_measurement(i, false, true));
                        if make_trend {
                            // Store trend saved by incident
                            storage_items_per_tick.push(self.make_trend(i, false, true));
                        }
                    } else {
                        // Store measurement saved by event
                        storage_items_per_tick.push(self.make_measurement(i, true, false));
                        if make_trend {
                            // Store trend saved by event
                            storage_items_per_tick.push(self.make_trend(i, true, false));
                        }
                    }
                } else {
                    // Store orphaned measurement
                    storage_items_per_tick.push(self.make_measurement(i, false, false));
                    if make_trend {
                        // Store orphaned trend
                        storage_items_per_tick.push(self.make_trend(i, false, false));
                    }
                }
            }

            self.storage.add_many(&mut storage_items_per_tick, i);

            if i % 100_000 == 0 {
                self.display_info(i);
            }
        }

        self.display_summary();
    }
}
