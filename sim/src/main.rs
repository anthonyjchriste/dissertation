pub mod config;
pub mod constants;
pub mod sim;
pub mod storage;

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
    let mut simulation = sim::Simulation::new(conf);
    simulation.run_simulation()
}
