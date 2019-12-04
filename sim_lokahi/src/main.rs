pub mod config;
pub mod constants;
pub mod sim;
pub mod storage;

fn main() {
    let conf = config::Config {
        //        ticks: constants::SECONDS_PER_YEAR * 3,
        ticks: constants::SECONDS_PER_DAY,
        percent_event_duration: constants::ESTIMATED_PERCENT_EVENT_DATA_DURATION,
        percent_event_to_incident: 0.0,
        mean_event_len: constants::ESTIMATED_EVENT_LEN_S,
        samples_ttl: constants::SECONDS_PER_FIFTEEN_MINUTES,
        measurements_ttl: constants::DEFAULT_MEASUREMENT_TTL,
        trends_ttl: constants::DEFAULT_TRENDS_TTL,
        events_ttl: constants::DEFAULT_EVENTS_TTL,
        incidents_ttl: constants::DEFAULT_INCIDENTS_TTL,
        phenomena_ttl: constants::DEFAULT_PHENOMENA_TTL,
        num_sensors: 1,
        bytes_per_sample: constants::ESTIMATED_BYTES_PER_META_SAMPLE_8000,
        bytes_per_measurement: constants::ESTIMATED_BYTES_PER_MEASUREMENT,
        bytes_per_trend: constants::ESTIMATED_BYTES_PER_TREND,
        bytes_per_event: constants::ESTIMATED_BYTES_PER_EVENT,
        bytes_per_incident: constants::ESTIMATED_BYTES_PER_INCIDENT,
        print_info_every_n_ticks: 500_000,
        //        write_info_every_n_ticks: 3600,
        write_info_every_n_ticks: 300,
        out_file: "/home/opq/Documents/anthony/dissertation/plotting/lokahi/sim_data_8000_iml.txt"
            .to_string(),
    };
    let mut simulation = sim::Simulation::new(conf);
    simulation.run_simulation();
}
