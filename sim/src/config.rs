pub struct Config {
    pub ticks: usize,
    pub percent_event_duration: f64,
    pub percent_event_to_incident: f64,
    pub mean_event_len: usize,
    pub measurements_ttl: usize,
    pub trends_ttl: usize,
    pub events_ttl: usize,
    pub incidents_ttl: usize,
    pub phenomena_ttl: usize,
}
