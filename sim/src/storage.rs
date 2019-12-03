use crate::constants;
use std::cmp::Ordering;

fn find_index<T: Ord>(v: &Vec<T>, val: &T) -> usize {
    match v.binary_search(val) {
        Ok(idx) => idx + 1,
        Err(idx) => idx,
    }
}

#[derive(PartialEq, Clone)]
pub enum StorageType {
    Sample(usize),
    Measurement(usize),
    Trend(usize),
    Detection(usize),
    Incident(usize),
    Phenomena(usize),
}

pub struct StorageItemStatistic {
    pub items: usize,
    pub percent_total: f64,
    pub total_bytes: usize,
}

impl StorageItemStatistic {
    pub fn fmt_percent(&self) -> String {
        format!("{:.*}%", 2, self.percent_total)
    }

    pub fn fmt_size_mb(&self) -> String {
        format!("{:.*}MB", 2, self.total_bytes as f64 / 1_000_000.0)
    }
}

pub struct StorageItem {
    pub storage_type: StorageType,
    pub ts: usize,
    pub ttl: usize,
    pub is_event: bool,
    pub is_incident: bool,
}

impl StorageItem {
    pub fn new(
        storage_type: StorageType,
        ts: usize,
        ttl: usize,
        is_event: Option<bool>,
        is_incident: Option<bool>,
    ) -> StorageItem {
        StorageItem {
            storage_type,
            ts,
            ttl,
            is_event: is_event.unwrap_or(false),
            is_incident: is_incident.unwrap_or(false),
        }
    }

    pub fn from_ttl(ttl: usize) -> StorageItem {
        StorageItem::new_measurement(0, ttl, None, None)
    }

    pub fn new_sample(
        ts: usize,
        ttl: usize,
        is_event: Option<bool>,
        is_incident: Option<bool>,
    ) -> StorageItem {
        StorageItem::new(
            StorageType::Sample(constants::ESTIMATED_BYTES_PER_META_SAMPLE),
            ts,
            ttl,
            is_event,
            is_incident,
        )
    }

    pub fn new_measurement(
        ts: usize,
        ttl: usize,
        is_event: Option<bool>,
        is_incident: Option<bool>,
    ) -> StorageItem {
        StorageItem::new(
            StorageType::Measurement(constants::ESTIMATED_BYTES_PER_MEASUREMENT),
            ts,
            ttl,
            is_event,
            is_incident,
        )
    }

    pub fn new_trend(
        ts: usize,
        ttl: usize,
        is_event: Option<bool>,
        is_incident: Option<bool>,
    ) -> StorageItem {
        StorageItem::new(
            StorageType::Trend(constants::ESTIMATED_BYTES_PER_TREND),
            ts,
            ttl,
            is_event,
            is_incident,
        )
    }
}

impl PartialOrd for StorageItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.ttl.partial_cmp(&other.ttl)
    }
}

impl PartialEq for StorageItem {
    fn eq(&self, other: &Self) -> bool {
        self.ttl == other.ttl
    }
}

impl Ord for StorageItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ttl.cmp(&other.ttl)
    }
}

impl Eq for StorageItem {}

pub struct Storage {
    pub storage_items: Vec<StorageItem>,
}

impl Storage {
    pub fn new() -> Storage {
        Storage {
            storage_items: vec![],
        }
    }

    fn check_gc(&mut self, time: usize) {
        if time % constants::DEFAULT_GC_INTERVAL == 0 {
            self.gc(time);
        }
    }

    pub fn add(&mut self, storage_item: StorageItem, time: usize) {
        self.storage_items.push(storage_item);
        self.check_gc(time);
    }

    pub fn add_many(&mut self, storage_items: &mut Vec<StorageItem>, time: usize) {
        self.storage_items.append(storage_items);
        self.check_gc(time);
    }

    pub fn gc(&mut self, time: usize) {
        self.storage_items.sort();
        match self
            .storage_items
            .binary_search(&StorageItem::from_ttl(time))
        {
            Ok(idx) => self.storage_items.drain(0..=idx),
            Err(idx) => self.storage_items.drain(0..idx),
        };
    }

    pub fn stat_storage_items(
        &self,
        storage_type: Option<StorageType>,
        is_event: Option<bool>,
        is_incident: Option<bool>,
    ) -> StorageItemStatistic {
        let filtered_storage_items: Vec<&StorageItem> = self
            .storage_items
            .iter()
            .filter(|storage_item| {
                if let Some(storage_type) = &storage_type {
                    if storage_item.storage_type != *storage_type {
                        return false;
                    }
                }

                if let Some(is_event) = is_event {
                    if storage_item.is_event != is_event {
                        return false;
                    }
                }

                if let Some(is_incident) = is_incident {
                    if storage_item.is_incident != is_incident {
                        return false;
                    }
                }

                true
            })
            .collect();

        let items = filtered_storage_items.len();
        let total_items = if storage_type.is_some() {
            self.storage_items
                .iter()
                .filter(|storage_item| storage_item.storage_type == storage_type.clone().unwrap())
                .count()
        } else {
            self.storage_items.len()
        };
        let percent_total: f64 = items as f64 / total_items as f64 * 100.0;
        let total_bytes: usize = filtered_storage_items
            .iter()
            .map(|storage_item| match storage_item.storage_type {
                StorageType::Sample(bytes) => bytes,
                StorageType::Measurement(bytes) => bytes,
                StorageType::Trend(bytes) => bytes,
                StorageType::Detection(bytes) => bytes,
                StorageType::Incident(bytes) => bytes,
                StorageType::Phenomena(bytes) => bytes,
            })
            .sum();

        StorageItemStatistic {
            items,
            percent_total,
            total_bytes,
        }
    }
}
