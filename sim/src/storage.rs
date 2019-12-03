use std::cmp::Ordering;

const GC_INTERVAL: usize = 600;

fn find_index<T: Ord>(v: &Vec<T>, val: &T) -> usize {
    match v.binary_search(val) {
        Ok(idx) => idx + 1,
        Err(idx) => idx,
    }
}

pub enum StorageType {
    Sample,
    Measurement,
    Trend,
    Detection,
    Incident,
    Phenomena,
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

    pub fn new_measurement(
        ts: usize,
        ttl: usize,
        is_event: Option<bool>,
        is_incident: Option<bool>,
    ) -> StorageItem {
        StorageItem::new(StorageType::Measurement, ts, ttl, is_event, is_incident)
    }

    pub fn new_trend(
        ts: usize,
        ttl: usize,
        is_event: Option<bool>,
        is_incident: Option<bool>,
    ) -> StorageItem {
        StorageItem::new(StorageType::Trend, ts, ttl, is_event, is_incident)
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
    pub time: usize,
}

impl Storage {
    pub fn new() -> Storage {
        Storage {
            storage_items: vec![],
            time: 0,
        }
    }

    pub fn tick(&mut self) {
        self.time += 1;
        if self.time % GC_INTERVAL == 0 {
            self.gc()
        }
    }

    pub fn add(&mut self, storage_item: StorageItem) {
        self.storage_items.push(storage_item);
        self.tick();
    }

    pub fn add_many(&mut self, storage_items: &mut Vec<StorageItem>) {
        self.storage_items.append(storage_items);
        self.tick()
    }

    pub fn gc(&mut self) {
        self.storage_items.sort();
        match self
            .storage_items
            .binary_search(&StorageItem::from_ttl(self.time))
        {
            Ok(idx) => self.storage_items.drain(0..=idx),
            Err(idx) => self.storage_items.drain(0..idx),
        };
    }
}
