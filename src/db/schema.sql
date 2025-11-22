-- Database schema for apoptosis experiments
-- Tracks experiments, metrics, and lifecycle events

CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uuid TEXT UNIQUE,
    timestamp TEXT,
    run_name TEXT,
    params TEXT,               -- JSON
    status TEXT,               -- pending | running | done | failed
    start_time REAL,
    end_time REAL,
    duration REAL,
    final_loss REAL,
    total_apoptosis_events INTEGER,
    total_senescence_events INTEGER,
    stdout TEXT,
    stderr TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    step INTEGER,
    layer TEXT,
    mean REAL,
    p05 REAL,
    p50 REAL,
    p95 REAL,
    var REAL,
    hist_json TEXT,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER,
    event_type TEXT,
    layer TEXT,
    neuron_index INTEGER,
    step INTEGER,
    FOREIGN KEY(experiment_id) REFERENCES experiments(id)
);
