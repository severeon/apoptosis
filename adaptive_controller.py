#!/usr/bin/env python3
"""
adaptive_controller.py

Adaptive experimentation daemon for apoptosis/senescence hyperparameter search.

- Uses SQLite DB to store experiments and metrics.
- Proposes candidates using a mixture of random / elite-mutation / local-search.
- Launches train.py (uses the uploaded file at /mnt/data/train.py) with --output_json.
- Ingests JSON output and stores metrics/events in the DB.
"""

import argparse
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from math import exp

# -----------------------
# Path to your train.py (file you uploaded)
# Developer instruction: use this path.
TRAIN_PY = "train.py"

# Default DB path
DEFAULT_DB = "experiments.db"

# -----------------------
# Default search space (expand as needed)
SPACE = {
    "fitness_alpha": [0.5, 1.0, 1.5, 2.0],
    "fitness_beta": [0.5, 1.0, 1.5, 2.0],
    "fitness_gamma": [0.5, 1.0, 1.5, 2.0, 3.0],
    "activation_ema_decay": [0.75, 0.85, 0.9, 0.95],
    "senescence_low_pct": [0.01, 0.05, 0.1],
    "senescence_patience": [5, 10, 20],
    "slope_window": [3, 5, 10, 15],
    "slope_threshold": [0.01, 0.025, 0.05],
    "max_escalations_per_step": [1, 2, 3, 5],
    "max_kills_per_layer": [1, 2, 3],
    # model/training knobs (kept small for fast experiments)
    "d_model": [32, 64, 128],
    "n_layers": [2, 4, 6],
    "batch_size": [64, 128],
    "lr": [1e-3, 3e-4, 1e-4],
    "num_steps": [200, 300, 500],
}

# -----------------------
# DB schema
# -----------------------
SCHEMA = """
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
"""

# -----------------------
# Helpers
# -----------------------

def open_db(db_path):
    con = sqlite3.connect(db_path, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def init_db(db_path):
    con = open_db(db_path)
    cur = con.cursor()
    cur.executescript(SCHEMA)
    con.commit()
    return con

def insert_experiment_row(con, params, run_name=None):
    cur = con.cursor()
    uid = str(uuid.uuid4())
    ts = datetime.utcnow().isoformat()
    params_json = json.dumps(params)
    run_name = run_name or f"auto_{uid[:8]}"
    cur.execute("""
        INSERT INTO experiments (uuid, timestamp, run_name, params, status)
        VALUES (?, ?, ?, ?, ?)
    """, (uid, ts, run_name, params_json, "pending"))
    con.commit()
    return cur.lastrowid

def set_status_running(con, exp_id):
    cur = con.cursor()
    cur.execute("UPDATE experiments SET status=?, start_time=? WHERE id=?",
                ("running", time.time(), exp_id))
    con.commit()

def finalize_experiment_row(con, exp_id, summary, stdout, stderr, rc=0):
    cur = con.cursor()
    end_time = time.time()
    duration = None
    if summary and summary.get("duration"):
        duration = summary.get("duration")
    else:
        # fallback
        cur.execute("SELECT start_time FROM experiments WHERE id=?", (exp_id,))
        row = cur.fetchone()
        if row and row["start_time"]:
            duration = end_time - row["start_time"]
    cur.execute("""
        UPDATE experiments
        SET status=?, end_time=?, duration=?, final_loss=?, total_apoptosis_events=?,
            total_senescence_events=?, stdout=?, stderr=?
        WHERE id=?
    """, (
        "done" if rc == 0 else "failed",
        end_time,
        duration,
        summary.get("final_loss") if summary else None,
        summary.get("total_apoptosis_events") if summary else None,
        summary.get("total_senescence_events") if summary else None,
        stdout,
        stderr,
        exp_id
    ))
    con.commit()

def insert_metrics(con, exp_id, metrics):
    cur = con.cursor()
    if not metrics:
        return
    rows = []
    for m in metrics:
        # metrics expected to be list of dicts with layer and stats
        # we accept both single-layer-per-step entries and flattened arrays
        layer = m.get("layer")
        rows.append((
            exp_id,
            m.get("step"),
            layer,
            m.get("mean"),
            m.get("p05"),
            m.get("p50"),
            m.get("p95"),
            m.get("var"),
            json.dumps(m.get("hist")),
        ))
    cur.executemany("""
        INSERT INTO metrics (experiment_id, step, layer, mean, p05, p50, p95, var, hist_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    con.commit()

def insert_events(con, exp_id, events):
    if not events:
        return
    cur = con.cursor()
    rows = [(exp_id, e.get("event_type"), e.get("layer"), e.get("neuron_index"), e.get("step")) for e in events]
    cur.executemany("""
        INSERT INTO events (experiment_id, event_type, layer, neuron_index, step)
        VALUES (?, ?, ?, ?, ?)
    """, rows)
    con.commit()

# -----------------------
# Simple scoring function (lower is better)
# Use final_loss primarily, penalize runs with weird high apoptosis
# -----------------------
def score_summary(summary):
    if summary is None:
        return float("inf")
    loss = summary.get("final_loss")
    if loss is None:
        return float("inf")
    events = summary.get("total_apoptosis_events", 0) or 0
    # Penalty coefficient (tunable)
    penalty = 0.01
    return float(loss) + penalty * float(events)

# -----------------------
# Proposer functions
# -----------------------
def random_candidate(space):
    params = {}
    for k, vals in space.items():
        params[k] = random.choice(vals)
    return params

def mutate_param_value(val, space_vals):
    # If val is from discrete list, mutate by picking neighbor or random
    try:
        # numeric?
        v = float(val)
    except Exception:
        # fallback: pick random from allowed
        return random.choice(space_vals)
    # small gaussian multiplicative perturbation
    if v == 0:
        v2 = random.choice(space_vals)
    else:
        scale = random.uniform(0.85, 1.15)
        v2 = v * scale
        # clamp to nearest allowed value in space if present and numeric
        numeric_vals = [x for x in space_vals if isinstance(x, (int, float))]
        if numeric_vals:
            # choose closest
            v2 = min(numeric_vals, key=lambda a: abs(float(a) - v2))
    return v2

def mutate_candidate(parent_params, space, mutation_rate=0.25):
    child = dict(parent_params)  # shallow copy
    keys = list(space.keys())
    # mutate ~ mutation_rate fraction of keys
    kcount = max(1, int(len(keys) * mutation_rate))
    keys_to_mutate = random.sample(keys, kcount)
    for k in keys_to_mutate:
        child[k] = mutate_param_value(parent_params.get(k, random.choice(space[k])), space[k])
    return child

def propose_candidates_from_db(con, space, n_candidates=8, elite_frac=0.2):
    cur = con.cursor()
    cur.execute("SELECT id, final_loss, params FROM experiments WHERE status='done' AND final_loss IS NOT NULL ORDER BY final_loss ASC LIMIT 100")
    rows = cur.fetchall()
    params_list = []
    for r in rows:
        try:
            p = json.loads(r["params"])
            params_list.append((r["id"], r["final_loss"], p))
        except Exception:
            continue

    candidates = []
    # If no finished runs yet, return random batch
    if not params_list:
        for _ in range(n_candidates):
            candidates.append(random_candidate(space))
        return candidates

    # elites
    elites = [p for (_id, _loss, p) in params_list][:max(1, int(len(params_list) * elite_frac))]
    # generate children via mutation
    for e in elites:
        # generate several mutated children
        for _ in range(max(1, n_candidates // max(1, len(elites)))):
            child = mutate_candidate(e, space, mutation_rate=random.choice([0.2, 0.35, 0.5]))
            candidates.append(child)

    # add randoms to preserve exploration
    while len(candidates) < n_candidates:
        candidates.append(random_candidate(space))

    # ensure uniqueness by params JSON
    uniq = []
    seen = set()
    for c in candidates:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq[:n_candidates]


# -----------------------
# Runner: execute a train.py with params and ingest results
# -----------------------
def run_training_job(con, params, run_name=None, timeout=3600):
    # Insert pending row
    exp_id = insert_experiment_row(con, params, run_name=run_name)
    set_status_running(con, exp_id)

    # Build CLI args
    args = ["python", TRAIN_PY]
    for k, v in params.items():
        # map keys to CLI flags - ensure flag naming matches train.py
        args.append(f"--{k}")
        args.append(str(v))
    args.append("--output_json")

    # Launch
    start = time.time()
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        stdout = proc.stdout.strip()
        stderr = proc.stderr.strip()
        rc = proc.returncode
        # parse JSON from stdout — train.py should print one JSON object
        summary = None
        try:
            summary = json.loads(stdout)
        except Exception:
            # Try to extract last JSON substring
            import re
            m = re.search(r"\{.*\}\s*$", stdout, re.DOTALL)
            if m:
                try:
                    summary = json.loads(m.group(0))
                except Exception:
                    summary = None

        # insert metrics + events
        if summary:
            insert_metrics(con, exp_id, summary.get("metrics", []))
            insert_events(con, exp_id, summary.get("events", []))

        finalize_experiment_row(con, exp_id, summary or {}, stdout, stderr, rc)
        return exp_id, summary
    except Exception as e:
        # mark failed
        finalize_experiment_row(con, exp_id, None, "", str(e), rc=1)
        return exp_id, None

# -----------------------
# Main controller loop
# -----------------------
def controller_loop(db_path, space, workers=2, seed_jobs=8, poll_interval=5):
    random.seed(42)
    con = init_db(db_path)
    executor = ThreadPoolExecutor(max_workers=workers)

    active_futures = set()
    outstanding = []

    try:
        # Seed initial random experiments if DB nearly empty
        cur = con.cursor()
        cur.execute("SELECT COUNT(*) as c FROM experiments")
        c = cur.fetchone()["c"]
        if c < max(1, seed_jobs // 2):
            print(f"[controller] seeding {seed_jobs} random experiments")
            for i in range(seed_jobs):
                params = random_candidate(space)
                outstanding.append(executor.submit(run_training_job, con, params, run_name=f"seed_{i}"))

        while True:
            # prune done futures
            done = [f for f in outstanding if f.done()]
            for f in done:
                try:
                    res = f.result()
                    print(f"[controller] finished seed job => exp_id={res[0]}")
                except Exception as e:
                    print("[controller] seed job exception:", e)
                outstanding.remove(f)

            # Count running
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) as c FROM experiments WHERE status='running'")
            running = cur.fetchone()["c"]

            # If capacity, propose new candidates
            capacity = workers - running
            if capacity > 0:
                # propose batch
                candidates = propose_candidates_from_db(con, space, n_candidates=capacity * 2)
                # Submit only up to capacity
                for cparams in candidates[:capacity]:
                    print(f"[controller] launching job with params snippet: { {k: cparams[k] for k in list(cparams)[:6]} }")
                    outstanding.append(executor.submit(run_training_job, con, cparams, run_name=f"autogen"))

            # Sleep and poll DB for new results (and let jobs finish)
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("[controller] keyboard interrupt, shutting down")
    finally:
        executor.shutdown(wait=True)
        con.close()

# -----------------------
# CLI entrypoint
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent train jobs")
    parser.add_argument("--seed_jobs", type=int, default=6, help="Initial random seeds")
    parser.add_argument("--poll", type=float, default=6.0, help="Poll interval seconds")
    args = parser.parse_args()

    if not os.path.exists(TRAIN_PY):
        print("ERROR: train.py not found at:", TRAIN_PY)
        sys.exit(1)

    controller_loop(args.db, SPACE, workers=args.workers, seed_jobs=args.seed_jobs, poll_interval=args.poll)
