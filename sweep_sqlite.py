import itertools
import json
import os
import sqlite3
import subprocess
import time
from datetime import datetime
from multiprocessing import Pool

DB_PATH = "experiments.db"
TRAIN_PY = "train.py"

# ============================================================== #
# SQLite setup
# ============================================================== #

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    # create tables (idempotent)
    cur.executescript(open("schema.sql").read())
    con.commit()
    con.close()


def insert_experiment(params):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    columns = ["timestamp"] + list(params.keys())
    values = [datetime.now().isoformat()] + list(params.values())

    q = f"""
    INSERT INTO experiments ({",".join(columns)})
    VALUES ({",".join(["?"] * len(values))})
    """
    cur.execute(q, values)
    exp_id = cur.lastrowid

    con.commit()
    con.close()
    return exp_id


def complete_experiment(exp_id, result_summary):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    q = """
    UPDATE experiments
    SET duration=?, final_loss=?, total_apoptosis_events=?,
        exit_code=?, completed=1
    WHERE id=?
    """
    cur.execute(q, (
        result_summary.get("duration", None),
        result_summary.get("final_loss", None),
        result_summary.get("total_apoptosis_events", None),
        result_summary.get("exit_code", None),
        exp_id
    ))
    con.commit()
    con.close()


def insert_metrics(exp_id, metrics):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executemany("""
        INSERT INTO metrics (experiment_id, step, loss, mean_fitness, min_fitness, max_fitness)
        VALUES (?, ?, ?, ?, ?, ?)
    """, [
        (
            exp_id,
            m.get("step"),
            m.get("loss"),
            m.get("mean_fitness"),
            m.get("min_fitness"),
            m.get("max_fitness")
        )
        for m in metrics
    ])
    con.commit()
    con.close()


def insert_events(exp_id, events):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.executemany("""
        INSERT INTO events (experiment_id, event_type, layer, neuron_index, step)
        VALUES (?, ?, ?, ?, ?)
    """, [
        (
            exp_id,
            e.get("event_type"),
            e.get("layer"),
            e.get("neuron_index"),
            e.get("step")
        )
        for e in events
    ])
    con.commit()
    con.close()


def store_logs(exp_id, stdout, stderr):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO logs (experiment_id, stdout, stderr)
        VALUES (?, ?, ?)
    """, (exp_id, stdout, stderr))
    con.commit()
    con.close()


# ============================================================== #
# Parameter grid
# ============================================================== #

CONFIG = {
    "fitness_alpha": [0.5, 1.0],
    "fitness_beta": [0.5, 1.0],
    "fitness_gamma": [1.0, 2.0],
    "activation_ema_decay": [0.8, 0.95],
    "num_steps": [300]
}

def param_grid():
    keys = list(CONFIG.keys())
    vals = [CONFIG[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))


# ============================================================== #
# Running experiments
# ============================================================== #

def run_exp(params):
    exp_id = insert_experiment(params)

    args = ["python", TRAIN_PY]
    for k, v in params.items():
        args.append(f"--{k}")
        args.append(str(v))
    args.append("--output_json")  # your train.py must support this

    start = time.time()
    proc = subprocess.run(args, capture_output=True, text=True)
    end = time.time()

    stdout = proc.stdout
    stderr = proc.stderr

    # record logs
    store_logs(exp_id, stdout, stderr)

    # parse output JSON
    try:
        summary = json.loads(stdout)
    except Exception:
        summary = {
            "final_loss": None,
            "total_apoptosis_events": None,
            "metrics": [],
            "events": []
        }

    # store metrics + events
    insert_metrics(exp_id, summary.get("metrics", []))
    insert_events(exp_id, summary.get("events", []))

    # finalize row
    complete_experiment(exp_id, {
        "duration": end - start,
        "final_loss": summary.get("final_loss"),
        "total_apoptosis_events": summary.get("total_apoptosis_events"),
        "exit_code": proc.returncode
    })

    return exp_id


def main(workers=4, max_runs=None):
    init_db()
    params_list = list(param_grid())
    if max_runs:
        params_list = params_list[:max_runs]

    with Pool(processes=workers) as p:
        p.map(run_exp, params_list)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max_runs", type=int)
    args = p.parse_args()
    main(args.workers, args.max_runs)
