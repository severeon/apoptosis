# autoscaler.py
import json
import random
import sqlite3
import time
from datetime import datetime

DB_PATH = "experiments.db"
POLL = 6.0  # seconds
EXPLOIT_THRESHOLD = 1.5   # final_loss threshold — tweak to your scale (lower = more selective)
FOLLOWUPS = 3
SCALE_FACTOR = 2         # multiply num_steps by this for followups
MUTATION_P = 0.2         # fraction of hyperparams to mutate
MAX_NUM_STEPS = 5000

def open_db():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def get_recent_done(con, window_sec=120):
    cur = con.cursor()
    t_cut = (datetime.utcnow().timestamp() - window_sec)
    # find runs completed in last window_sec seconds
    cur.execute("SELECT id, final_loss, params FROM experiments WHERE status='done' AND end_time > ?", (t_cut,))
    rows = cur.fetchall()
    return rows

def propose_followups(params_json, n=FOLLOWUPS, scale=SCALE_FACTOR):
    params = json.loads(params_json)
    base_steps = int(params.get("num_steps", 300))
    new_steps = min(int(base_steps * scale), MAX_NUM_STEPS)
    children = []
    keys = list(params.keys())
    for i in range(n):
        child = dict(params)
        child["num_steps"] = new_steps
        # mutate a few numeric knobs
        for k in random.sample(keys, max(1, int(len(keys)*MUTATION_P))):
            if isinstance(child[k], (int, float)) and k not in ("num_steps",):
                if isinstance(child[k], int):
                    child[k] = max(1, int(child[k] + random.randint(-1, 2)))
                else:
                    child[k] = float(child[k] * random.uniform(0.9, 1.1))
        # unique run_name
        child["run_name"] = f"exploit_{int(time.time())}_{random.randint(0,9999)}"
        children.append(child)
    return children

def insert_pending(con, params):
    cur = con.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO experiments (uuid, timestamp, run_name, params, status) VALUES (?, ?, ?, ?, ?)",
                (str(time.time()) + "_" + str(random.randint(0,9999)), ts, params.get("run_name","auto"), json.dumps(params), "pending"))
    con.commit()
    return cur.lastrowid

def autoscale_loop():
    con = open_db()
    print("[autoscaler] starting")
    seen = set()
    try:
        while True:
            done = get_recent_done(con, window_sec=10)
            for r in done:
                rid = r["id"]
                if rid in seen:
                    continue
                seen.add(rid)
                loss = r["final_loss"]
                if loss is None:
                    continue
                print(f"[autoscaler] saw done run {rid} loss={loss:.4f}")
                if loss < EXPLOIT_THRESHOLD:
                    print(f"[autoscaler] promising run -> scheduling followups")
                    children = propose_followups(r["params"], n=FOLLOWUPS)
                    for c in children:
                        insert_pending(con, c)
                        print(f"[autoscaler] inserted followup run_name={c.get('run_name')}")
            time.sleep(POLL)
    except KeyboardInterrupt:
        print("[autoscaler] exiting")

if __name__ == "__main__":
    autoscale_loop()
