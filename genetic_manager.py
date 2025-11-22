# genetic_manager.py
import json
import random
import sqlite3
import time
from datetime import datetime

DB_PATH = "experiments.db"
POPULATION_SIZE = 20
ELITE_FRACTION = 0.25
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.5
POLL = 8.0

SEARCH_KEYS = [
    "fitness_alpha", "fitness_beta", "fitness_gamma",
    "activation_ema_decay", "senescence_low_pct", "senescence_patience",
    "slope_window", "slope_threshold", "max_escalations_per_step",
    "max_kills_per_layer", "d_model", "n_layers", "batch_size", "lr", "num_steps",
]

def open_db():
    con = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con

def get_population(con, limit=1000):
    cur = con.cursor()
    cur.execute("SELECT id, final_loss, params FROM experiments WHERE status='done' AND final_loss IS NOT NULL ORDER BY final_loss ASC LIMIT ?", (limit,))
    rows = cur.fetchall()
    parsed = []
    for r in rows:
        try:
            parsed.append((r["id"], r["final_loss"], json.loads(r["params"])))
        except Exception:
            continue
    return parsed

def select_elites(pop, fraction=ELITE_FRACTION):
    n = max(1, int(len(pop) * fraction))
    return pop[:n]

def crossover(parent_a, parent_b):
    child = {}
    for k in SEARCH_KEYS:
        va = parent_a.get(k)
        vb = parent_b.get(k)
        if va is None and vb is None:
            continue
        # pick from either parent or average if numeric
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and random.random() < 0.5:
            # average or interpolation
            child[k] = (float(va) + float(vb)) / 2.0 if random.random() < 0.5 else random.choice([va, vb])
            # keep integer types integer
            if isinstance(va, int) and isinstance(vb, int):
                child[k] = int(round(child[k]))
        else:
            child[k] = random.choice([va if va is not None else vb, vb if vb is not None else va])
    return child

def mutate(child):
    for k in SEARCH_KEYS:
        if k not in child:
            continue
        if random.random() < MUTATION_RATE:
            v = child[k]
            if isinstance(v, int):
                child[k] = max(1, v + random.randint(-2, 2))
            elif isinstance(v, float):
                child[k] = float(v * random.uniform(0.85, 1.15))
            else:
                # categorical fallback: no-op
                pass
    # ensure run_name
    child["run_name"] = f"gen_{int(time.time())}_{random.randint(0,9999)}"
    return child

def insert_pending(con, params):
    cur = con.cursor()
    ts = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO experiments (uuid, timestamp, run_name, params, status) VALUES (?, ?, ?, ?, ?)",
                (str(time.time()) + "_" + str(random.randint(0,9999)), ts, params.get("run_name","auto"), json.dumps(params), "pending"))
    con.commit()
    return cur.lastrowid

def genetic_loop():
    con = open_db()
    print("[genetic] starting genetic manager")
    try:
        while True:
            pop = get_population(con, limit=200)
            if len(pop) < 10:
                # not enough data -> seed random experiments via insert
                for i in range(5):
                    # random simple seed
                    params = {
                        "fitness_alpha": random.choice([0.5,1.0,1.5]),
                        "fitness_beta": random.choice([0.5,1.0,1.5]),
                        "fitness_gamma": random.choice([1.0,1.5,2.0]),
                        "activation_ema_decay": random.choice([0.8,0.9,0.95]),
                        "num_steps": random.choice([200,300]),
                        "lr": random.choice([1e-3,3e-4])
                    }
                    insert_pending(con, params)
                time.sleep(POLL)
                continue

            elites = select_elites(pop)
            children = []
            # crossover elites pairwise
            for i in range(len(elites)):
                for j in range(i+1, len(elites)):
                    pa = elites[i][2]
                    pb = elites[j][2]
                    if random.random() < CROSSOVER_RATE:
                        child = crossover(pa, pb)
                        child = mutate(child)
                        children.append(child)
                    if len(children) >= POPULATION_SIZE:
                        break
                if len(children) >= POPULATION_SIZE:
                    break

            # if not enough children, fill with mutated elites
            while len(children) < POPULATION_SIZE:
                p = random.choice(elites)[2]
                child = mutate(dict(p))
                children.append(child)

            # insert children as pending
            for c in children:
                insert_pending(con, c)
                print(f"[genetic] inserted child run_name={c['run_name']}")
            # wait for controller to run them
            time.sleep(POLL * 3)
    except KeyboardInterrupt:
        print("[genetic] exiting")

if __name__ == "__main__":
    genetic_loop()
