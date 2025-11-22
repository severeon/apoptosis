import itertools
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from multiprocessing import Pool

# ==============================================================
# Configuration: EXPERIMENT SPACES
# ==============================================================

CONFIG = {
    "fitness_alpha":        [2],
    "fitness_beta":         [2],
    "fitness_gamma":        [4],
    "activation_ema_decay": [1.1],

    # Apoptosis/Senescence knobs
    "senescence_low_pct":         [0.1],
    "senescence_patience":        [10],
    "slope_window":               [10],
    "slope_threshold":            [2e-4],
    "max_escalations_per_step":   [8],
    "max_kills_per_layer":        [8],

    # Model knobs
    "d_model":            [128],
    "n_heads":            [4],
    "n_layers":           [8],
    "seq_len":            [64],

    # Training knobs (kept short for fast exploration)
    "batch_size":               [64],
    "lr":                       [2e-3],
    "lifecycle_warmup_steps":   [75],
    "num_steps":                [300],   # short runs
}

TRAIN_PY = "train.py"  # Path to your train script


# ==============================================================
# Utility – turn dict into CLI args
# ==============================================================

def make_args(param_dict):
    args = []
    for k, v in param_dict.items():
        args.append(f"--{k}")
        args.append(str(v))
    return args

# ==============================================================
# Run a single experiment
# ==============================================================

def run_experiment(exp):
    run_id, params, out_dir = exp
    params["run_name"] = f"rid{run_id}_fa{params['fitness_alpha']}_fb{params['fitness_beta']}_fg{params['fitness_gamma']}_ed{params['activation_ema_decay']}|lp{params['senescence_low_pct']}_sp{params['senescence_patience']}_sw{params['slope_window']}_st{params['slope_threshold']}|me{params['max_escalations_per_step']}_mk{params['max_kills_per_layer']}|dm{params['d_model']}_nh{params['n_heads']}_nl{params['n_layers']}_sl{params['seq_len']}_bs{params['batch_size']}_lr{params['lr']}_ws{params['lifecycle_warmup_steps']}_ns{params['num_steps']}"

    if params['max_kills_per_layer'] > params['max_escalations_per_step']:
        return

    args = ["python", TRAIN_PY] + make_args(params)
    print(args)

    result = {}
    start = time.time()

    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=3600)
        end = time.time()

        result = {
            "run_id": run_id,
            "params": params,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "return_code": proc.returncode,
            "duration_sec": end - start,
        }
    except Exception as e:
        result = {
            "run_id": run_id,
            "params": params,
            "error": str(e),
        }

    # Write result file
    result_file = out_dir / f"run_{run_id}_result.json"
    with result_file.open("w") as f:
        json.dump(result, f, indent=2)

    return result_file


# ==============================================================
# Build parameter combinations (Cartesian or subsample)
# ==============================================================

def param_grid(config):
    keys = list(config.keys())
    values = [config[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


# ==============================================================
# Master sweep
# ==============================================================

def main(workers=4, max_runs=None):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(f"sweep_runs/{timestamp}")
    sweep_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting sweep → {sweep_dir}")

    combos = list(param_grid(CONFIG))

    if max_runs:
        combos = combos[:max_runs]

    experiments = []
    for i, params in enumerate(combos):
        out_dir = sweep_dir / f"run_{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        experiments.append((i, params, out_dir))

    # Parallel processing
    with Pool(processes=workers) as p:
        res_files = p.map(run_experiment, experiments)

    print("\nSweep complete.")
    print(f"Results saved in: {sweep_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_runs", type=int, default=None)

    args = parser.parse_args()
    main(workers=args.workers, max_runs=args.max_runs)
