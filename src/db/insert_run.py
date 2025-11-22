#!/usr/bin/env python3
"""
Experiment run CRUD operations for the database.
"""

import json
import time
import uuid
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional


def insert_experiment_row(
    con: sqlite3.Connection,
    params: Dict[str, Any],
    run_name: Optional[str] = None
) -> int:
    """
    Insert a new experiment row in 'pending' status.

    Args:
        con: Database connection
        params: Experiment parameters (will be JSON serialized)
        run_name: Optional run name (auto-generated if not provided)

    Returns:
        Experiment ID (row ID)
    """
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


def set_status_running(con: sqlite3.Connection, exp_id: int):
    """
    Mark an experiment as 'running' and record start time.

    Args:
        con: Database connection
        exp_id: Experiment ID
    """
    cur = con.cursor()
    cur.execute(
        "UPDATE experiments SET status=?, start_time=? WHERE id=?",
        ("running", time.time(), exp_id)
    )
    con.commit()


def finalize_experiment_row(
    con: sqlite3.Connection,
    exp_id: int,
    summary: Optional[Dict[str, Any]],
    stdout: str,
    stderr: str,
    rc: int = 0
):
    """
    Finalize an experiment with results and outputs.

    Args:
        con: Database connection
        exp_id: Experiment ID
        summary: Experiment summary dict (from ExperimentLogger)
        stdout: Standard output capture
        stderr: Standard error capture
        rc: Return code (0 for success, non-zero for failure)
    """
    cur = con.cursor()
    end_time = time.time()
    duration = None

    if summary and summary.get("duration"):
        duration = summary.get("duration")
    else:
        # Fallback: compute from start_time
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


def get_experiment_by_id(con: sqlite3.Connection, exp_id: int) -> Optional[sqlite3.Row]:
    """
    Get experiment by ID.

    Args:
        con: Database connection
        exp_id: Experiment ID

    Returns:
        Row dict or None if not found
    """
    cur = con.cursor()
    cur.execute("SELECT * FROM experiments WHERE id=?", (exp_id,))
    return cur.fetchone()


def get_experiments_by_status(
    con: sqlite3.Connection,
    status: str,
    limit: Optional[int] = None
) -> list:
    """
    Get experiments by status.

    Args:
        con: Database connection
        status: Status to filter by ('pending', 'running', 'done', 'failed')
        limit: Maximum number of rows to return

    Returns:
        List of experiment rows
    """
    cur = con.cursor()
    query = "SELECT * FROM experiments WHERE status=? ORDER BY timestamp DESC"
    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query, (status,))
    return cur.fetchall()


def get_top_experiments(
    con: sqlite3.Connection,
    limit: int = 10,
    order_by: str = 'final_loss'
) -> list:
    """
    Get top performing experiments.

    Args:
        con: Database connection
        limit: Number of experiments to return
        order_by: Column to order by (default: 'final_loss')

    Returns:
        List of experiment rows
    """
    cur = con.cursor()
    cur.execute(f"""
        SELECT * FROM experiments
        WHERE status='done' AND final_loss IS NOT NULL
        ORDER BY {order_by} ASC
        LIMIT ?
    """, (limit,))
    return cur.fetchall()


def delete_experiment(con: sqlite3.Connection, exp_id: int):
    """
    Delete an experiment and all its metrics/events.

    Args:
        con: Database connection
        exp_id: Experiment ID
    """
    cur = con.cursor()

    # Delete related data first (foreign key constraints)
    cur.execute("DELETE FROM metrics WHERE experiment_id=?", (exp_id,))
    cur.execute("DELETE FROM events WHERE experiment_id=?", (exp_id,))
    cur.execute("DELETE FROM experiments WHERE id=?", (exp_id,))

    con.commit()
