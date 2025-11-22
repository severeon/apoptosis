#!/usr/bin/env python3
"""
Metrics insertion and retrieval for experiment tracking.
"""

import json
import sqlite3
from typing import List, Dict, Any, Optional


def insert_metrics(
    con: sqlite3.Connection,
    exp_id: int,
    metrics: List[Dict[str, Any]]
):
    """
    Insert metrics for an experiment.

    Args:
        con: Database connection
        exp_id: Experiment ID
        metrics: List of metric dicts with keys:
            - step: Training step
            - layer: Layer name
            - mean: Mean activation
            - p05, p50, p95: Percentiles
            - var: Variance
            - hist: Histogram data (will be JSON serialized)
    """
    if not metrics:
        return

    cur = con.cursor()
    rows = []

    for m in metrics:
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
            json.dumps(m.get("hist")) if m.get("hist") is not None else None,
        ))

    cur.executemany("""
        INSERT INTO metrics (experiment_id, step, layer, mean, p05, p50, p95, var, hist_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    con.commit()


def get_metrics_for_experiment(
    con: sqlite3.Connection,
    exp_id: int,
    layer: Optional[str] = None,
    limit: Optional[int] = None
) -> List[sqlite3.Row]:
    """
    Get metrics for an experiment.

    Args:
        con: Database connection
        exp_id: Experiment ID
        layer: Optional layer name to filter by
        limit: Maximum number of rows to return

    Returns:
        List of metric rows
    """
    cur = con.cursor()

    if layer:
        query = """
            SELECT * FROM metrics
            WHERE experiment_id=? AND layer=?
            ORDER BY step ASC
        """
        params = (exp_id, layer)
    else:
        query = """
            SELECT * FROM metrics
            WHERE experiment_id=?
            ORDER BY step ASC, layer ASC
        """
        params = (exp_id,)

    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query, params)
    return cur.fetchall()


def get_latest_metrics_for_run(
    con: sqlite3.Connection,
    exp_id: int,
    num_steps: int = 10
) -> List[sqlite3.Row]:
    """
    Get the most recent N steps of metrics for an experiment.

    Args:
        con: Database connection
        exp_id: Experiment ID
        num_steps: Number of recent steps to retrieve

    Returns:
        List of metric rows
    """
    cur = con.cursor()

    cur.execute("""
        SELECT * FROM metrics
        WHERE experiment_id=? AND step IN (
            SELECT DISTINCT step FROM metrics
            WHERE experiment_id=?
            ORDER BY step DESC
            LIMIT ?
        )
        ORDER BY step DESC, layer ASC
    """, (exp_id, exp_id, num_steps))

    return cur.fetchall()


def get_metric_summary_by_layer(
    con: sqlite3.Connection,
    exp_id: int
) -> Dict[str, Dict[str, float]]:
    """
    Get aggregated metric summary for each layer.

    Args:
        con: Database connection
        exp_id: Experiment ID

    Returns:
        Dict mapping layer name to summary stats (avg_mean, avg_var, etc.)
    """
    cur = con.cursor()

    cur.execute("""
        SELECT
            layer,
            AVG(mean) as avg_mean,
            AVG(var) as avg_var,
            AVG(p50) as avg_median,
            MIN(mean) as min_mean,
            MAX(mean) as max_mean
        FROM metrics
        WHERE experiment_id=?
        GROUP BY layer
    """, (exp_id,))

    results = {}
    for row in cur.fetchall():
        results[row['layer']] = {
            'avg_mean': row['avg_mean'],
            'avg_var': row['avg_var'],
            'avg_median': row['avg_median'],
            'min_mean': row['min_mean'],
            'max_mean': row['max_mean']
        }

    return results
