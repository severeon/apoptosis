#!/usr/bin/env python3
"""
Event insertion and retrieval for experiment tracking.

Tracks apoptosis, senescence, and other lifecycle events.
"""

import sqlite3
from typing import List, Dict, Any, Optional


def insert_events(
    con: sqlite3.Connection,
    exp_id: int,
    events: List[Dict[str, Any]]
):
    """
    Insert events for an experiment.

    Args:
        con: Database connection
        exp_id: Experiment ID
        events: List of event dicts with keys:
            - event_type: Type of event ('apoptosis', 'senescence', etc.)
            - layer: Layer name
            - neuron_index: Index of affected neuron
            - step: Training step when event occurred
    """
    if not events:
        return

    cur = con.cursor()
    rows = [
        (exp_id, e.get("event_type"), e.get("layer"), e.get("neuron_index"), e.get("step"))
        for e in events
    ]

    cur.executemany("""
        INSERT INTO events (experiment_id, event_type, layer, neuron_index, step)
        VALUES (?, ?, ?, ?, ?)
    """, rows)

    con.commit()


def get_events_for_experiment(
    con: sqlite3.Connection,
    exp_id: int,
    event_type: Optional[str] = None,
    layer: Optional[str] = None,
    limit: Optional[int] = None
) -> List[sqlite3.Row]:
    """
    Get events for an experiment.

    Args:
        con: Database connection
        exp_id: Experiment ID
        event_type: Optional event type to filter by
        layer: Optional layer name to filter by
        limit: Maximum number of rows to return

    Returns:
        List of event rows
    """
    cur = con.cursor()

    # Build query dynamically based on filters
    query = "SELECT * FROM events WHERE experiment_id=?"
    params = [exp_id]

    if event_type:
        query += " AND event_type=?"
        params.append(event_type)

    if layer:
        query += " AND layer=?"
        params.append(layer)

    query += " ORDER BY step ASC"

    if limit:
        query += f" LIMIT {limit}"

    cur.execute(query, params)
    return cur.fetchall()


def count_events_by_type(
    con: sqlite3.Connection,
    exp_id: int
) -> Dict[str, int]:
    """
    Count events by type for an experiment.

    Args:
        con: Database connection
        exp_id: Experiment ID

    Returns:
        Dict mapping event_type to count
    """
    cur = con.cursor()

    cur.execute("""
        SELECT event_type, COUNT(*) as count
        FROM events
        WHERE experiment_id=?
        GROUP BY event_type
    """, (exp_id,))

    return {row['event_type']: row['count'] for row in cur.fetchall()}


def count_events_by_layer(
    con: sqlite3.Connection,
    exp_id: int,
    event_type: Optional[str] = None
) -> Dict[str, int]:
    """
    Count events by layer for an experiment.

    Args:
        con: Database connection
        exp_id: Experiment ID
        event_type: Optional event type to filter by

    Returns:
        Dict mapping layer name to count
    """
    cur = con.cursor()

    if event_type:
        cur.execute("""
            SELECT layer, COUNT(*) as count
            FROM events
            WHERE experiment_id=? AND event_type=?
            GROUP BY layer
        """, (exp_id, event_type))
    else:
        cur.execute("""
            SELECT layer, COUNT(*) as count
            FROM events
            WHERE experiment_id=?
            GROUP BY layer
        """, (exp_id,))

    return {row['layer']: row['count'] for row in cur.fetchall()}


def get_event_timeline(
    con: sqlite3.Connection,
    exp_id: int,
    bin_size: int = 100
) -> List[Dict[str, Any]]:
    """
    Get event timeline binned by step.

    Args:
        con: Database connection
        exp_id: Experiment ID
        bin_size: Size of step bins (default: 100)

    Returns:
        List of dicts with step_bin, event_type, and count
    """
    cur = con.cursor()

    cur.execute(f"""
        SELECT
            (step / {bin_size}) * {bin_size} as step_bin,
            event_type,
            COUNT(*) as count
        FROM events
        WHERE experiment_id=?
        GROUP BY step_bin, event_type
        ORDER BY step_bin, event_type
    """, (exp_id,))

    return [
        {
            'step_bin': row['step_bin'],
            'event_type': row['event_type'],
            'count': row['count']
        }
        for row in cur.fetchall()
    ]
