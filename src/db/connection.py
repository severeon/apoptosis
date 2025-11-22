#!/usr/bin/env python3
"""
Database connection management for experiment tracking.
"""

import sqlite3
from pathlib import Path
from typing import Optional

# Default database path
DEFAULT_DB_PATH = "experiments.db"


def get_schema() -> str:
    """Load SQL schema from schema.sql file."""
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path) as f:
        return f.read()


def open_db(db_path: str = DEFAULT_DB_PATH, timeout: float = 30.0) -> sqlite3.Connection:
    """
    Open a connection to the experiments database.

    Args:
        db_path: Path to SQLite database file
        timeout: Timeout in seconds for acquiring locks (default: 30.0)

    Returns:
        SQLite connection with Row factory enabled
    """
    con = sqlite3.connect(db_path, timeout=timeout, check_same_thread=False)
    con.row_factory = sqlite3.Row
    return con


def init_db(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Initialize database with schema if it doesn't exist.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection
    """
    con = open_db(db_path)
    cur = con.cursor()
    schema = get_schema()
    cur.executescript(schema)
    con.commit()
    return con


def ensure_db_exists(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Ensure database exists and is initialized.

    This is a convenience function that checks if the database file exists,
    and initializes it if not.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLite connection
    """
    db_file = Path(db_path)
    if not db_file.exists():
        return init_db(db_path)
    return open_db(db_path)
