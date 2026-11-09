"""SQLite persistence functions for the example project."""

import sqlite3
from pathlib import Path


DATABASE_PATH = Path(__file__).with_name("demo.sqlite3")


def initialize_database() -> None:
    """Create the users table when it does not exist."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT)"
        )


def save_user(email: str) -> None:
    """Save one user email in the local SQLite database."""
    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("INSERT INTO users (email) VALUES (?)", (email,))
