"""
app/db/connection.py
Database session and connection management for PostgreSQL (Aiven / Local).
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Generator, Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.exc import PendingRollbackError
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# Load .env file so DATABASE_URL (and other vars) are available
load_dotenv()

# Environment variable for database connection
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

Base = declarative_base()

engine = None
SessionLocal = None

if DATABASE_URL:
    # Handle sslmode or postgresql:// fix if user inputs postgres://
    url = DATABASE_URL
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    connect_args = {}
    if "sslmode" not in url and ("aivencloud" in url or "aiven" in url):
        connect_args["sslmode"] = "require"

    try:
        engine = create_engine(
            url,
            pool_pre_ping=True,       # Auto-reconnect if cloud connection is dropped
            pool_recycle=300,         # Recycle connections every 5 minutes
            connect_args=connect_args,
        )
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    except Exception as exc:
        print(f"[Warning] Failed to initialize database engine: {exc}")
        engine = None
        SessionLocal = None


def is_db_connected() -> bool:
    """Check if a valid database connection is available."""
    if engine is None or SessionLocal is None:
        return False
    try:
        with engine.connect() as conn:
            return True
    except Exception:
        return False


def get_db() -> Generator[Optional[Session], None, None]:
    """FastAPI dependency or context generator for database sessions."""
    if SessionLocal is None:
        yield None
        return

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except PendingRollbackError:
        # CRUD layer caught an exception internally and left the session in a
        # rolled-back state. Roll back cleanly so the client still gets a 200.
        db.rollback()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


@contextmanager
def get_db_context() -> Generator[Optional[Session], None, None]:
    """Context manager for standalone scripts or non-FastAPI tasks."""
    if SessionLocal is None:
        yield None
        return

    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db() -> None:
    """Create all tables in the database if they don't already exist."""
    if engine is not None:
        Base.metadata.create_all(bind=engine)
