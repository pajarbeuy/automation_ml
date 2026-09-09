from __future__ import annotations

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from config import settings

# Support SQLite fallback for standalone local execution without PostgreSQL container
try:
    connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        echo=settings.debug,
        connect_args=connect_args,
    )
    # Test connection
    with engine.connect() as conn:
        pass
except Exception:
    # Graceful fallback to local SQLite database file
    sqlite_url = "sqlite:///./automl.db"
    engine = create_engine(
        sqlite_url,
        pool_pre_ping=True,
        echo=False,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
