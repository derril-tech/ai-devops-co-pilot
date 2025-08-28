"""
Database configuration and connection management
"""
import os
from typing import Optional
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from contextlib import contextmanager

# Database connection settings
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://copilot_user:copilot_password@localhost:5432/devops_copilot")
CLICKHOUSE_URL = os.getenv("CLICKHOUSE_URL", "clickhouse://copilot_user:copilot_password@localhost:9000/devops_copilot")

# PostgreSQL connection pool settings
POOL_SIZE = int(os.getenv("DB_POOL_SIZE", "10"))
MAX_OVERFLOW = int(os.getenv("DB_MAX_OVERFLOW", "20"))
POOL_TIMEOUT = int(os.getenv("DB_POOL_TIMEOUT", "30"))

# Create PostgreSQL engine
postgres_engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=POOL_SIZE,
    max_overflow=MAX_OVERFLOW,
    pool_timeout=POOL_TIMEOUT,
    pool_pre_ping=True,
    echo=False  # Set to True for SQL query logging in development
)

# Create ClickHouse engine
clickhouse_engine = create_engine(
    CLICKHOUSE_URL,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_timeout=30,
    pool_pre_ping=True,
    echo=False
)

# Create session factories
PostgresSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=postgres_engine)
ClickHouseSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=clickhouse_engine)

# Metadata for table creation
metadata = MetaData()


@contextmanager
def get_postgres_session() -> Session:
    """Get a PostgreSQL database session"""
    session = PostgresSessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def get_clickhouse_session() -> Session:
    """Get a ClickHouse database session"""
    session = ClickHouseSessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_database():
    """Initialize database tables"""
    from ..models.base import Base
    Base.metadata.create_all(bind=postgres_engine)


def close_database_connections():
    """Close all database connections"""
    postgres_engine.dispose()
    clickhouse_engine.dispose()


# Row Level Security (RLS) helpers
def set_org_context(session: Session, org_id: str):
    """Set the organization context for RLS"""
    session.execute(f"SET LOCAL app.org_id = '{org_id}'")


def enable_rls(session: Session):
    """Enable Row Level Security"""
    session.execute("SET LOCAL app.enable_rls = 'true'")


def disable_rls(session: Session):
    """Disable Row Level Security"""
    session.execute("SET LOCAL app.enable_rls = 'false'")


# Health check functions
def check_postgres_health() -> bool:
    """Check PostgreSQL connection health"""
    try:
        with get_postgres_session() as session:
            session.execute("SELECT 1")
        return True
    except Exception:
        return False


def check_clickhouse_health() -> bool:
    """Check ClickHouse connection health"""
    try:
        with get_clickhouse_session() as session:
            session.execute("SELECT 1")
        return True
    except Exception:
        return False


# Database migration helpers
def get_current_schema_version() -> Optional[str]:
    """Get current database schema version"""
    try:
        with get_postgres_session() as session:
            result = session.execute("SELECT version FROM alembic_version LIMIT 1")
            return result.scalar()
    except Exception:
        return None


def run_migrations():
    """Run database migrations using Alembic"""
    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
