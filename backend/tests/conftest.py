from __future__ import annotations

import os

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://raguser:ragpass@db:5432/ragpodcast_test",
)

# Truncate order matters: children before parents (FK constraints).
TABLES_TO_CLEAR = ["chunk", "episode", "podcast"]


@pytest_asyncio.fixture
async def db_session():
    """A session against the real Postgres test DB, tables cleared after each test.

    Uses the already-migrated `ragpodcast_test` database (see backend README /
    conftest module docstring) reached over the docker-compose network as `db`.
    Truncating between tests (rather than wrapping in a rolled-back transaction)
    matches how the app itself commits mid-flow in ingestion/service.py.
    """
    engine = create_async_engine(TEST_DATABASE_URL)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    async with session_factory() as session:
        yield session

    async with engine.begin() as conn:
        for table in TABLES_TO_CLEAR:
            await conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))

    await engine.dispose()


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    """Point settings.data_dir at a throwaway directory for the test."""
    from rag_podcast.config import settings

    monkeypatch.setattr(settings, "data_dir", str(tmp_path))
    return tmp_path
