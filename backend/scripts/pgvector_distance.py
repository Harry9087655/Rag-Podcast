"""Probe: confirm `pgvector.sqlalchemy`'s `.cosine_distance()` compiles to the
`<=>` operator (index-eligible under `vector_cosine_ops`), and that the recall
SQL — chunk ⋈ episode ⋈ podcast, order-by distance, optional podcast filter —
assembles and executes as expected.

Run:  cd backend && uv run python scripts/pgvector_distance.py
"""
import asyncio
import random

from sqlalchemy import select, text, bindparam
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector
from rag_podcast.db import async_session
from rag_podcast.models.chunk import Chunk
from rag_podcast.models.episode import Episode
from rag_podcast.models.podcast import Podcast


def build_select(podcast_id: int | None, limit: int):
    vector = [0.0] * 1024  # placeholder — length is irrelevant for SQL text
    stmt = (
        select(Chunk, Episode.title, Podcast.name)
        .join(Episode, Episode.id == Chunk.episode_id)
        .join(Podcast, Podcast.id == Chunk.podcast_id)
        .order_by(Chunk.embedding.cosine_distance(vector))
        .limit(limit)
    )
    if podcast_id is not None:
        stmt = stmt.where(Chunk.podcast_id == podcast_id)
    return stmt


def compile_check():
    no_filter = str(build_select(None, 5).compile(dialect=postgresql.dialect()))
    with_filter = str(build_select(3, 5).compile(dialect=postgresql.dialect()))

    print("compiled SQL (no podcast filter):")
    print(no_filter)
    print()
    print("compiled SQL (podcast_id = 3):")
    print(with_filter)
    print()

    assert "<=>" in no_filter, "cosine_distance did NOT compile to `<=>`"
    assert "chunk.podcast_id" in with_filter
    print("PASS: .cosine_distance() -> `<=>`; podcast filter present when set")


async def live_check():
    random.seed(0)
    vector = [random.random() for _ in range(1024)]  # non-zero: avoid NaN on cosine
    stmt = (
        select(
            Chunk.id,
            Chunk.embedding.cosine_distance(vector).label("distance"),
            Episode.title,
            Podcast.name,
        )
        .join(Episode, Episode.id == Chunk.episode_id)
        .join(Podcast, Podcast.id == Chunk.podcast_id)
        .order_by(Chunk.embedding.cosine_distance(vector))
        .limit(3)
    )
    async with async_session() as session:
        rows = (await session.execute(stmt)).all()
    print(f"live query: {len(rows)} row(s)")
    for r in rows:
        print(f"  chunk {r.id}  dist={r.distance:.4f}  {r.title!r}  {r.name!r}")


async def direct_statement_check():
    random.seed(1)
    vector = [random.random() for _ in range(1024)]
    async with async_session() as session:
        res = (await session.execute(text(
            """select c.id, c.embedding <=> CAST(:test_vector as vector(1024)) as distance, 
                e.title, p.name from Chunk c join Episode e on
                c.episode_id = e.id 
                join Podcast p on c.podcast_id = p.id
                where p.id = 3;""")
                .bindparams(bindparam("test_vector", type_=Vector(1024))),
                {'test_vector':vector})).all()
    print(f"live query: {len(res)} row(s)")
    for r in res:
        print(f"  chunk {r.id}  dist={r.distance:.4f}  {r.title!r}  {r.name!r}")
        



if __name__ == "__main__":
    # compile_check()
    asyncio.run(direct_statement_check())
