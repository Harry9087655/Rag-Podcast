# Database Guidelines

> Database patterns and conventions for this project.

---

## Overview

SQLAlchemy 2.0 async ORM with `asyncpg` driver against PostgreSQL 16 + pgvector.
Migrations are managed via Alembic. Models live in `backend/src/rag_podcast/models/`.

---

## Scenario: JSONB for Semi-Structured Attachments

### 1. Scope / Trigger
- Trigger: Adding transcription output (`transcript_data`) to the Episode model.
  Decided against a separate `transcript_segments` table because the data is always
  read/written as a unit attached to one episode, and downstream indexing needs the
  full blob, not individual segment queries.

### 2. Signatures

```python
# models/episode.py
from sqlalchemy.dialects.postgresql import JSONB

transcript_data: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
```

Migration:
```python
# alembic/versions/XXXX_add_transcript_data.py
op.add_column("episode", sa.Column("transcript_data", JSONB, nullable=True))
```

### 3. Contracts

| Field | Type | Required | Constraints |
|-------|------|----------|-------------|
| `transcript_data` | `dict` or `None` | No (nullable) | Valid JSON when set |

JSONB structure:
```json
{
  "model": "small",
  "language": "en",
  "duration": 523.14,
  "segments": [
    {"text": "...", "start": 0.0, "end": 3.2, "words": [...]}
  ]
}
```

### 4. Validation & Error Matrix

| Condition | Result |
|-----------|--------|
| `transcript_data` is `None` | Episode not yet transcribed (normal for PENDING/DOWNLOADED) |
| `transcript_data` set, status not DONE | Inconsistent — worker always sets both atomically |
| JSONB too large (>1 GB column limit) | Not reachable for podcasts (2h episode = ~200 KB segments) |

### 5. Good/Base/Bad Cases

- **Good**: Single UPDATE sets both `transcript_data` and `transcript_status = 'done'` in one commit
- **Base**: Episode with `transcript_data = NULL` — no transcription yet, indexing module skips it
- **Bad**: Episode with `transcript_data` set but indexing hasn't run — no embedding, retrieval misses it

### 6. Tests Required

- Model import: `Episode.transcript_data` exists as `Mapped[dict | None]`
- Migration: column present in `episode` table after `alembic upgrade head`
- Round-trip: write dict → commit → re-read → same dict

### 7. Wrong vs Correct

#### Wrong
```python
# Separate table for transcript segments — overkill when data is always
# read/written as one unit attached to a single episode
class TranscriptSegment(Base):
    episode_id = FK("episode.id")
    text: str
    start: float
    end: float
```

#### Correct
```python
# JSONB on the parent row — simpler, no JOIN, indexing module reads the blob
episode.transcript_data = {"model": "small", "segments": [...]}
await session.commit()
```

---

## Scenario: Atomic Claiming via Status Guard

### 1. Scope / Trigger
- Trigger: Worker polls for DOWNLOADED episodes and must never double-process.
  Two concurrent workers (or a crashed-and-restarted worker) must not claim the same episode.

### 2. Signatures

```python
# transcription/worker.py
async def claim_episode(session: AsyncSession, episode_id: int) -> Episode | None:
    stmt = (
        update(Episode)
        .where(Episode.id == episode_id, Episode.transcript_status == TranscriptStatus.DOWNLOADED)
        .values(transcript_status=TranscriptStatus.PROCESSING)
        .returning(Episode)
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.scalar_one_or_none()
```

### 3. Contracts

- Input: `episode_id` (int), the episode to claim
- Output: `Episode` if successfully claimed, `None` if already claimed by another worker
- Guard: `WHERE transcript_status = 'downloaded'` — the only row-level lock needed
- Commit: immediately after UPDATE so the claim is visible to other workers

### 4. Validation & Error Matrix

| Condition | Result |
|-----------|--------|
| Episode is DOWNLOADED, one worker claims | Returns Episode; status → PROCESSING |
| Episode is DOWNLOADED, two workers race | One returns Episode; other returns None |
| Episode is already PROCESSING | Returns None (status guard rejects) |
| Episode is DONE or FAILED | Returns None |

### 5. Good/Base/Bad Cases

- **Good**: Single worker claims → transcribes → marks DONE. No contention.
- **Base**: No DOWNLOADED episodes → `poll_downloaded()` returns None → worker sleeps.
- **Bad**: Worker crashes mid-transcription → episode stuck in PROCESSING. Mitigated by `reset_stale_processing()` on next startup.

### 6. Tests Required

- Claim a DOWNLOADED episode → returns Episode with status PROCESSING
- Claim the same episode again → returns None
- Claim a non-DOWNLOADED episode → returns None

### 7. Wrong vs Correct

#### Wrong
```python
# SELECT then UPDATE — race condition window between the two statements
episode = await session.get(Episode, episode_id)
if episode.transcript_status == TranscriptStatus.DOWNLOADED:
    episode.transcript_status = TranscriptStatus.PROCESSING
    await session.commit()
```

#### Correct
```python
# Single atomic UPDATE with status guard — no race window
stmt = (
    update(Episode)
    .where(Episode.id == episode_id, Episode.transcript_status == TranscriptStatus.DOWNLOADED)
    .values(transcript_status=TranscriptStatus.PROCESSING)
    .returning(Episode)
)
```

---

## Scenario: Stale State Recovery on Worker Startup

### 1. Scope / Trigger
- Trigger: If a worker crashes mid-transcription, episodes are left with
  `transcript_status = 'processing'`. They must be reset so the restarted
  worker picks them up.

### 2. Signatures

```python
async def reset_stale_processing(session: AsyncSession) -> int:
    stmt = (
        update(Episode)
        .where(Episode.transcript_status == TranscriptStatus.PROCESSING)
        .values(transcript_status=TranscriptStatus.DOWNLOADED)
    )
    result = await session.execute(stmt)
    await session.commit()
    return result.rowcount
```

### 3. Contracts

- Called once at worker startup, before the poll loop
- Resets ALL PROCESSING episodes (single-worker assumption — if concurrent workers exist, only the first to start should reset)
- Returns count of reset episodes for logging

### 4. Validation & Error Matrix

| Condition | Result |
|-----------|--------|
| No PROCESSING episodes | Returns 0, worker proceeds to poll loop |
| N PROCESSING episodes | All N reset to DOWNLOADED, worker logs count |
| Worker crashes again mid-reset | Next startup retries — idempotent |

### 5. Good/Base/Bad Cases

- **Good**: Clean shutdown — no PROCESSING episodes, reset is a no-op
- **Base**: Crash left 1 episode PROCESSING — reset recovers it, worker re-transcribes
- **Bad**: Reset runs while another worker is actively transcribing — that worker's claim is silently stolen. Avoid by ensuring only one worker instance.

### 6. Tests Required

- Insert episodes with various statuses → reset → only PROCESSING ones change
- Idempotent: run reset twice → same result

### 7. Wrong vs Correct

#### Wrong
```python
# Don't reset — orphaned episodes are invisible to the worker forever
async def run_worker(transcriber, poll_interval):
    while True:
        episode = await poll_downloaded(session)
        ...
```

#### Correct
```python
# Reset on startup so crashed-in-flight episodes are re-processed
async def run_worker(transcriber, poll_interval):
    async with session_factory() as session:
        count = await reset_stale_processing(session)
        if count:
            logger.info("Reset %d stale PROCESSING episode(s) back to DOWNLOADED", count)
    while True:
        ...
```

---

## Query Patterns

- Use `select(Model).where(...)` — never raw SQL for routine queries
- Use `update(Model).where(...).values(...).returning(Model)` for atomic state transitions
- Commit per-episode in worker loop (not batch) so progress survives crashes

---

## Migrations

- Alembic auto-generation: `alembic revision --autogenerate -m "description"`
- Apply: `alembic upgrade head`
- Rollback: `alembic downgrade -1`
- Migrations are numbered sequentially (`0001_`, `0002_`, `0003_`...) and chain via `down_revision`

---

## Naming Conventions

- Table names: lowercase singular (`episode`, `podcast`, `chunk`)
- Column names: `snake_case`
- Enum types: `PascalCase` classes with lowercase string values (`TranscriptStatus`)
- JSONB columns: suffix `_data` when storing structured blobs (`transcript_data`)

---

## Common Mistakes

- **SELECT-then-UPDATE for state transitions.** Creates a race condition window. Use atomic `UPDATE...WHERE...RETURNING` with a status guard instead.
- **Separate table for 1:1 attachment data.** When data is always read/written as a unit with its parent row, JSONB is simpler and avoids unnecessary JOINs. Use a separate table only when you need to query individual rows of the nested data.
- **Model tables not registered in `Base.metadata`.** A model's table only exists in SQLAlchemy's metadata if the module defining it has been imported — registration is a side effect of import. A worker/CLI that imports `Episode` and `Chunk` but not `Podcast` will fail at flush time with `Foreign key ... could not find table 'podcast'` when inserting a `chunk` (which has `podcast_id → podcast.id`), because `podcast`'s table was never registered. `SELECT`/`UPDATE` on the unregistered target's *own* table can still work, so the failure only surfaces at INSERT flush — far from the missing import. Centralize model imports in `models/__init__.py` (import `Podcast`, `Episode`, `Chunk` after defining `Base`) so every entry point — API, workers, CLIs, alembic — gets the full table set regardless of import path. The one-line import lives at the bottom of `__init__.py`, *after* `Base` is defined, so the models' `from . import Base` resolves without a circular import.