# Ingestion Module — Implementation Plan

> High-level design for podcast RSS feed fetching, audio download, and database storage. No implementation code — this is the reasoning layer before writing any code.

---

## Scope

Single end-to-end flow: accept an RSS URL → parse the feed → store podcast + episode metadata in Postgres → download the audio file to local disk. MVP handles **1 episode** with the architecture supporting N episodes (just change `max_episodes`).

---

## 1. Pre-requisite: Schema Changes

### 1.1 Episode table needs new columns

The current `episode` table (from `0001_init.py`) is missing fields that ingestion cannot function without:

| Column | Rationale |
|---|---|
| `guid: str` (unique per podcast) | **Dedup key.** Without it, re-importing the same RSS feed creates duplicate episodes. RSS GUIDs are the podcasting standard for "is this the same episode." The podcast-rag reference project uses this pattern — batch-fetch existing GUIDs, skip already-known ones. |
| `enclosure_url: str` | **Download target.** The audio URL extracted from the RSS feed. Must be stored so: (a) the download step knows what to fetch, (b) retry on failure doesn't require re-parsing the feed, (c) a future re-sync can detect URL changes. |
| `duration_seconds: int | None` | **Estimated runtime.** Feed often includes this (iTunes `<itunes:duration>` tag). Useful for: progress estimation during download, prioritizing short episodes for M1 testing, and the worker knowing if a 3-hour episode fits within GPU memory before starting transcription. Optional — the feed may not provide it. |

### 1.2 TranscriptStatus enum needs a DOWNLOADED state

Current values: `PENDING | PROCESSING | DONE | FAILED`

**Problem:** `PENDING` is ambiguous — does it mean "pending download" or "pending transcription"? The worker polling for transcription jobs has no way to distinguish "audio downloaded and ready" from "just created, nothing on disk yet."

**Add `DOWNLOADED`** between `PENDING` and `PROCESSING`:

```
PENDING        ← episode row created, nothing downloaded yet
  ↓
DOWNLOADED     ← audio file on disk, ready for transcription worker
  ↓
PROCESSING     ← worker is actively transcribing
  ↓
DONE | FAILED  ← terminal states
```

**Why not also add `DOWNLOADING`?** The download happens synchronously within the API request for a single episode (M1). The intermediate state provides no value — the request either completes (→ DOWNLOADED) or fails (→ FAILED). Add `DOWNLOADING` later if downloads become async/background.

### 1.3 Where these changes live

The models are defined in `models/podcast.py`, `models/episode.py`, and `models/chunk.py` — each entity in its own file, with `models/__init__.py` providing the shared `Base` class. Ingestion code imports directly from the relevant module:

```python
from rag_podcast.models.podcast import Podcast
from rag_podcast.models.episode import Episode, TranscriptStatus
```

So the schema changes above mean:
- **Edit** `models/episode.py` — add `guid`, `enclosure_url`, `duration_seconds` columns, add `DOWNLOADED` to the enum
- **New** Alembic migration `0002_episode_ingestion_fields.py` — mirrors those column additions + constraint change as DDL

No new model file is created — ingestion operates on the existing `Podcast` and `Episode` classes directly.

### 1.4 Migration approach

A new Alembic migration (`0002_episode_ingestion_fields.py`) that:
- Adds `guid`, `enclosure_url`, `duration_seconds` to `episode`
- Alters the `transcript_status` check constraint to include `'downloaded'`
- All existing rows (from M0 scaffold) get `guid = ''` (placeholder — they were synthetic test data)

---

## 2. Module Structure

```
ingestion/
├── __init__.py
├── PLAN.md          ← this file
├── parser.py        ← RSS feed parsing (feedparser wrapper)
├── downloader.py    ← streaming audio download with retry
├── service.py       ← orchestrates parse → store → download
└── router.py        ← POST /podcasts endpoint
```

**Why four files, not one?** Each has a single reason to change:
- `parser.py` changes if feed format handling needs adjustment (new enclosure location, different namespace)
- `downloader.py` changes if retry strategy or file naming conventions change
- `service.py` changes if the orchestration order changes (e.g., download before DB insert vs. after)
- `router.py` changes if the API contract changes

---

## 3. Parser (`parser.py`)

### 3.1 Why feedparser

`feedparser` is already in `pyproject.toml`. It handles RSS 2.0, Atom, and iTunes namespace extensions out of the box — no XML parsing to write, no namespace handling to debug. The podcast-rag reference project uses the same library successfully.

### 3.2 What gets extracted

**Podcast-level:**
- `title` → `podcast.name`
- `author` (feed-level, or `itunes:author`) → `podcast.author`
- `image` (from `itunes:image` href, or feed `<image>` element) → `podcast.cover_url`

**Episode-level (for each `<item>` in the feed):**
- `guid` → `episode.guid`
- `title` → `episode.title`
- `published` date (parsed from RFC 2822) → `episode.published_date`
- Audio enclosure URL → `episode.enclosure_url`
- `itunes:duration` → `episode.duration_seconds`

### 3.3 Enclosure extraction — why it checks multiple locations

RSS feeds are inconsistent. The audio URL can appear in:
1. `<enclosure url="..." type="audio/mpeg"/>` — standard RSS 2.0 enclosure (most common)
2. `<media:content url="..." type="audio/mpeg"/>` — Media RSS extension
3. `<link rel="enclosure" href="..." type="audio/mpeg"/>` — Atom-style enclosure link

The podcast-rag project's `_extract_enclosure()` checks all three. We follow the same pattern — if the first location is empty, fall through to the next.

### 3.4 Audio type detection

Some feeds omit the MIME type or use generic `application/octet-stream`. Fallback: inspect file extension in the URL path (`.mp3`, `.m4a`, `.mp4`, `.ogg`, `.opus`, `.wav`, `.aac`). Entries without a recognizable audio URL are skipped — not all RSS items are episodes (some feeds include blog posts, ads, bonus content without audio).

### 3.5 Episode limit

`max_episodes` parameter (default 1 for M1). Episodes are processed in feed order (usually reverse chronological — newest first). The parser extracts all episodes, but only the first N are stored + downloaded. **Why extract all then truncate?** The feed is already downloaded and parsed — it's the truncation that's cheap. And this way, when max_episodes is increased later, no parsing logic changes.

---

## 4. Downloader (`downloader.py`)

### 4.1 Sync or async?

**Sync (requests + streaming) for M1.** Rationale:
- Single episode download in M1 — concurrency provides zero benefit
- `requests` streaming is simpler to debug than `aiohttp` + `asyncio.Semaphore`
- The download happens during the POST /podcasts request — user is waiting anyway
- Can switch to `ThreadPoolExecutor` later for batch downloads without changing the interface

### 4.2 Retry strategy

Uses `urllib3.Retry` mounted on a `requests.Session`:
- 3 attempts with exponential backoff (1s, 2s, 4s)
- Retries on: 429 (rate limit), 500, 502, 503, 504
- Does NOT retry on 4xx (except 429) — a 403 or 404 is a permanent failure, retrying wastes time

**Why retry at the HTTP layer, not at the orchestration layer?** A transient network blip during a 90-minute download should not fail the entire operation. The `urllib3.Retry` adapter handles this transparently — the stream resumes from where it dropped (if the server supports range requests) or retries from scratch without the service layer even knowing.

### 4.3 File storage layout

```
DATA_DIR/
└── {podcast_id}/
    └── {episode_id}.{ext}
```

**Why `podcast_id` not podcast name?** IDs are stable; names change (rebranding, typos fixed in feed). `podcast_id` also avoids filesystem sanitization edge cases (Unicode, emoji, long names). The directory is created on first download — no pre-creation step needed.

**Why `episode_id.{ext}` not the episode title?** Same reason — IDs are stable, titles change. Also, the `audio_local_path` column in the DB is the authoritative reference — the user never browses these files directly.

### 4.4 File extension detection

1. From the enclosure URL path (e.g., `.../episode.mp3` → `.mp3`)
2. From the MIME type (e.g., `audio/mp4` → `.m4a`)
3. Default fallback: `.mp3` (most common podcast format)

### 4.5 Post-download

- Update `episode.audio_local_path` to the absolute path
- Update `episode.transcript_status` to `DOWNLOADED`
- If download fails: status → `FAILED`, clean up partial file

---

## 5. Service (`service.py`)

### 5.1 Orchestration flow

```
ingest_podcast(rss_url, max_episodes=1):
  1. PARSE    → feedparser extracts podcast + N episodes from URL
  2. UPSERT?  → check if rss_url already exists in DB
                - YES: return existing podcast + "already imported" message
                - NO:  proceed
  3. INSERT   → write podcast row, get podcast_id
  4. DEDUP    → batch-fetch existing GUIDs for this podcast (empty on first import)
  5. INSERT   → write up to max_episodes episode rows (skip GUIDs already known)
  6. DOWNLOAD → for each new episode, download audio → update path + status
  7. RETURN   → podcast metadata + list of created episodes + download status
```

### 5.2 Why check for existing podcast by rss_url?

The `podcast.rss_url` column has a UNIQUE constraint. Without the pre-check, inserting a duplicate URL would raise an `IntegrityError` — which is cryptic to the API consumer. The explicit check gives a clear response: "This podcast is already imported."

### 5.3 Why dedup by GUID?

Podcast feeds are re-fetched over time (new episodes, metadata updates). Without GUID dedup, every re-import would create duplicate episode rows. GUIDs are the podcast standard — every major podcast app uses them to track "have I seen this episode before."

### 5.4 Error handling strategy

Failures are scoped so one bad episode doesn't block the rest:

| Step | Failure behavior |
|---|---|
| Feed fetch | Abort entire ingestion — can't proceed without feed data |
| Feed parse | Abort — malformed feed, nothing to store |
| Podcast insert | Abort — can't store episodes without a podcast row |
| Episode insert (individual) | Skip that episode, log, continue |
| Audio download (individual) | Set episode status to FAILED, continue to next episode |

---

## 6. API (`router.py`)

### 6.1 Endpoint

```
POST /podcasts
Body: { "rss_url": "https://...", "max_episodes": 1 }
Response: {
  "podcast": { "id": 1, "name": "...", "author": "...", "cover_url": "..." },
  "episodes": [
    { "id": 1, "title": "...", "transcript_status": "downloaded" },
    ...
  ],
  "new_episodes": 1,
  "skipped": 0
}
```

### 6.2 Why POST is synchronous (not a background task)

For 1 episode (M1), the download takes seconds to a few minutes — acceptable within an HTTP request. The user waits once during import and gets immediate feedback (success/failure). For multi-episode imports later, this can become a background task that returns immediately with a "import in progress" status — but that's a future enhancement, not MVP complexity.

### 6.3 Why max_episodes is in the request body, not hardcoded

M1 uses `max_episodes=1`, M3 uses `max_episodes=20`. The API caller (frontend) controls this — the frontend can offer "Import latest episode" vs. "Import last 10 episodes" without a backend code change.

---

## 7. Connection to the Rest of the Pipeline

The ingestion module's output is:
- Podcast + episode rows in Postgres
- Audio files on disk at `DATA_DIR/{podcast_id}/{episode_id}.ext`
- Episodes at status `DOWNLOADED`

The transcription worker (future module) polls for episodes where `transcript_status = 'downloaded'`, transcribes them, and advances the status to `PROCESSING` → `DONE`. Ingestion doesn't need to know anything about transcription — it just leaves episodes in the `DOWNLOADED` state.

```
┌──────────┐     ┌───────────────┐     ┌──────────┐
│ Ingestion │ ──→ │ DOWNLOADED    │ ←── │ Worker   │
│ (this     │     │ (status in    │     │ (polls   │
│  module)  │     │  DB, audio    │     │  for this│
│           │     │  on disk)     │     │  status) │
└──────────┘     └───────────────┘     └──────────┘
```

---

## 8. What This Plan Intentionally Leaves Out

- **Concurrent downloads** — M1 is 1 episode; ThreadPoolExecutor is future work
- **Feed re-sync** (detecting new episodes since last import) — the podcast-rag project has full `sync_podcast` logic for this. Not needed for M1's "import once" flow
- **Progress callbacks** — single episode doesn't need progress reporting
- **Audio cleanup** (deleting files after transcription is done) — belongs to the worker, not ingestion
- **OPML import** (batch-importing multiple podcasts) — out of scope for MVP
- **iTunes Search API lookup** (finding a feed URL from a podcast name) — user pastes the RSS URL directly
