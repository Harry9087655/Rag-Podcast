"""Single-turn QA core: the one entry point ``api``/``cli`` call.

``answer()`` is the seam.  The scope is chosen explicitly by the caller —
``episode_id`` (single episode, agentic full-context eligible), ``podcast_id``
alone (whole podcast, search only), or neither (whole library, search only).
The full-context arm feeds the cleaned transcript to the LLM and reverse-looks
up cited passages via the ``find_timestamp`` tool; the search arm numbers
retrieved chunks and maps ``[n]`` markers back to them.  Both arms return the
same ``{answer, sources[]}`` contract.

Design contracts: .trellis/tasks/09-16-qa-core/design.md,
.trellis/tasks/09-24-qa-agentic-citation/design.md
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from dataclasses import dataclass

from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..cleaning.filler_words import clean_words
from ..config import settings
from ..models.episode import Episode
from ..models.podcast import Podcast
from ..retrieval.BaseRecallerInterface import SearchResult
from ..retrieval.search import search
from .prompt import (
    AGENTIC_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_context_block,
    build_transcript_context,
    count_tokens,
)

logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(\d+)\]")

_FIND_TIMESTAMP_TOOL = {
    "type": "function",
    "function": {
        "name": "find_timestamp",
        "description": (
            "Return the exact start/end (seconds) of a passage. Pass the passage "
            "text verbatim — copy it exactly, no ellipsis or abbreviation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fragment": {
                    "type": "string",
                    "minLength": 1,
                    "description": (
                        "The passage copied verbatim from the transcript (a complete "
                        "sentence or paragraph). Copy exactly — no ellipsis, "
                        "abbreviation, or rewording; skip the citation rather than "
                        "truncate it."
                    ),
                }
            },
            "required": ["fragment"],
        },
    },
}


@dataclass
class SourceRef:
    """One source chunk (or full-context segment), self-contained for the
    ``api``/``frontend`` consumer.

    Deliberately omits ``audio_url`` — the api layer resolves ``episode_id →
    URL`` (parent R3).  A strict superset of the retrieval ``SearchResult``
    fields minus ``score``.  ``chunk_id`` is ``None`` for full-context segment
    sources; search chunks carry a real id.
    """

    text: str
    start: float
    end: float
    podcast_id: int
    podcast_name: str
    episode_id: int
    episode_title: str
    chunk_id: int | None = None


@dataclass
class TranscriptSegment:
    """One sentence-level transcript segment: display text plus its time span."""

    text: str
    start: float
    end: float


@dataclass
class EpisodeContext:
    """One episode's cleaned full transcript plus its identifying metadata."""

    episode_id: int
    episode_title: str
    podcast_id: int
    podcast_name: str
    language: str | None
    full_transcript: list[TranscriptSegment]


@dataclass
class AnswerResult:
    """The qa-core output contract (parent data contract ``{answer, sources[]}``)."""

    answer: str
    sources: list[SourceRef]


class QAError(Exception):
    """Raised when the LLM call fails (no retry, mirroring ``embed_batch``)."""


def _input_budget() -> int:
    """Tokens available for the *input*: window minus the answer reserve."""
    return settings.llm_max_context_tokens - settings.llm_response_reserve_tokens


def _to_source_refs(results: list[SearchResult]) -> list[SourceRef]:
    """Adapt retrieval ``SearchResult`` → qa ``SourceRef`` (drops ``score``)."""
    return [
        SourceRef(
            chunk_id=r.chunk_id,
            text=r.text,
            start=r.start,
            end=r.end,
            podcast_id=r.podcast_id,
            podcast_name=r.podcast_name,
            episode_id=r.episode_id,
            episode_title=r.episode_title,
        )
        for r in results
    ]


def _build_messages(
    question: str,
    history: list[dict[str, str]],
    context_block: str,
    *,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, str]]:
    """Assemble the shared prompt skeleton: system → history → context+question.

    ``system_prompt`` defaults to ``SYSTEM_PROMPT`` (search arm); the agentic arm
    passes ``AGENTIC_SYSTEM_PROMPT``.
    """
    return [
        {"role": "system", "content": system_prompt},
        *history,
        {"role": "user", "content": f"{context_block}\n\nquestion: {question}"},
    ]


def _display_segments(
    raw: list[dict], language: str | None
) -> list[TranscriptSegment]:
    """Clean raw ASR segments into display text (filler words removed).

    Reuses ``cleaning.filler_words.clean_words`` — the same source the indexing
    uses — so the full-context arm and the search arm feed the LLM identical
    text.  ``start``/``end`` pass through unchanged.
    """
    cleaned = clean_words(raw, language)
    return [
        TranscriptSegment(text=s["text"], start=s["start"], end=s["end"])
        for s in cleaned
    ]


async def _load_episode_context(
    session: AsyncSession, episode_id: int
) -> EpisodeContext | None:
    """Load one episode's cleaned full transcript, or ``None`` if absent/empty.

    Reads ``Episode.transcript_data["segments"]`` (raw sentence-level ASR) and
    cleans it via ``_display_segments``, dropping filler words and empty
    segments.
    """
    row = (
        await session.execute(
            select(Episode, Podcast.name)
            .join(Podcast, Podcast.id == Episode.podcast_id)
            .where(Episode.id == episode_id)
        )
    ).first()
    if row is None:
        return None
    episode, podcast_name = row
    raw_segments = (episode.transcript_data or {}).get("segments", [])
    if not raw_segments:
        return None
    full_transcript = _display_segments(raw_segments, episode.language)
    if not full_transcript:
        return None
    return EpisodeContext(
        episode_id=episode.id,
        episode_title=episode.title,
        podcast_id=episode.podcast_id,
        podcast_name=podcast_name,
        language=episode.language,
        full_transcript=full_transcript,
    )


def _normalize(text: str) -> str:
    """Fold ``text`` to a comparison form: NFKC (full→half width) + lower, then
    drop whitespace and punctuation.  Fragment and transcript text are both
    normalized before matching, so spacing, punctuation, case, and full/half
    width differences never break a verbatim hit."""
    folded = unicodedata.normalize("NFKC", text).lower()
    return "".join(
        ch for ch in folded
        if not ch.isspace() and unicodedata.category(ch)[0] != "P"
    )


def _find_timestamp(fragment: str, full_transcript: list[TranscriptSegment]) -> dict:
    """Reverse-lookup a verbatim ``fragment`` → its stored ``{start, end}``.

    Matches against the cleaned display text of ``full_transcript``, so the
    returned ``start``/``end`` are stored ground-truth (never recomputed).
    Exact normalized substring match first, then a fuzzy ``difflib`` fallback;
    ``{"found": False}`` when empty, unmatchable, or below the fuzzy threshold.
    """
    if not isinstance(fragment, str):
        return {"found": False}
    norm_frag = _normalize(fragment)
    if not norm_frag:
        return {"found": False}

    norm_texts = [_normalize(s.text) for s in full_transcript]
    joined = "".join(norm_texts)
    if not joined:
        return {"found": False}

    pos = joined.find(norm_frag)
    if pos < 0:
        match = difflib.SequenceMatcher(
            None, joined, norm_frag, autojunk=False
        ).find_longest_match(0, len(joined), 0, len(norm_frag))
        if match.size / len(norm_frag) < 0.7:
            return {"found": False}
        pos, frag_len = match.a, match.size
    else:
        frag_len = len(norm_frag)

    # Map the [pos, pos+frag_len) span in the joined text back to the segments
    # it covers: start of the first, end of the last.
    first = last = None
    offset = 0
    for i, nt in enumerate(norm_texts):
        seg_start, seg_end = offset, offset + len(nt)
        offset = seg_end
        if seg_end <= pos or seg_start >= pos + frag_len:
            continue
        if first is None:
            first = i
        last = i
    if first is None:
        return {"found": False}

    segs = full_transcript[first : last + 1]
    return {
        "found": True,
        "text": " ".join(s.text for s in segs),
        "start": segs[0].start,
        "end": segs[-1].end,
    }


def _safe_parse_args(raw: str) -> dict:
    """Parse a tool-call ``arguments`` JSON string; ``{}`` on any failure.

    ``arguments`` arrives as a JSON string; a malformed or non-object payload
    falls through to ``{}`` so ``args.get("fragment", "")`` → ``""`` → a safe
    ``{"found": False}`` instead of a crash.
    """
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def _chat_agentic(
    messages: list[dict[str, str]], full_transcript: list[TranscriptSegment]
) -> tuple[str, list[dict]]:
    """Bounded ``find_timestamp`` tool loop: initial call + at most one tool round.

    Returns ``(answer_text, results)`` — ``results`` are the ordered
    ``find_timestamp`` returns (found or not), i.e. the ``[n]`` citation index.
    A response with no ``tool_calls`` ends the loop and returns its content.
    """
    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    try:
        results: list[dict] = []
        msgs = list(messages)
        rounds = 0
        while rounds < 2:
            rounds += 1
            try:
                resp = await client.chat.completions.create(
                    model=settings.llm_model,
                    messages=msgs,
                    tools=[_FIND_TIMESTAMP_TOOL],
                    tool_choice="auto",
                )
            except Exception as exc:
                raise QAError(f"LLM chat.completions request failed: {exc}") from exc

            msg = resp.choices[0].message
            if not msg.tool_calls:
                return msg.content or "", results

            msgs.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ],
                }
            )
            for tc in msg.tool_calls:
                # Beyond the citation cap, still answer the call (so the message
                # list stays consistent) but drop it from ``results``: the model
                # sees found:False and, per prompt rule 6, will not cite it.
                if len(results) >= settings.qa_max_citations:
                    result = {"found": False}
                else:
                    args = _safe_parse_args(tc.function.arguments)
                    result = _find_timestamp(args.get("fragment", ""), full_transcript)
                    results.append(result)
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
        raise QAError("find_timestamp tool loop exceeded bounds")
    finally:
        await client.close()


def _sources_from_results(
    results: list[dict], ctx: EpisodeContext
) -> list[SourceRef]:
    """Assemble ``sources[]`` from tool results: found passages in call order.

    ``chunk_id`` is ``None`` — full-context segment sources have no chunk; the
    episode/podcast identity comes from ``ctx``.
    """
    return [
        SourceRef(
            chunk_id=None,
            text=r["text"],
            start=r["start"],
            end=r["end"],
            podcast_id=ctx.podcast_id,
            podcast_name=ctx.podcast_name,
            episode_id=ctx.episode_id,
            episode_title=ctx.episode_title,
        )
        for r in results
        if r.get("found")
    ]


def _parse_citations(answer: str, context_chunks: list[SourceRef]) -> list[SourceRef]:
    """Map inline ``[n]`` markers to chunks, deduped in first-appearance order."""
    sources: list[SourceRef] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(answer):
        n = int(match.group(1))
        if 1 <= n <= len(context_chunks):
            chunk = context_chunks[n - 1]
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                sources.append(chunk)
    return sources


async def _chat(messages: list[dict[str, str]]) -> str:
    """One LLM ``chat.completions`` call, mirroring ``embed_batch``'s lifecycle."""
    client = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
    )
    try:
        try:
            resp = await client.chat.completions.create(
                model=settings.llm_model, messages=messages
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:
            raise QAError(f"LLM chat.completions request failed: {exc}") from exc
    finally:
        await client.close()


async def _get_context(
    session: AsyncSession,
    *,
    episode_id: int | None,
    podcast_id: int | None,
    top_k: int | None,
    question: str,
) -> list[SourceRef]:
    """Resolve search-arm context chunks for a question.

    Every scope here routes to ``search()``; the single-episode agentic
    full-context arm is handled by ``answer()`` before this is reached.
    """
    if episode_id is not None:
        results = await search(
            question, session=session, top_k=top_k, episode_id=episode_id
        )
        return _to_source_refs(results)

    if podcast_id is not None:
        results = await search(
            question, session=session, top_k=top_k, podcast_id=podcast_id
        )
        return _to_source_refs(results)

    results = await search(question, session=session, top_k=top_k)
    return _to_source_refs(results)


async def answer(
    question: str,
    *,
    session: AsyncSession,
    episode_id: int | None = None,
    podcast_id: int | None = None,
    history: list[dict[str, str]] | None = None,
    top_k: int | None = None,
) -> AnswerResult:
    """Answer ``question`` from the podcast corpus and return the cited sources.

    Scope routing: ``episode_id`` → single episode (agentic full-context when
    the cleaned transcript fits the input budget, else in-episode search);
    ``podcast_id`` alone → cross-episode search; neither → whole-library
    search.  An empty context short-circuits to ``AnswerResult("", [])`` with
    no LLM call.
    """
    if history is None:
        history = []

    # Single-episode scope: agentic full-context when a transcript exists and
    # fits the budget; otherwise fall through to the (unchanged) search arm.
    if episode_id is not None:
        ctx = await _load_episode_context(session, episode_id)
        if ctx is not None:
            messages = _build_messages(
                question,
                history,
                build_transcript_context(ctx.full_transcript),
                system_prompt=AGENTIC_SYSTEM_PROMPT,
            )
            if count_tokens(messages) <= _input_budget():
                logger.info(
                    "qa arm=full-context segments=%d", len(ctx.full_transcript)
                )
                answer_text, results = await _chat_agentic(
                    messages, ctx.full_transcript
                )
                return AnswerResult(
                    answer=answer_text,
                    sources=_sources_from_results(results, ctx),
                )

    context_chunks = await _get_context(
        session,
        episode_id=episode_id,
        podcast_id=podcast_id,
        top_k=top_k,
        question=question,
    )
    logger.info("qa arm=search context_chunks=%d", len(context_chunks))

    if not context_chunks:
        return AnswerResult("", [])

    messages = _build_messages(question, history, build_context_block(context_chunks))
    answer_text = await _chat(messages)
    sources = _parse_citations(answer_text, context_chunks)
    return AnswerResult(answer=answer_text, sources=sources)
