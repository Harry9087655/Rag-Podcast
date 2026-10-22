"""Live and replay adapters projecting QA answers into de-identified observations.

``LiveAdapter`` drives ``answer()`` case by case and projects each result onto
the evaluation ``Observation`` whitelist; ``PredictionFileAdapter`` replays a
previously saved bundle for offline scoring. Both return an ``ObservationBundle``
whose provenance is preserved. Production imports (QA/embedding/retrieval) are
lazy so the pure scoring path never pulls in the DB/SDK stack at import time.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections import Counter
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, Literal

from .contracts import (
    ContractError,
    Dataset,
    DatasetFingerprintMismatch,
    Observation,
    ObservationBundle,
    Source,
)
from .loaders import load_observations

Arm = Literal["full-context", "search"]
AnswerFn = Callable[..., Awaitable[Any]]
SessionFactory = Callable[..., Any]  # returns an async context manager yielding a session


def _require_k(k: int) -> None:
    if type(k) is not int or k <= 0:
        raise ContractError("k must be a positive integer")


def _require_case_timeout(case_timeout: float | None) -> None:
    if case_timeout is None:
        return
    if (
        type(case_timeout) not in (int, float)
        or not math.isfinite(case_timeout)
        or case_timeout <= 0
    ):
        raise ContractError("case_timeout must be a finite positive number")


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def _project_source(ref: Any) -> Source:
    """Whitelist-projection of a QA source onto the evaluation ``Source``.

    Drops ``podcast_id``/``podcast_name``/``episode_title`` and any provider
    payload; keeps identity, exact time span, text and parent linkage, in
    original order.
    """
    return Source(
        chunk_id=ref.chunk_id,
        start=ref.start,
        end=ref.end,
        episode_id=ref.episode_id,
        text=ref.text,
        level=ref.level,
        child_chunk_ids=list(ref.child_chunk_ids),
    )


class LiveAdapter:
    """Drive ``answer()`` per case and project to de-identified observations.

    ``answer_fn`` and ``session_factory`` are injected; the adapter itself stays
    free of production imports. Each case opens its own read-only session and
    never commits. Provider failures and post-routing timeouts become error
    observations and continue; pre-routing, infrastructure and programming
    failures propagate.
    """

    def __init__(self, answer_fn: AnswerFn, session_factory: SessionFactory):
        self._answer_fn = answer_fn
        self._session_factory = session_factory

    async def execute(
        self,
        dataset: Dataset,
        dataset_fingerprint: str,
        *,
        k: int,
        case_timeout: float | None = None,
    ) -> ObservationBundle:
        _require_k(k)
        _require_case_timeout(case_timeout)

        # Lazy imports: only the live path needs the provider error types.
        from rag_podcast.embedding.embedder import EmbeddingError
        from rag_podcast.qa.core import QAError
        from rag_podcast.retrieval.BaseRecallerInterface import RetrievalError

        provider_errors = (QAError, EmbeddingError, RetrievalError)

        observations: list[Observation] = []
        for case in dataset.cases:
            observations.append(
                await self._execute_case(case, k, case_timeout, provider_errors)
            )
        return ObservationBundle(
            dataset_fingerprint=dataset_fingerprint,
            provenance="live",
            observations=observations,
        )

    async def _execute_case(
        self,
        case: Any,
        k: int,
        case_timeout: float | None,
        provider_errors: tuple[type[Exception], ...],
    ) -> Observation:
        arm: Arm | None = None

        def observe(observed: Arm) -> None:
            nonlocal arm
            if arm is not None:
                raise ContractError(f"{case.id}: answer emitted the arm more than once")
            arm = observed

        start = time.monotonic()
        try:
            async with self._session_factory() as session:
                coro = self._answer_fn(
                    case.question,
                    session=session,
                    episode_id=case.scope.episode_id,
                    podcast_id=case.scope.podcast_id,
                    history=None,
                    top_k=k,
                    on_arm=observe,
                )
                if case_timeout is not None:
                    async with asyncio.timeout(case_timeout):
                        result = await coro
                else:
                    result = await coro
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            if arm is None:
                raise
            return Observation(
                case_id=case.id, arm=arm, answer_text="", sources=[],
                duration_ms=_elapsed_ms(start), error="case_timeout",
            )
        except provider_errors:
            if arm is None:
                raise
            return Observation(
                case_id=case.id, arm=arm, answer_text="", sources=[],
                duration_ms=_elapsed_ms(start), error="provider_failed",
            )

        if arm is None:
            raise ContractError(f"{case.id}: answer returned without emitting an arm")
        return Observation(
            case_id=case.id,
            arm=arm,
            answer_text=result.answer,
            sources=[_project_source(s) for s in result.sources],
            duration_ms=_elapsed_ms(start),
            error=None,
        )


class PredictionFileAdapter:
    """Replay a saved observation bundle for offline scoring.

    Reads ``Observations`` via ``load_observations``, validates the dataset
    fingerprint and observation identities, then returns the bundle unchanged so
    its original provenance (``live`` vs ``contract_smoke``) is preserved.
    Missing observations are allowed and become failed cases downstream.
    """

    def __init__(self, path: Path):
        self._path = path

    def execute(self, dataset: Dataset, dataset_fingerprint: str) -> ObservationBundle:
        bundle = load_observations(self._path)
        if bundle.dataset_fingerprint != dataset_fingerprint:
            raise DatasetFingerprintMismatch("observation and dataset fingerprints differ")

        case_ids = {case.id for case in dataset.cases}
        counts = Counter(obs.case_id for obs in bundle.observations)
        duplicates = sorted(case_id for case_id, count in counts.items() if count > 1)
        if duplicates:
            raise ContractError(f"Duplicate observation case_ids: {duplicates}")
        unknown = sorted(counts.keys() - case_ids)
        if unknown:
            raise ContractError(f"Unknown observation case_ids: {unknown}")
        return bundle
