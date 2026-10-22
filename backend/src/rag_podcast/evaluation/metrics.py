"""Eval scoring: search + full-context tracks, per-case scoring, aggregation.

The search track compares retrieved chunks against ``gold_chunks`` (leaf ids).
Because ``auto_merge`` can replace sibling leaves with their parent block, a
retrieved item is either a leaf (its own ``chunk_id``) or a merged parent
(``child_chunk_ids`` = the leaves it absorbed).  Scoring happens at leaf
granularity for recall/coverage: a parent expands to its member leaves before
comparison, so gold leaf ids match regardless of the merge.
"""

from __future__ import annotations

from collections import defaultdict
import math
from .contracts import (
    Case,
    CasePassPolicy,
    CaseResult,
    GoldSegment,
    MetricValue,
    Observation,
    Source,
)
from .case_policy import evaluate_case_policy


def _leaf_set(source: Source) -> set[int]:
    """Leaf chunk ids a retrieved ``Source`` represents.

    ``auto_merge`` can replace sibling leaves with their parent block, so a
    retrieved item is either a leaf (its own ``chunk_id``) or a merged parent
    (``child_chunk_ids`` = the leaves it absorbed).  Scoring happens at leaf
    granularity: a parent expands to its member leaves, so gold leaf ids match
    regardless of the merge.  ``chunk_id`` is ``None`` only for full-context
    segment sources, which never reach the search track.
    """
    if source.level == "parent" and source.child_chunk_ids:
        return set(source.child_chunk_ids)
    if source.chunk_id is None:
        return set()
    return {source.chunk_id}


def _covered(retrieved: list[Source], k: int | None = None) -> set[int]:
    """Union of leaf ids covered by the first ``k`` retrieved items (all if ``None``)."""
    items = retrieved if k is None else retrieved[:k]
    covered: set[int] = set()
    for source in items:
        if source.level == "parent" and source.child_chunk_ids:
            covered |= set(source.child_chunk_ids)
            continue
        if source.chunk_id is None:
            continue
        covered.add(source.chunk_id)
    return covered


# ---- search track: lists of chunk IDs ----
# recall/coverage are leaf-granular: a merged parent covers every member leaf.
# precision/mrr/ndcg are item-granular: a merged parent is one item, relevant
# iff it covers at least one gold leaf.
def recall_at_k(retrieved: list[Source], gold: set[int], k: int) -> float:
    if not gold:
        return 0.0
    return len(_covered(retrieved, k) & gold) / len(gold)


def precision_at_k(retrieved: list[Source], gold: set[int], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant = sum(1 for s in top if _leaf_set(s) & gold)
    return relevant / len(top)


def mrr_at_k(retrieved: list[Source], gold: set[int], k: int) -> float:
    for rank, source in enumerate(retrieved[:k], start=1):
        if _leaf_set(source) & gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[Source], gold: set[int], k: int) -> float:
    """Binary item-level DCG; idcg places one ideal item per gold leaf."""
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, source in enumerate(retrieved[:k], start=1)
        if _leaf_set(source) & gold
    )
    ideal_count = min(k, len(gold))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg else 0.0


def gold_chunk_coverage(retrieved: list[Source], gold: set[int]) -> float:
    if not gold:
        return 0.0
    return len(_covered(retrieved) & gold) / len(gold)


# ---- full-context track: time ranges ----
def _covers(source: Source, seg: GoldSegment) -> bool:
    return source.start <= seg.start and source.end >= seg.end


def citation_hit(sources: list[Source], gold: list[GoldSegment]) -> float:
    if not gold:
        return 0.0

    return sum(1 for gs in gold if any(_covers(s, gs) for s in sources)) / len(gold)


def citation_precision(sources: list[Source], gold: list[GoldSegment]) -> float:
    if not sources:
        return 0.0
    return sum(1 for s in sources if any(_covers(s, gs) for gs in gold)) / len(sources)


# ---- dispatch ----
def score_case(case: Case, obs: Observation, k: int = 5) -> dict[str, float]:
    if case.gold_chunks is not None:               # search 臂
        gold = set(case.gold_chunks)
        return {
            f"recall@{k}": recall_at_k(obs.sources, gold, k),
            f"precision@{k}": precision_at_k(obs.sources, gold, k),
            f"mrr@{k}": mrr_at_k(obs.sources, gold, k),
            f"ndcg@{k}": ndcg_at_k(obs.sources, gold, k),
            "gold_chunk_coverage": gold_chunk_coverage(obs.sources, gold),
        }
    if case.gold_segments is not None:             # full-context 臂
        return {
            "citation_hit": citation_hit(obs.sources, case.gold_segments),
            "citation_precision": citation_precision(obs.sources, case.gold_segments),
        }
    return {}   # no_knowledge


def case_passed(
    case: Case,
    obs: Observation | None,
    metrics: dict[str, float],
    policy: CasePassPolicy | None = None,
) -> tuple[bool, dict[str, float | bool | str]]:
    """Apply an injectable per-case policy; defaults preserve the original criteria."""
    return evaluate_case_policy(case, obs, metrics, policy if policy is not None else CasePassPolicy())


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _metrics_for(results: list[CaseResult], metric_keys: list[str]) -> dict[str, MetricValue]:
    n_total = len(results)
    n_observed = sum(1 for r in results if not r.checks.get("missing_observation", False))
    n_provider_failed = sum(1 for r in results if r.checks.get("provider_failed", False))
    passed = sum(1 for r in results if r.passed)

    out: dict[str, MetricValue] = {
        "case_pass_rate": MetricValue(
            value=passed / n_total if n_total else 0.0,
            eligible_cases=n_total,
        ),
        "provider_failure_rate": MetricValue(
            value=n_provider_failed / n_observed if n_observed else 0.0,
            eligible_cases=n_observed,
        ),
    }
    for key in metric_keys:
        values = [r.metrics[key] for r in results if key in r.metrics]
        out[key] = MetricValue(value=_mean(values), eligible_cases=len(values))
    return out


def aggregate(
    cases: list[Case], results: list[CaseResult]
) -> tuple[dict[str, MetricValue], dict[str, dict[str, MetricValue]]]:
    tags_by_case_id = {case.id: case.tags for case in cases}
    metric_keys = sorted({key for result in results for key in result.metrics})

    by_tag: dict[str, list[CaseResult]] = defaultdict(list)
    for result in results:
        for tag in tags_by_case_id[result.case_id]:
            by_tag[tag].append(result)

    metrics = _metrics_for(results, metric_keys)
    slices = {tag: _metrics_for(subset, metric_keys) for tag, subset in by_tag.items()}
    return metrics, slices
