"""Tests for eval scoring: search + full-context tracks, case_passed, aggregate.

The search track compares retrieved sources against ``gold_chunks`` (leaf ids).
A retrieved item may be a merged parent (``level="parent"``) whose
``child_chunk_ids`` are the leaves it absorbed; recall/coverage expand it back
to leaves so gold leaf ids match regardless of the merge.  precision/mrr/ndcg
are item-granular — a parent is one relevant item.
"""

from __future__ import annotations

import math

import pytest

from rag_podcast.evaluation.contracts import (
    Case,
    CasePassPolicy,
    CaseResult,
    ContractError,
    GoldSegment,
    Observation,
    Scope,
    Source,
)
from rag_podcast.evaluation.metrics import (
    aggregate,
    case_passed,
    citation_hit,
    citation_precision,
    gold_chunk_coverage,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def _src(chunk_id: int, *, level: str = "leaf", child: list[int] | None = None) -> Source:
    return Source(
        chunk_id=chunk_id,
        start=0.0,
        end=1.0,
        episode_id=1,
        text=f"text-{chunk_id}",
        level=level,
        child_chunk_ids=child or [],
    )


def _parent(chunk_id: int, children: list[int]) -> Source:
    return _src(chunk_id, level="parent", child=children)


def test_parent_expands_to_member_leaves():
    gold = {512, 513}
    retrieved = [_parent(100, [512, 513]), _src(600), _src(601)]

    assert recall_at_k(retrieved, gold, 5) == 1.0
    assert gold_chunk_coverage(retrieved, gold) == 1.0


def test_leaf_matches_own_id():
    gold = {404}
    assert recall_at_k([_src(404), _src(600)], gold, 5) == 1.0


def test_miss_returns_zero():
    gold = {512, 513}
    retrieved = [_src(600), _parent(100, [700, 701])]

    assert recall_at_k(retrieved, gold, 5) == 0.0
    assert gold_chunk_coverage(retrieved, gold) == 0.0


def test_parent_covering_partial_gold_gives_fractional_recall():
    gold = {512, 513}
    retrieved = [_parent(100, [512]), _src(600)]

    assert recall_at_k(retrieved, gold, 5) == 0.5


def test_mrr_uses_the_parent_slot_rank():
    gold = {512, 513}
    assert mrr_at_k([_src(600), _parent(100, [512, 513])], gold, 5) == 0.5
    assert mrr_at_k([_parent(100, [512, 513])], gold, 5) == 1.0


def test_precision_counts_relevant_items_not_leaves():
    gold = {512, 513}
    retrieved = [
        _parent(100, [512, 513]),
        _src(600),
        _src(601),
        _src(602),
        _src(603),
    ]

    # One relevant item out of five — the parent is a single item.
    assert precision_at_k(retrieved, gold, 5) == 0.2


def test_ndcg_single_gold_leaf_at_rank_one_is_perfect():
    gold = {512}
    assert ndcg_at_k([_parent(100, [512])], gold, 5) == 1.0


def test_ndcg_two_gold_leaves_merged_into_one_parent():
    # idcg assumes one ideal item per gold leaf (two items); the merged parent
    # is credited as one relevant item at rank 1, so ndcg < 1.
    gold = {512, 513}
    retrieved = [_parent(100, [512, 513]), _src(600), _src(601)]
    idcg = 1.0 + 1.0 / math.log2(3)
    assert ndcg_at_k(retrieved, gold, 5) == pytest.approx(1.0 / idcg)


def test_empty_gold_returns_zero():
    retrieved = [_src(404)]
    assert recall_at_k(retrieved, set(), 5) == 0.0
    assert precision_at_k(retrieved, set(), 5) == 0.0
    assert mrr_at_k(retrieved, set(), 5) == 0.0
    assert ndcg_at_k(retrieved, set(), 5) == 0.0
    assert gold_chunk_coverage(retrieved, set()) == 0.0


# ---- full-context track: time ranges ----


def _seg(start: float, end: float) -> GoldSegment:
    return GoldSegment(start=start, end=end)


def _fc_source(start: float, end: float) -> Source:
    return Source(chunk_id=None, start=start, end=end, episode_id=1, text="t")


def test_citation_hit_source_wider_than_segment():
    gold = [_seg(2.0, 8.0)]
    assert citation_hit([_fc_source(0.0, 10.0)], gold) == 1.0


def test_citation_hit_equal_boundaries_are_covered():
    gold = [_seg(2.0, 8.0)]
    assert citation_hit([_fc_source(2.0, 8.0)], gold) == 1.0


def test_citation_hit_narrower_source_not_covered():
    gold = [_seg(2.0, 8.0)]
    assert citation_hit([_fc_source(3.0, 7.0)], gold) == 0.0


def test_citation_hit_partial_overlap_not_covered():
    gold = [_seg(2.0, 8.0)]
    assert citation_hit([_fc_source(0.0, 5.0)], gold) == 0.0


def test_citation_hit_counts_each_covered_segment():
    gold = [_seg(2.0, 8.0), _seg(10.0, 12.0)]
    assert citation_hit([_fc_source(0.0, 9.0)], gold) == 0.5


def test_citation_precision_counts_relevant_sources():
    gold = [_seg(2.0, 8.0)]
    sources = [_fc_source(0.0, 10.0), _fc_source(100.0, 110.0)]
    assert citation_precision(sources, gold) == 0.5


def test_citation_precision_empty_sources_is_zero():
    assert citation_precision([], [_seg(2.0, 8.0)]) == 0.0


def test_citation_hit_empty_sources_is_zero():
    assert citation_hit([], [_seg(2.0, 8.0)]) == 0.0


def test_citation_empty_gold_is_zero():
    assert citation_hit([_fc_source(0.0, 10.0)], []) == 0.0
    assert citation_precision([_fc_source(0.0, 10.0)], []) == 0.0


# ---- case_passed ----


def _case(id: str, tags: list[str], *, gold_chunks=None, gold_segments=None) -> Case:
    return Case(
        id=id,
        tags=tags,
        critical=False,
        question="q",
        expected_arm="search" if gold_chunks is not None else "full-context",
        scope=Scope(episode_id=1),
        gold_chunks=gold_chunks,
        gold_segments=gold_segments,
        reference_answer="ref",
        required_claims=[],
    )


def _obs(sources: list[Source] | None = None, *, error: str | None = None) -> Observation:
    return Observation(
        case_id="c1", arm="search", answer_text="ans", sources=sources or [], error=error
    )


def test_case_passed_missing_observation():
    case = _case("c1", [], gold_chunks=[1])
    assert case_passed(case, None, {}) == (False, {"missing_observation": True})


def test_case_passed_provider_failed():
    case = _case("c1", [], gold_chunks=[1])
    assert case_passed(case, _obs(error="provider_failed"), {}) == (
        False,
        {"provider_failed": True},
    )


def test_case_passed_search_pass():
    case = _case("c1", [], gold_chunks=[1])
    metrics = {"gold_chunk_coverage": 1.0}
    assert case_passed(case, _obs(), metrics) == (True, {"gold_chunk_coverage": 1.0})


def test_case_passed_search_fail():
    case = _case("c1", [], gold_chunks=[1])
    metrics = {"gold_chunk_coverage": 0.5}
    assert case_passed(case, _obs(), metrics) == (False, {"gold_chunk_coverage": 0.5})


def test_case_passed_full_context_pass():
    case = _case("c1", [], gold_segments=[_seg(2.0, 8.0)])
    assert case_passed(case, _obs(), {"citation_hit": 1.0}) == (True, {"citation_hit": 1.0})


def test_case_passed_full_context_fail():
    case = _case("c1", [], gold_segments=[_seg(2.0, 8.0)])
    assert case_passed(case, _obs(), {"citation_hit": 0.0}) == (False, {"citation_hit": 0.0})


def test_case_passed_no_knowledge_abstained():
    assert case_passed(_case("c1", []), _obs(), {}) == (True, {"abstained": True})


def test_case_passed_no_knowledge_not_abstained():
    obs = _obs(sources=[_src(404)])
    assert case_passed(_case("c1", []), obs, {}) == (False, {"abstained": False})


@pytest.mark.parametrize("precision, passed", [(0.59, False), (0.6, True)])
def test_case_policy_requires_all_configured_metrics(precision, passed):
    policy = CasePassPolicy(search={"gold_chunk_coverage": 1.0, "precision@5": 0.6})
    metrics = {"gold_chunk_coverage": 1.0, "precision@5": precision, "mrr@5": 0.1}
    assert case_passed(_case("c1", [], gold_chunks=[1]), _obs(), metrics, policy) == (
        passed, {"gold_chunk_coverage": 1.0, "precision@5": precision}
    )


def test_case_policy_can_replace_default_and_use_dynamic_rank():
    policy = CasePassPolicy(search={"mrr@10": 0.5, "ndcg@10": 0.7})
    case = _case("c1", [], gold_chunks=[1])
    assert case_passed(case, _obs(), {"mrr@10": 0.5, "ndcg@10": 0.7}, policy)[0]


def test_case_policy_full_context_precision():
    policy = CasePassPolicy(full_context={"citation_hit": 1.0, "citation_precision": 0.8})
    case = _case("c1", [], gold_segments=[_seg(2.0, 8.0)])
    assert not case_passed(case, _obs(), {"citation_hit": 1.0, "citation_precision": 0.5}, policy)[0]


def test_case_policy_missing_required_metric_fails():
    policy = CasePassPolicy(search={"precision@10": 0.6})
    case = _case("c1", [], gold_chunks=[1])
    assert case_passed(case, _obs(), {"precision@5": 1.0}, policy) == (
        False, {"missing_metric:precision@10": True}
    )


def test_case_policy_can_disable_abstention_check_but_not_runtime_failures():
    policy = CasePassPolicy(require_abstention=False)
    case = _case("c1", [])
    assert case_passed(case, _obs(sources=[_src(404)]), {}, policy) == (
        True, {"abstained": False}
    )
    assert not case_passed(case, None, {}, policy)[0]
    assert not case_passed(case, _obs(error="provider_failed"), {}, policy)[0]


@pytest.mark.parametrize("kwargs", [
    {"search": {}},
    {"full_context": {}},
    {"search": {"typo_metric": 0.5}},
    {"search": {"precision@0": 0.5}},
    {"full_context": {"precision@5": 0.5}},
    *[{"search": {"recall@5": value}} for value in [True, "0.5", -0.1, 1.1, float("nan"), float("inf")]],
    {"require_abstention": "false"},
])
def test_case_policy_rejects_invalid_configuration(kwargs):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        CasePassPolicy(**kwargs)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_case_policy_rejects_invalid_metric_values(value):
    with pytest.raises(ContractError, match="finite number"):
        case_passed(_case("c1", [], gold_chunks=[1]), _obs(), {"gold_chunk_coverage": value})


def test_case_policy_default_maps_are_independent():
    policy = CasePassPolicy()
    policy.search["gold_chunk_coverage"] = 0.0
    case = _case("c1", [], gold_chunks=[1])
    assert not case_passed(case, _obs(), {"gold_chunk_coverage": 0.5})[0]


# ---- aggregate ----


def _result(case_id: str, *, passed: bool, metrics=None, checks=None) -> CaseResult:
    return CaseResult(
        case_id=case_id, passed=passed, metrics=metrics or {}, checks=checks or {}
    )


def test_aggregate_eligible_cases_by_arm():
    cases = [
        _case("a", ["parameter"], gold_chunks=[1]),
        _case("b", ["parameter", "temporal"], gold_chunks=[1]),
        _case("c", ["temporal"], gold_segments=[_seg(2.0, 8.0)]),
    ]
    results = [
        _result("a", passed=True, metrics={"recall@5": 0.8, "gold_chunk_coverage": 1.0}),
        _result("b", passed=False, metrics={"recall@5": 0.4, "gold_chunk_coverage": 0.5}),
        _result("c", passed=True, metrics={"citation_hit": 1.0}),
    ]
    metrics, _ = aggregate(cases, results)

    assert metrics["recall@5"].value == pytest.approx(0.6)
    assert metrics["recall@5"].eligible_cases == 2
    assert metrics["citation_hit"].value == 1.0
    assert metrics["citation_hit"].eligible_cases == 1
    assert metrics["case_pass_rate"].value == pytest.approx(2 / 3)
    assert metrics["case_pass_rate"].eligible_cases == 3
    assert metrics["provider_failure_rate"].value == 0.0
    assert metrics["provider_failure_rate"].eligible_cases == 3


def test_aggregate_provider_failure_rate_excludes_missing():
    cases = [_case("a", ["parameter"]), _case("b", ["parameter"])]
    results = [
        _result("a", passed=False, checks={"provider_failed": True}),
        _result("b", passed=False, checks={"missing_observation": True}),
    ]
    metrics, _ = aggregate(cases, results)

    assert metrics["provider_failure_rate"].value == 1.0
    assert metrics["provider_failure_rate"].eligible_cases == 1


def test_aggregate_slices_multi_tag_counted_in_each():
    cases = [
        _case("a", ["parameter"], gold_chunks=[1]),
        _case("b", ["parameter", "temporal"], gold_chunks=[1]),
        _case("c", ["temporal"], gold_segments=[_seg(2.0, 8.0)]),
    ]
    results = [
        _result("a", passed=True, metrics={"recall@5": 1.0}),
        _result("b", passed=True, metrics={"recall@5": 1.0}),
        _result("c", passed=True, metrics={"citation_hit": 1.0}),
    ]
    _, slices = aggregate(cases, results)

    assert slices["parameter"]["case_pass_rate"].eligible_cases == 2
    assert slices["temporal"]["case_pass_rate"].eligible_cases == 2
    assert slices["parameter"]["recall@5"].eligible_cases == 2
    assert slices["temporal"]["recall@5"].eligible_cases == 1


def test_aggregate_empty_results():
    metrics, slices = aggregate([], [])

    assert metrics["case_pass_rate"].value == 0.0
    assert metrics["case_pass_rate"].eligible_cases == 0
    assert metrics["provider_failure_rate"].value == 0.0
    assert metrics["provider_failure_rate"].eligible_cases == 0
    assert slices == {}
