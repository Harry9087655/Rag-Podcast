"""Offline lifecycle, report integration, and deterministic isolation tests."""

import asyncio
import json
import subprocess
import sys
from types import SimpleNamespace

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, AuthenticationError, BadRequestError

from rag_podcast.evaluation.contracts import (
    Case, CaseResult, ContractError, Dataset, DatasetFingerprintMismatch,
    GateResult, JudgeScores, Observation, ObservationBundle, Report, Scope,
)
from rag_podcast.evaluation.judge import judge_answer, judge_report
from rag_podcast.evaluation.loaders import load_baseline, save_baseline
from rag_podcast.evaluation.metrics import aggregate
from rag_podcast.evaluation.render import render_json, render_markdown


SCORES = dict(
    correctness=1, groundedness=1, relevance=1, completeness=1, context_relevance=0,
    unsupported_claim_rate=0, conflict_disclosure_rate=0, reason="Correct abstention.",
)


def make_case(case_id="case"):
    return Case(
        id=case_id, tags=["no_knowledge"], critical=False, question="Unknown?",
        expected_arm="search", scope=Scope(), gold_chunks=None, gold_segments=None,
        reference_answer="No information.", required_claims=[],
    )


def make_obs(case_id="case", error=None):
    return Observation(case_id=case_id, arm="search", answer_text="No information.",
                       sources=[], error=error)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.chat = SimpleNamespace(completions=self)
        self.closed = False
        self.calls = 0
        self.options = None

    def factory(self, **kwargs):
        self.options = kwargs
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def create(self, **kwargs):
        self.calls += 1
        self.last_request = kwargs
        outcome = next(self.outcomes)
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=outcome),
        )])


@pytest.fixture
def settings_stub(monkeypatch):
    settings = SimpleNamespace(
        evaluator_model="judge-model", llm_model="answer-model", llm_api_key="secret-key",
        llm_base_url="https://example.invalid", llm_timeout_seconds=1,
    )
    monkeypatch.setitem(sys.modules, "rag_podcast.config", SimpleNamespace(settings=settings))
    return settings


@pytest.mark.parametrize("error_kind", ["timeout", "connection", "auth", "rejected"])
@pytest.mark.parametrize("after_format_failure", [False, True])
async def test_provider_failure_is_not_retried_and_logs_no_secrets(
    settings_stub, caplog, error_kind, after_format_failure
):
    request = httpx.Request("POST", "https://example.invalid")
    if error_kind == "timeout":
        error = APITimeoutError(request)
        category = "judge_timeout"
    elif error_kind == "connection":
        error = APIConnectionError(message="secret-provider-payload", request=request)
        category = "judge_connection_failed"
    else:
        response = httpx.Response(401 if error_kind == "auth" else 400, request=request)
        cls = AuthenticationError if error_kind == "auth" else BadRequestError
        error = cls("secret-provider-payload", response=response, body={"key": "secret-key"})
        category = "judge_provider_failed"
    outcomes = (["bad JSON"] if after_format_failure else []) + [error]
    client = FakeClient(outcomes)
    obs = make_obs()
    before = obs.model_dump()

    assert await judge_answer(make_case(), obs, client_factory=client.factory) is None
    assert client.calls == len(outcomes)
    assert client.closed
    assert obs.model_dump() == before
    assert category in caplog.text
    assert "secret-provider-payload" not in caplog.text
    assert "secret-key" not in caplog.text


@pytest.mark.parametrize("error", [asyncio.CancelledError(), RuntimeError("bug")])
async def test_cancellation_and_programming_errors_propagate(settings_stub, error):
    client = FakeClient([error])
    with pytest.raises(type(error)):
        await judge_answer(make_case(), make_obs(), client_factory=client.factory)
    assert client.calls == 1
    assert client.closed


async def test_identity_and_qa_failure_do_not_create_client():
    def forbidden_factory(**kwargs):
        pytest.fail("client must not be created")

    with pytest.raises(ContractError, match="does not match"):
        await judge_answer(make_case(), make_obs("other"), client_factory=forbidden_factory)
    assert await judge_answer(
        make_case(), make_obs(error="provider_failed"), client_factory=forbidden_factory,
    ) is None


@pytest.mark.parametrize("field,value", [
    ("extra", 1), ("correctness", float("nan")), ("correctness", float("inf")),
    ("correctness", -1), ("correctness", "1"), ("correctness", True), ("reason", ""),
])
async def test_invalid_contract_retries_once(settings_stub, caplog, field, value):
    payload = {**SCORES, field: value}
    client = FakeClient([json.dumps(payload)] * 2)
    assert await judge_answer(make_case(), make_obs(), client_factory=client.factory) is None
    assert client.calls == 2
    assert client.closed
    assert "judge_invalid_output" in caplog.text


@pytest.mark.parametrize("model", [None, "judge-model"])
async def test_client_options_and_model_selection(settings_stub, model):
    settings_stub.evaluator_model = model
    client = FakeClient([json.dumps(SCORES)])
    result = await judge_answer(make_case(), make_obs(), client_factory=client.factory)
    assert result.reason == SCORES["reason"]
    assert client.options["max_retries"] == 0
    assert client.options["timeout"] == 1
    assert client.last_request["model"] == (model or "answer-model")
    assert client.last_request["response_format"] == {"type": "json_object"}


def test_import_and_contract_construction_do_not_load_model_or_settings():
    script = '''
import builtins
original_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "openai" or name.startswith("openai.") or name == "rag_podcast.config":
        raise AssertionError("unexpected model/config import")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded
import rag_podcast.evaluation
from rag_podcast.evaluation.judge import build_judge_messages
from rag_podcast.evaluation.contracts import Scope
Scope()
'''
    completed = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr


def report_fixture():
    cases = [make_case(name) for name in ("a", "b", "qa-failed", "missing")]
    dataset = Dataset(dataset_name="test", dataset_version="1", schema_version="1", cases=cases)
    observations = [make_obs("b"), make_obs("qa-failed", "provider_failed"), make_obs("a")]
    bundle = ObservationBundle(dataset_fingerprint="fp", provenance="contract_smoke",
                               observations=observations)
    results = [CaseResult(case_id=c.id, passed=c.id in {"a", "b"}, metrics={}, checks={})
               for c in cases]
    metrics, slices = aggregate(cases, results)
    report = Report(
        dataset_name="test", dataset_fingerprint="fp", case_count=4,
        metrics=metrics, slices=slices, cases=results,
        gates=[GateResult(name="example", status="FAIL", actual=0.5, baseline=None,
                          threshold=1, detail="deterministic")],
        passed=False, metadata={"provenance": "contract_smoke", "k": 5},
    )
    return dataset, bundle, report


@pytest.mark.parametrize("mode", ["disabled", "success", "failed", "partial"])
async def test_report_counts_rendering_and_deterministic_isolation(mode, tmp_path):
    dataset, bundle, report = report_fixture()
    before = report.model_dump()
    called = []

    async def fake_judge(case, obs):
        called.append(case.id)
        assert case.id == obs.case_id
        if mode == "failed" or (mode == "partial" and case.id == "b"):
            return None
        return JudgeScores(**SCORES)

    enriched = await judge_report(dataset, bundle, report, run_judge=mode != "disabled",
                                  judge_fn=fake_judge)
    assert called == ([] if mode == "disabled" else ["a", "b"])
    attempted = len(called)
    succeeded = {"disabled": 0, "success": 2, "failed": 0, "partial": 1}[mode]
    assert enriched.metadata == {
        "provenance": "contract_smoke", "k": 5, "judge_requested": mode != "disabled",
        "judge_attempted_cases": attempted, "judge_succeeded_cases": succeeded,
        "judge_failed_cases": attempted - succeeded, "judge_skipped_cases": 4 - attempted,
        "judge_available": succeeded > 0,
    }
    assert report.model_dump() == before
    for field in ("metrics", "slices", "gates", "passed"):
        assert getattr(enriched, field) == getattr(report, field)
    assert [c.passed for c in enriched.cases] == [c.passed for c in report.cases]
    assert aggregate(dataset.cases, enriched.cases) == (report.metrics, report.slices)
    data = json.loads(render_json(enriched))
    markdown = render_markdown(enriched)
    if succeeded:
        assert data["cases"][0]["judge"]["reason"] == SCORES["reason"]
        assert SCORES["reason"] in markdown
        assert "conflict omission rate" in markdown
        assert "Lower is better" in markdown
    assert f"judge_failed_cases: {attempted - succeeded}" in markdown
    if mode == "disabled":
        repeated = await judge_report(dataset, bundle, report)
        assert render_json(enriched) == render_json(repeated)
    original_path, enriched_path = tmp_path / "original.json", tmp_path / "enriched.json"
    save_baseline(report, original_path)
    save_baseline(enriched, enriched_path)
    original = load_baseline(original_path).model_dump(exclude={"created_at"})
    assert load_baseline(enriched_path).model_dump(exclude={"created_at"}) == original


async def test_disabled_report_clears_stale_scores():
    dataset, bundle, report = report_fixture()
    report.cases[0].judge = JudgeScores(**SCORES)
    enriched = await judge_report(dataset, bundle, report)
    assert all(case.judge is None for case in enriched.cases)
    assert report.cases[0].judge is not None


async def test_report_contract_errors_precede_judge():
    dataset, bundle, report = report_fixture()
    bundle.dataset_fingerprint = "different"
    with pytest.raises(DatasetFingerprintMismatch):
        await judge_report(dataset, bundle, report, run_judge=True)
    bundle.dataset_fingerprint = "fp"
    bundle.observations.append(make_obs("unknown"))
    with pytest.raises(ContractError, match="unknown"):
        await judge_report(dataset, bundle, report, run_judge=True)
