from __future__ import annotations


from typing import Literal, Annotated

from pydantic import BaseModel, ConfigDict, Field


class GoldSegment(BaseModel):   # {start: float, end: float}
    start: float; end: float
    model_config = ConfigDict(extra="forbid")

class Scope(BaseModel):
    episode_id: int | None = None
    podcast_id: int | None = None
    model_config = ConfigDict(extra="forbid")

class Case(BaseModel):
    id: str
    tags: list[str]
    critical: bool
    question: str
    expected_arm: Literal["full-context", "search"]
    scope: Scope
    gold_chunks: list[int] | None
    gold_segments: list[GoldSegment] | None
    reference_answer: str
    required_claims: list[str]
    conflicts: list[str] = Field(default_factory=list)   # 可选，dataset 里常缺省
    model_config = ConfigDict(extra="forbid")

class CaseResult(BaseModel):
    case_id: str
    passed: bool
    metrics: dict[str, float]
    checks: dict[str, float | bool | str]
    judge: JudgeScores | None = None   # judge 跳过/失败时为 None
    model_config = ConfigDict(extra="forbid")

class Dataset(BaseModel):
    dataset_name: str
    dataset_version: str
    schema_version: str
    cases: list[Case]
    model_config = ConfigDict(extra="forbid")


class Source(BaseModel):
    chunk_id: int | None
    start: float
    end: float
    episode_id: int
    text: str
    level: Literal["leaf", "parent"] = "leaf"
    child_chunk_ids: list[int] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

class Observation(BaseModel):
    case_id: str
    arm: Literal["full-context", "search"]   # 实际臂（informational）
    answer_text: str
    sources: list[Source]
    duration_ms: int | None = None
    error: str | None = None                 # 脱敏错误码（如 "provider_failed"），非原始异常
    model_config = ConfigDict(extra="forbid")

class ObservationBundle(BaseModel):
    dataset_fingerprint: str
    provenance: Literal["contract_smoke", "live"]
    observations: list[Observation]
    model_config = ConfigDict(extra="forbid")


class MetricGate(BaseModel):
    metric: str                       # 指标名（见 §5/§6 规范名）
    minimum: float | None = Field(default=None, strict=True, ge=0, le=1, allow_inf_nan=False)
    maximum: float | None = Field(default=None, strict=True, ge=0, le=1, allow_inf_nan=False)
    max_regression: float | None = Field(default=None, strict=True, ge=0, le=1, allow_inf_nan=False)
    model_config = ConfigDict(extra="forbid")

class GatePolicy(BaseModel):
    metric_gates: list[MetricGate]
    critical_no_regression: bool = Field(default=True, strict=True)
    required_provenance: list[Literal["contract_smoke", "live"]] = Field(default_factory=list)
    k: int = Field(default=5, strict=True)
    model_config = ConfigDict(extra="forbid")


# Per-case minimums are independent of aggregate MetricGate thresholds.
CaseThreshold = Annotated[
    float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)
]
SearchMetric = Annotated[
    str, Field(pattern=r"^(gold_chunk_coverage|(recall|precision|mrr|ndcg)@[1-9][0-9]*)$")
]
ContextMetric = Literal["citation_hit", "citation_precision"]


class CasePassPolicy(BaseModel):
    """All configured minimums must pass; omitted metrics do not affect passing."""

    search: dict[SearchMetric, CaseThreshold] = Field(
        default_factory=lambda: {"gold_chunk_coverage": 1.0}, min_length=1
    )
    full_context: dict[ContextMetric, CaseThreshold] = Field(
        default_factory=lambda: {"citation_hit": 1.0}, min_length=1
    )
    require_abstention: bool = Field(default=True, strict=True)
    model_config = ConfigDict(extra="forbid")


class GateResult(BaseModel):
    name: str
    status: Literal["PASS", "FAIL", "SKIP", "NO_BASELINE"]
    actual: float | None
    baseline: float | None
    threshold: float | None
    detail: str
    model_config = ConfigDict(extra="forbid")



class MetricValue(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: float
    eligible_cases: int





class Report(BaseModel):
    dataset_name: str
    dataset_fingerprint: str
    case_count: int
    metrics: dict[str, MetricValue]
    slices: dict[str, dict[str, MetricValue]]   # tag -> metrics
    cases: list[CaseResult]
    gates: list[GateResult]
    passed: bool
    metadata: dict[str, str | int | bool]       # provenance、judge_available、arm 分布…
    model_config = ConfigDict(extra="forbid")

class Baseline(BaseModel):   # 提交的回归锚点
    dataset_fingerprint: str
    k: int = Field(strict=True, gt=0)
    provenance: Literal["contract_smoke", "live"]
    metrics: dict[str, Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]]
    eligible_cases: dict[str, Annotated[int, Field(ge=0, strict=True)]]
    case_passed: dict[str, Annotated[bool, Field(strict=True)]]
    created_at: str
    model_config = ConfigDict(extra="forbid")

class DatasetFingerprintMismatch(Exception):
    """Raised when the fingerprint of the dataset does not match the stored finger print"""
    
class ContractError(Exception):
    """Unknown error beside DatasetFingerprintMismatch"""


class JudgeScores(BaseModel):
    correctness: float = Field(ge=0, le=1, allow_inf_nan=False)
    groundedness: float = Field(ge=0, le=1, allow_inf_nan=False)
    relevance: float = Field(ge=0, le=1, allow_inf_nan=False)
    completeness: float = Field(ge=0, le=1, allow_inf_nan=False)
    context_relevance: float = Field(ge=0, le=1, allow_inf_nan=False)
    unsupported_claim_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    conflict_disclosure_rate: float = Field(ge=0, le=1, allow_inf_nan=False)
    reason: str = Field(min_length=1)
    model_config = ConfigDict(extra="forbid")
