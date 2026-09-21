"""Serializable analysis, diagnostic, power, and report models."""

from typing import Any

from pydantic import Field, field_validator

from experimentation_toolkit.models.core import Experiment, ToolkitModel
from experimentation_toolkit.models.enums import (
    AlternativeHypothesis,
    CorrectionMethod,
    DiagnosticStatus,
    MDEType,
    MetricType,
)


class ConfidenceInterval(ToolkitModel):
    """An interval whose ``None`` endpoint represents an unbounded side."""

    lower: float | None
    upper: float | None
    confidence_level: float = Field(gt=0.0, lt=1.0)

    @field_validator("upper")
    @classmethod
    def validate_order(cls, upper: float | None, info: Any) -> float | None:
        lower = info.data.get("lower")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("confidence interval lower bound must not exceed upper bound")
        return upper


class DiagnosticResult(ToolkitModel):
    """A factual diagnostic emitted by an analysis."""

    code: str
    status: DiagnosticStatus
    message: str
    details: dict[str, int | float | str | bool | None] = Field(default_factory=dict)


class AnalysisResult(ToolkitModel):
    """Common result schema for treatment-minus-control analyses."""

    metric_name: str | None = None
    metric_type: MetricType
    control_estimate: float
    treatment_estimate: float
    absolute_effect: float
    relative_effect: float | None
    relative_effect_definition: str | None
    confidence_interval: ConfidenceInterval
    test_statistic: float | None
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_level: float = Field(gt=0.0, lt=1.0)
    alternative: AlternativeHypothesis
    control_sample_size: int = Field(gt=0)
    treatment_sample_size: int = Field(gt=0)
    control_standard_deviation: float | None = Field(default=None, ge=0.0)
    treatment_standard_deviation: float | None = Field(default=None, ge=0.0)
    degrees_of_freedom: float | None = Field(default=None, gt=0.0)
    method: str
    null_hypothesis: str
    alternative_hypothesis: str
    diagnostics: tuple[DiagnosticResult, ...] = ()


class SRMResult(ToolkitModel):
    """Chi-square goodness-of-fit sample-ratio-mismatch result."""

    observed_counts: dict[str, int]
    expected_counts: dict[str, float]
    chi_square_statistic: float = Field(ge=0.0)
    p_value: float = Field(ge=0.0, le=1.0)
    significance_level: float = Field(gt=0.0, lt=1.0)
    status: DiagnosticStatus
    method: str = "Pearson chi-square goodness-of-fit test"
    message: str


class MultipleTestingResult(ToolkitModel):
    """Adjusted p-values in the same order as the input hypotheses."""

    original_p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float = Field(gt=0.0, lt=1.0)
    method: CorrectionMethod
    hypothesis_labels: tuple[str, ...] | None = None
    error_rate_control: str


class ProportionSampleSizeResult(ToolkitModel):
    """Normal-approximation sample-size planning result."""

    baseline_rate: float = Field(ge=0.0, le=1.0)
    treatment_rate: float = Field(ge=0.0, le=1.0)
    minimum_detectable_effect: float = Field(gt=0.0)
    mde_type: MDEType
    alpha: float = Field(gt=0.0, lt=1.0)
    target_power: float = Field(gt=0.0, lt=1.0)
    achieved_power: float = Field(ge=0.0, le=1.0)
    allocation_ratio: float = Field(gt=0.0)
    alternative: AlternativeHypothesis
    control_sample_size: int = Field(gt=0)
    treatment_sample_size: int = Field(gt=0)
    total_sample_size: int = Field(gt=0)
    method: str


class AchievedPowerResult(ToolkitModel):
    """Approximate achieved power for fixed group sizes and rates."""

    baseline_rate: float = Field(ge=0.0, le=1.0)
    treatment_rate: float = Field(ge=0.0, le=1.0)
    control_sample_size: int = Field(gt=0)
    treatment_sample_size: int = Field(gt=0)
    alpha: float = Field(gt=0.0, lt=1.0)
    power: float = Field(ge=0.0, le=1.0)
    alternative: AlternativeHypothesis
    method: str


class ExperimentReport(ToolkitModel):
    """Serializable collection of experiment analyses and diagnostics."""

    experiment: Experiment
    analyses: tuple[AnalysisResult, ...]
    diagnostics: tuple[SRMResult | DiagnosticResult, ...] = ()
    multiple_testing: MultipleTestingResult | None = None
    metadata: dict[str, int | float | str | bool | None] = Field(default_factory=dict)
