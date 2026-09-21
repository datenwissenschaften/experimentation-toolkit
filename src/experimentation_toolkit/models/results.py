"""Serializable analysis, diagnostic, power, and report models."""

from math import isclose
from typing import Any

from pydantic import Field, field_validator, model_validator

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

    @model_validator(mode="after")
    def validate_bounded_side(self) -> "ConfidenceInterval":
        if self.lower is None and self.upper is None:
            raise ValueError("at least one confidence interval endpoint must be bounded")
        return self


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

    @model_validator(mode="after")
    def validate_result_coherence(self) -> "AnalysisResult":
        if self.confidence_interval.confidence_level != self.confidence_level:
            raise ValueError("analysis and interval confidence levels must match")
        expected_effect = self.treatment_estimate - self.control_estimate
        if not isclose(self.absolute_effect, expected_effect, rel_tol=1e-12, abs_tol=0.0):
            raise ValueError("absolute effect must equal treatment estimate minus control estimate")
        if (self.relative_effect is None) != (self.relative_effect_definition is None):
            raise ValueError(
                "relative effect and its definition must either both be set or both be null"
            )
        if self.alternative is AlternativeHypothesis.TWO_SIDED and (
            self.confidence_interval.lower is None or self.confidence_interval.upper is None
        ):
            raise ValueError("a two-sided analysis requires two bounded interval endpoints")
        if self.alternative is AlternativeHypothesis.GREATER and (
            self.confidence_interval.lower is None or self.confidence_interval.upper is not None
        ):
            raise ValueError("a greater alternative requires a lower-bounded interval")
        if self.alternative is AlternativeHypothesis.LESS and (
            self.confidence_interval.lower is not None or self.confidence_interval.upper is None
        ):
            raise ValueError("a less alternative requires an upper-bounded interval")
        return self


class SRMResult(ToolkitModel):
    """Chi-square goodness-of-fit sample-ratio-mismatch result."""

    observed_counts: dict[str, int]
    expected_counts: dict[str, float]
    chi_square_statistic: float = Field(ge=0.0)
    p_value: float = Field(ge=0.0, le=1.0)
    significance_level: float = Field(gt=0.0, lt=1.0)
    degrees_of_freedom: int = Field(ge=1)
    status: DiagnosticStatus
    method: str = "Pearson chi-square goodness-of-fit test"
    message: str
    diagnostics: tuple[DiagnosticResult, ...] = ()

    @model_validator(mode="after")
    def validate_srm_structure(self) -> "SRMResult":
        if set(self.observed_counts) != set(self.expected_counts):
            raise ValueError("observed and expected SRM counts must have identical keys")
        if any(value < 0 for value in self.observed_counts.values()):
            raise ValueError("observed SRM counts must be non-negative")
        if any(value <= 0.0 for value in self.expected_counts.values()):
            raise ValueError("expected SRM counts must be positive")
        if self.degrees_of_freedom != len(self.observed_counts) - 1:
            raise ValueError("SRM degrees of freedom must equal the number of groups minus one")
        if self.status not in (DiagnosticStatus.PASS, DiagnosticStatus.FAIL):
            raise ValueError("SRM status must be PASS or FAIL")
        return self


class MultipleTestingResult(ToolkitModel):
    """Adjusted p-values in the same order as the input hypotheses."""

    original_p_values: tuple[float, ...]
    adjusted_p_values: tuple[float, ...]
    rejected: tuple[bool, ...]
    alpha: float = Field(gt=0.0, lt=1.0)
    method: CorrectionMethod
    hypothesis_labels: tuple[str, ...] | None = None
    error_rate_control: str

    @model_validator(mode="after")
    def validate_parallel_results(self) -> "MultipleTestingResult":
        size = len(self.original_p_values)
        if size == 0:
            raise ValueError("multiple-testing results must contain at least one hypothesis")
        if len(self.adjusted_p_values) != size or len(self.rejected) != size:
            raise ValueError("multiple-testing result arrays must have equal lengths")
        if self.hypothesis_labels is not None and len(self.hypothesis_labels) != size:
            raise ValueError("hypothesis labels must match the number of p-values")
        if any(value < 0.0 or value > 1.0 for value in self.original_p_values):
            raise ValueError("original p-values must be in [0, 1]")
        if any(value < 0.0 or value > 1.0 for value in self.adjusted_p_values):
            raise ValueError("adjusted p-values must be in [0, 1]")
        if self.rejected != tuple(value <= self.alpha for value in self.adjusted_p_values):
            raise ValueError("rejection decisions must correspond to adjusted p-values and alpha")
        expected_error_rate = (
            "false discovery rate"
            if self.method is CorrectionMethod.BENJAMINI_HOCHBERG
            else "family-wise error rate"
        )
        if self.error_rate_control != expected_error_rate:
            raise ValueError("error-rate description must correspond to the correction method")
        return self


class ProportionSampleSizeResult(ToolkitModel):
    """Normal-approximation sample-size planning result."""

    baseline_rate: float = Field(gt=0.0, lt=1.0)
    treatment_rate: float = Field(gt=0.0, lt=1.0)
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

    @model_validator(mode="after")
    def validate_total_sample_size(self) -> "ProportionSampleSizeResult":
        if self.total_sample_size != self.control_sample_size + self.treatment_sample_size:
            raise ValueError("total sample size must equal control plus treatment sample sizes")
        if self.achieved_power < self.target_power:
            raise ValueError("achieved power must meet or exceed target power")
        return self


class AchievedPowerResult(ToolkitModel):
    """Approximate achieved power for fixed group sizes and rates."""

    baseline_rate: float = Field(gt=0.0, lt=1.0)
    treatment_rate: float = Field(gt=0.0, lt=1.0)
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
