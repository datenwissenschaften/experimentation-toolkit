"""Welch analysis for independent continuous outcomes."""

from math import hypot, sqrt
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import stats

from experimentation_toolkit.models.enums import (
    AlternativeHypothesis,
    DiagnosticStatus,
    MetricType,
)
from experimentation_toolkit.models.results import (
    AnalysisResult,
    ConfidenceInterval,
    DiagnosticResult,
)
from experimentation_toolkit.validation.inputs import (
    normalize_alternative,
    normalize_numeric_sample,
    validate_probability,
)


def _stable_mean_and_standard_deviation(
    values: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Compute float64 moments after scaling to avoid avoidable overflow/underflow."""
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return 0.0, 0.0
    scaled = values / scale
    scaled_mean = float(np.mean(scaled, dtype=np.float64))
    centered = scaled - scaled_mean
    scaled_sum_of_squares = float(np.dot(centered, centered))
    mean = scaled_mean * scale
    standard_deviation = scale * sqrt(scaled_sum_of_squares / (values.size - 1))
    if not np.isfinite(mean) or not np.isfinite(standard_deviation):
        raise ValueError("sample moments are non-finite; rescale extreme observations")
    return mean, standard_deviation


def _welch_degrees_of_freedom(
    control_standard_error: float,
    control_size: int,
    treatment_standard_error: float,
    treatment_size: int,
) -> float:
    scale = max(control_standard_error, treatment_standard_error)
    control_term = (control_standard_error / scale) ** 2
    treatment_term = (treatment_standard_error / scale) ** 2
    return (control_term + treatment_term) ** 2 / (
        control_term**2 / (control_size - 1) + treatment_term**2 / (treatment_size - 1)
    )


def _welch_interval(
    effect: float,
    standard_error: float,
    degrees_of_freedom: float,
    confidence_level: float,
    alternative: AlternativeHypothesis,
) -> ConfidenceInterval:
    alpha = 1.0 - confidence_level
    tail_probability = alpha / 2.0 if alternative is AlternativeHypothesis.TWO_SIDED else alpha
    critical_value = float(stats.t.ppf(1.0 - tail_probability, degrees_of_freedom))
    margin = critical_value * standard_error
    if alternative is AlternativeHypothesis.GREATER:
        return ConfidenceInterval(
            lower=effect - margin, upper=None, confidence_level=confidence_level
        )
    if alternative is AlternativeHypothesis.LESS:
        return ConfidenceInterval(
            lower=None, upper=effect + margin, confidence_level=confidence_level
        )
    return ConfidenceInterval(
        lower=effect - margin,
        upper=effect + margin,
        confidence_level=confidence_level,
    )


def analyze_continuous(
    *,
    control: npt.ArrayLike,
    treatment: npt.ArrayLike,
    confidence_level: float = 0.95,
    alternative: AlternativeHypothesis | str = AlternativeHypothesis.TWO_SIDED,
    nonfinite_policy: Literal["raise", "omit"] = "raise",
    metric_name: str | None = None,
) -> AnalysisResult:
    """Analyze independent means using Welch's unequal-variance t procedure.

    Inputs must be one-dimensional and contain at least two finite values per arm.
    Non-finite observations raise by default; callers may explicitly request omission.
    A relative effect is emitted only when both group means are non-negative and the
    control mean is numerically resolved relative to the observed data scale.
    """
    control_values, control_omitted = normalize_numeric_sample(
        control, "control", nonfinite_policy=nonfinite_policy
    )
    treatment_values, treatment_omitted = normalize_numeric_sample(
        treatment, "treatment", nonfinite_policy=nonfinite_policy
    )
    confidence_level = validate_probability(
        confidence_level, "confidence_level", open_interval=True
    )
    alternative = normalize_alternative(alternative)
    control_size = int(control_values.size)
    treatment_size = int(treatment_values.size)
    control_mean, control_sd = _stable_mean_and_standard_deviation(control_values)
    treatment_mean, treatment_sd = _stable_mean_and_standard_deviation(treatment_values)
    effect = treatment_mean - control_mean
    if not np.isfinite(effect):
        raise ValueError("the mean difference is non-finite; rescale extreme observations")
    diagnostics: list[DiagnosticResult] = []

    omitted = control_omitted + treatment_omitted
    if omitted:
        diagnostics.append(
            DiagnosticResult(
                code="NONFINITE_OBSERVATIONS_OMITTED",
                status=DiagnosticStatus.WARN,
                message="Non-finite observations were omitted by explicit caller policy.",
                details={
                    "control_omitted": control_omitted,
                    "treatment_omitted": treatment_omitted,
                },
            )
        )
    if control_size < 30 or treatment_size < 30:
        diagnostics.append(
            DiagnosticResult(
                code="SMALL_SAMPLE",
                status=DiagnosticStatus.WARN,
                message=(
                    "At least one arm has fewer than 30 observations; Welch inference relies "
                    "more strongly on approximate normality of that arm's sample mean."
                ),
                details={"control_size": control_size, "treatment_size": treatment_size},
            )
        )

    control_standard_error = control_sd / sqrt(control_size)
    treatment_standard_error = treatment_sd / sqrt(treatment_size)
    standard_error = hypot(control_standard_error, treatment_standard_error)
    if standard_error == 0.0:
        diagnostics.append(
            DiagnosticResult(
                code="DEGENERATE_ZERO_VARIANCE",
                status=DiagnosticStatus.WARN,
                message=(
                    "Both samples have zero variance; Welch's reference distribution is "
                    "undefined, so no test statistic or p-value is reported."
                ),
            )
        )
        interval = ConfidenceInterval(
            lower=effect if alternative is not AlternativeHypothesis.LESS else None,
            upper=effect if alternative is not AlternativeHypothesis.GREATER else None,
            confidence_level=confidence_level,
        )
        statistic = None
        p_value = None
        degrees_of_freedom = None
    else:
        degrees_of_freedom = _welch_degrees_of_freedom(
            control_standard_error,
            control_size,
            treatment_standard_error,
            treatment_size,
        )
        statistic = effect / standard_error
        if alternative is AlternativeHypothesis.GREATER:
            p_value = float(stats.t.sf(statistic, degrees_of_freedom))
        elif alternative is AlternativeHypothesis.LESS:
            p_value = float(stats.t.cdf(statistic, degrees_of_freedom))
        else:
            p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_of_freedom))
        interval = _welch_interval(
            effect,
            standard_error,
            degrees_of_freedom,
            confidence_level,
            alternative,
        )
        if control_sd == 0.0 or treatment_sd == 0.0:
            diagnostics.append(
                DiagnosticResult(
                    code="ONE_ZERO_VARIANCE_SAMPLE",
                    status=DiagnosticStatus.WARN,
                    message=(
                        "One sample has zero variance; Welch inference is computable but the "
                        "data-generating assumptions deserve review."
                    ),
                )
            )

    observed_scale = max(
        float(np.max(np.abs(control_values))),
        float(np.max(np.abs(treatment_values))),
        np.finfo(np.float64).tiny,
    )
    denominator_resolution = sqrt(np.finfo(np.float64).eps) * observed_scale
    relative_effect = (
        effect / control_mean
        if control_mean > denominator_resolution and treatment_mean >= 0.0
        else None
    )
    if relative_effect is None:
        diagnostics.append(
            DiagnosticResult(
                code="RELATIVE_EFFECT_NOT_REPORTED",
                status=DiagnosticStatus.WARN,
                message=(
                    "Relative mean difference is not reported because the observed means do "
                    "not support a non-negative, numerically resolved ratio-scale interpretation."
                ),
                details={"minimum_resolved_control_mean": float(denominator_resolution)},
            )
        )
    direction = {
        AlternativeHypothesis.TWO_SIDED: "mu_treatment - mu_control != 0",
        AlternativeHypothesis.GREATER: "mu_treatment - mu_control > 0",
        AlternativeHypothesis.LESS: "mu_treatment - mu_control < 0",
    }[alternative]
    return AnalysisResult(
        metric_name=metric_name,
        metric_type=MetricType.CONTINUOUS,
        control_estimate=control_mean,
        treatment_estimate=treatment_mean,
        absolute_effect=effect,
        relative_effect=relative_effect,
        relative_effect_definition=(
            "(treatment mean - control mean) / control mean"
            if relative_effect is not None
            else None
        ),
        confidence_interval=interval,
        test_statistic=statistic,
        p_value=p_value,
        confidence_level=confidence_level,
        alternative=alternative,
        control_sample_size=control_size,
        treatment_sample_size=treatment_size,
        control_standard_deviation=control_sd,
        treatment_standard_deviation=treatment_sd,
        degrees_of_freedom=degrees_of_freedom,
        method="Welch's two-sample t-test and Welch-Satterthwaite confidence interval",
        null_hypothesis="mu_treatment - mu_control = 0",
        alternative_hypothesis=direction,
        diagnostics=tuple(diagnostics),
    )
