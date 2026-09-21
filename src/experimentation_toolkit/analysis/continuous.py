"""Welch analysis for independent continuous outcomes."""

from math import sqrt
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


def _welch_degrees_of_freedom(
    control_variance: float,
    control_size: int,
    treatment_variance: float,
    treatment_size: int,
) -> float:
    control_term = control_variance / control_size
    treatment_term = treatment_variance / treatment_size
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
    control mean is strictly positive, a conservative check for a ratio-scale metric.
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
    with np.errstate(over="ignore", invalid="ignore"):
        control_mean = float(np.mean(control_values, dtype=np.float64))
        treatment_mean = float(np.mean(treatment_values, dtype=np.float64))
        control_variance = float(np.var(control_values, ddof=1, dtype=np.float64))
        treatment_variance = float(np.var(treatment_values, ddof=1, dtype=np.float64))
    if not all(
        np.isfinite(value)
        for value in (control_mean, treatment_mean, control_variance, treatment_variance)
    ):
        raise ValueError("sample moments are non-finite; rescale extreme observations")
    control_sd = sqrt(control_variance)
    treatment_sd = sqrt(treatment_variance)
    effect = treatment_mean - control_mean
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

    standard_error = sqrt(control_variance / control_size + treatment_variance / treatment_size)
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
            control_variance, control_size, treatment_variance, treatment_size
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
        if control_variance == 0.0 or treatment_variance == 0.0:
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

    relative_effect = (
        effect / control_mean if control_mean > 0.0 and treatment_mean >= 0.0 else None
    )
    if relative_effect is None:
        diagnostics.append(
            DiagnosticResult(
                code="RELATIVE_EFFECT_NOT_REPORTED",
                status=DiagnosticStatus.WARN,
                message=(
                    "Relative mean difference is not reported because the observed means do "
                    "not support a non-negative ratio-scale interpretation."
                ),
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
