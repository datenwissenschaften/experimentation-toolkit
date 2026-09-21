"""Two-sample analysis for binary outcomes."""

from math import sqrt

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
    validate_nonnegative_integer,
    validate_positive_integer,
    validate_probability,
)


def _wilson_bounds(successes: int, total: int, z_value: float) -> tuple[float, float]:
    proportion = successes / total
    z_squared = z_value**2
    denominator = 1.0 + z_squared / total
    center = (proportion + z_squared / (2.0 * total)) / denominator
    half_width = (
        z_value
        * sqrt(proportion * (1.0 - proportion) / total + z_squared / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _newcombe_difference_interval(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    confidence_level: float,
    alternative: AlternativeHypothesis,
) -> ConfidenceInterval:
    alpha = 1.0 - confidence_level
    tail_probability = alpha / 2.0 if alternative is AlternativeHypothesis.TWO_SIDED else alpha
    z_value = float(stats.norm.ppf(1.0 - tail_probability))
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    effect = treatment_rate - control_rate
    control_lower, control_upper = _wilson_bounds(control_successes, control_total, z_value)
    treatment_lower, treatment_upper = _wilson_bounds(treatment_successes, treatment_total, z_value)
    lower = effect - sqrt(
        (treatment_rate - treatment_lower) ** 2 + (control_upper - control_rate) ** 2
    )
    upper = effect + sqrt(
        (treatment_upper - treatment_rate) ** 2 + (control_rate - control_lower) ** 2
    )
    if alternative is AlternativeHypothesis.GREATER:
        return ConfidenceInterval(lower=float(lower), upper=None, confidence_level=confidence_level)
    if alternative is AlternativeHypothesis.LESS:
        return ConfidenceInterval(lower=None, upper=float(upper), confidence_level=confidence_level)
    return ConfidenceInterval(
        lower=float(lower), upper=float(upper), confidence_level=confidence_level
    )


def _score_test(
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    alternative: AlternativeHypothesis,
) -> tuple[float, float]:
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
    standard_error = sqrt(
        pooled_rate * (1.0 - pooled_rate) * (1.0 / control_total + 1.0 / treatment_total)
    )
    if standard_error == 0.0:
        return 0.0, 1.0
    statistic = (treatment_rate - control_rate) / standard_error
    if alternative is AlternativeHypothesis.GREATER:
        p_value = stats.norm.sf(statistic)
    elif alternative is AlternativeHypothesis.LESS:
        p_value = stats.norm.cdf(statistic)
    else:
        p_value = 2.0 * stats.norm.sf(abs(statistic))
    return float(statistic), float(p_value)


def analyze_proportion(
    *,
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int,
    confidence_level: float = 0.95,
    alternative: AlternativeHypothesis | str = AlternativeHypothesis.TWO_SIDED,
    metric_name: str | None = None,
) -> AnalysisResult:
    """Analyze a treatment-minus-control difference in independent proportions.

    The test is the pooled two-sample score z-test for ``H0: p_t - p_c = 0``.
    The confidence interval is Newcombe's hybrid score interval based on Wilson
    intervals, without continuity correction. One-sided alternatives return the
    corresponding one-sided interval with ``None`` for its unbounded endpoint.
    """
    control_total = validate_positive_integer(control_total, "control_total")
    treatment_total = validate_positive_integer(treatment_total, "treatment_total")
    control_successes = validate_nonnegative_integer(control_successes, "control_successes")
    treatment_successes = validate_nonnegative_integer(treatment_successes, "treatment_successes")
    if control_successes > control_total:
        raise ValueError("control_successes must not exceed control_total")
    if treatment_successes > treatment_total:
        raise ValueError("treatment_successes must not exceed treatment_total")
    confidence_level = validate_probability(
        confidence_level, "confidence_level", open_interval=True
    )
    alternative = normalize_alternative(alternative)

    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    effect = treatment_rate - control_rate
    relative_effect = effect / control_rate if control_rate > 0.0 else None
    diagnostics: list[DiagnosticResult] = []
    if relative_effect is None:
        diagnostics.append(
            DiagnosticResult(
                code="RELATIVE_EFFECT_UNDEFINED",
                status=DiagnosticStatus.WARN,
                message="Relative lift is undefined because the control rate is zero.",
            )
        )

    pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
    expected_cells = (
        control_total * pooled_rate,
        control_total * (1.0 - pooled_rate),
        treatment_total * pooled_rate,
        treatment_total * (1.0 - pooled_rate),
    )
    if min(expected_cells) < 5.0:
        diagnostics.append(
            DiagnosticResult(
                code="ASYMPTOTIC_APPROXIMATION_WARNING",
                status=DiagnosticStatus.WARN,
                message=(
                    "At least one expected success/failure count under the null is below 5; "
                    "the score-test normal approximation may be inaccurate."
                ),
                details={"minimum_expected_cell_count": float(min(expected_cells))},
            )
        )

    statistic, p_value = _score_test(
        control_successes,
        control_total,
        treatment_successes,
        treatment_total,
        alternative,
    )
    interval = _newcombe_difference_interval(
        control_successes,
        control_total,
        treatment_successes,
        treatment_total,
        confidence_level,
        alternative,
    )
    direction = {
        AlternativeHypothesis.TWO_SIDED: "p_treatment - p_control != 0",
        AlternativeHypothesis.GREATER: "p_treatment - p_control > 0",
        AlternativeHypothesis.LESS: "p_treatment - p_control < 0",
    }[alternative]
    return AnalysisResult(
        metric_name=metric_name,
        metric_type=MetricType.PROPORTION,
        control_estimate=float(control_rate),
        treatment_estimate=float(treatment_rate),
        absolute_effect=float(effect),
        relative_effect=float(relative_effect) if relative_effect is not None else None,
        relative_effect_definition=(
            "(treatment rate - control rate) / control rate"
            if relative_effect is not None
            else None
        ),
        confidence_interval=interval,
        test_statistic=statistic,
        p_value=p_value,
        confidence_level=confidence_level,
        alternative=alternative,
        control_sample_size=control_total,
        treatment_sample_size=treatment_total,
        method=(
            "Pooled two-sample score z-test; Newcombe hybrid Wilson confidence interval "
            "for the absolute risk difference (no continuity correction)"
        ),
        null_hypothesis="p_treatment - p_control = 0",
        alternative_hypothesis=direction,
        diagnostics=tuple(diagnostics),
    )
