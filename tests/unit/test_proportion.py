from math import sqrt

import pytest
from scipy import stats

from experimentation_toolkit import AlternativeHypothesis, analyze_proportion


def test_proportion_matches_reference_score_test() -> None:
    result = analyze_proportion(
        control_successes=1200,
        control_total=10000,
        treatment_successes=1280,
        treatment_total=10050,
    )
    control_rate = 1200 / 10000
    treatment_rate = 1280 / 10050
    pooled = (1200 + 1280) / (10000 + 10050)
    expected_z = (treatment_rate - control_rate) / sqrt(
        pooled * (1 - pooled) * (1 / 10000 + 1 / 10050)
    )
    expected_p = 2 * stats.norm.sf(abs(expected_z))
    assert result.control_estimate == pytest.approx(control_rate)
    assert result.treatment_estimate == pytest.approx(treatment_rate)
    assert result.absolute_effect == pytest.approx(treatment_rate - control_rate)
    assert result.relative_effect == pytest.approx((treatment_rate - control_rate) / control_rate)
    assert result.test_statistic == pytest.approx(expected_z)
    assert result.p_value == pytest.approx(expected_p)


def test_newcombe_interval_uses_independent_scipy_wilson_limits() -> None:
    control_successes, control_total = 7, 53
    treatment_successes, treatment_total = 18, 71
    result = analyze_proportion(
        control_successes=control_successes,
        control_total=control_total,
        treatment_successes=treatment_successes,
        treatment_total=treatment_total,
    )
    control_interval = stats.binomtest(control_successes, control_total).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    treatment_interval = stats.binomtest(treatment_successes, treatment_total).proportion_ci(
        confidence_level=0.95, method="wilson"
    )
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    difference = treatment_rate - control_rate
    expected_lower = difference - sqrt(
        (treatment_rate - treatment_interval.low) ** 2 + (control_interval.high - control_rate) ** 2
    )
    expected_upper = difference + sqrt(
        (treatment_interval.high - treatment_rate) ** 2 + (control_rate - control_interval.low) ** 2
    )
    assert result.confidence_interval.lower == pytest.approx(expected_lower)
    assert result.confidence_interval.upper == pytest.approx(expected_upper)


@pytest.mark.parametrize("alternative", ["greater", "less"])
def test_one_sided_newcombe_interval_matches_scipy_wilson_limits(alternative: str) -> None:
    control_successes, control_total = 3, 25
    treatment_successes, treatment_total = 8, 31
    result = analyze_proportion(
        control_successes=control_successes,
        control_total=control_total,
        treatment_successes=treatment_successes,
        treatment_total=treatment_total,
        alternative=alternative,
    )
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    difference = treatment_rate - control_rate
    if alternative == "greater":
        treatment_lower = (
            stats.binomtest(treatment_successes, treatment_total, alternative="greater")
            .proportion_ci(confidence_level=0.95, method="wilson")
            .low
        )
        control_upper = (
            stats.binomtest(control_successes, control_total, alternative="less")
            .proportion_ci(confidence_level=0.95, method="wilson")
            .high
        )
        expected = difference - sqrt(
            (treatment_rate - treatment_lower) ** 2 + (control_upper - control_rate) ** 2
        )
        assert result.confidence_interval.lower == pytest.approx(expected)
    else:
        treatment_upper = (
            stats.binomtest(treatment_successes, treatment_total, alternative="less")
            .proportion_ci(confidence_level=0.95, method="wilson")
            .high
        )
        control_lower = (
            stats.binomtest(control_successes, control_total, alternative="greater")
            .proportion_ci(confidence_level=0.95, method="wilson")
            .low
        )
        expected = difference + sqrt(
            (treatment_upper - treatment_rate) ** 2 + (control_rate - control_lower) ** 2
        )
        assert result.confidence_interval.upper == pytest.approx(expected)


def test_percentage_points_and_relative_lift_are_distinct() -> None:
    result = analyze_proportion(
        control_successes=100,
        control_total=1000,
        treatment_successes=110,
        treatment_total=1000,
    )
    assert result.absolute_effect == pytest.approx(0.01)
    assert result.relative_effect == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("alternative", "bounded_side"),
    [(AlternativeHypothesis.GREATER, "lower"), (AlternativeHypothesis.LESS, "upper")],
)
def test_one_sided_interval_has_explicit_unbounded_endpoint(
    alternative: AlternativeHypothesis, bounded_side: str
) -> None:
    result = analyze_proportion(
        control_successes=100,
        control_total=1000,
        treatment_successes=130,
        treatment_total=1000,
        alternative=alternative,
    )
    if bounded_side == "lower":
        assert result.confidence_interval.lower is not None
        assert result.confidence_interval.upper is None
    else:
        assert result.confidence_interval.lower is None
        assert result.confidence_interval.upper is not None


def test_directional_p_values_respect_observed_direction() -> None:
    greater = analyze_proportion(
        control_successes=140,
        control_total=1000,
        treatment_successes=100,
        treatment_total=1000,
        alternative="greater",
    )
    less = analyze_proportion(
        control_successes=140,
        control_total=1000,
        treatment_successes=100,
        treatment_total=1000,
        alternative="less",
    )
    assert greater.p_value is not None and greater.p_value > 0.5
    assert less.p_value is not None and less.p_value < 0.01
    assert greater.p_value + less.p_value == pytest.approx(1.0)


def test_two_sided_swap_invariance() -> None:
    original = analyze_proportion(
        control_successes=120,
        control_total=900,
        treatment_successes=170,
        treatment_total=1100,
    )
    swapped = analyze_proportion(
        control_successes=170,
        control_total=1100,
        treatment_successes=120,
        treatment_total=900,
    )
    assert swapped.absolute_effect == pytest.approx(-original.absolute_effect)
    assert swapped.test_statistic == pytest.approx(-original.test_statistic)
    assert swapped.p_value == pytest.approx(original.p_value)
    assert swapped.confidence_interval.lower == pytest.approx(
        -original.confidence_interval.upper  # type: ignore[operator]
    )
    assert swapped.confidence_interval.upper == pytest.approx(
        -original.confidence_interval.lower  # type: ignore[operator]
    )


def test_one_sided_direction_reversal() -> None:
    greater = analyze_proportion(
        control_successes=13,
        control_total=80,
        treatment_successes=29,
        treatment_total=110,
        alternative="greater",
    )
    reversed_less = analyze_proportion(
        control_successes=29,
        control_total=110,
        treatment_successes=13,
        treatment_total=80,
        alternative="less",
    )
    assert reversed_less.test_statistic == pytest.approx(-greater.test_statistic)
    assert reversed_less.p_value == pytest.approx(greater.p_value)
    assert reversed_less.confidence_interval.upper == pytest.approx(
        -greater.confidence_interval.lower  # type: ignore[operator]
    )


@pytest.mark.parametrize("successes", [0, 10])
def test_all_failure_and_all_success_are_finite(successes: int) -> None:
    result = analyze_proportion(
        control_successes=successes,
        control_total=10,
        treatment_successes=successes,
        treatment_total=10,
    )
    assert result.test_statistic == 0.0
    assert result.p_value == 1.0
    assert any(item.code == "ASYMPTOTIC_APPROXIMATION_WARNING" for item in result.diagnostics)


def test_zero_control_rate_omits_relative_lift() -> None:
    result = analyze_proportion(
        control_successes=0,
        control_total=100,
        treatment_successes=3,
        treatment_total=100,
    )
    assert result.relative_effect is None
    assert any(item.code == "RELATIVE_EFFECT_UNDEFINED" for item in result.diagnostics)


@pytest.mark.parametrize(
    ("control_successes", "treatment_successes"),
    [(0, 10), (10, 0), (0, 5), (10, 5)],
)
def test_boundary_groups_produce_ordered_finite_two_sided_intervals(
    control_successes: int, treatment_successes: int
) -> None:
    result = analyze_proportion(
        control_successes=control_successes,
        control_total=10,
        treatment_successes=treatment_successes,
        treatment_total=10,
    )
    assert 0.0 <= result.p_value <= 1.0  # type: ignore[operator]
    assert result.confidence_interval.lower is not None
    assert result.confidence_interval.upper is not None
    assert result.confidence_interval.lower <= result.confidence_interval.upper


def test_very_large_unequal_counts_remain_finite() -> None:
    result = analyze_proportion(
        control_successes=123_456_789,
        control_total=1_000_000_000,
        treatment_successes=130_000_000,
        treatment_total=1_100_000_000,
    )
    assert result.test_statistic is not None
    assert result.p_value is not None and 0.0 <= result.p_value <= 1.0
    assert result.confidence_interval.lower is not None
    assert result.confidence_interval.upper is not None


def test_larger_proportion_sample_reduces_interval_width() -> None:
    small = analyze_proportion(
        control_successes=100,
        control_total=1000,
        treatment_successes=120,
        treatment_total=1000,
    )
    large = analyze_proportion(
        control_successes=1000,
        control_total=10000,
        treatment_successes=1200,
        treatment_total=10000,
    )
    small_width = small.confidence_interval.upper - small.confidence_interval.lower  # type: ignore[operator]
    large_width = large.confidence_interval.upper - large.confidence_interval.lower  # type: ignore[operator]
    assert large_width < small_width


@pytest.mark.parametrize(
    "kwargs",
    [
        {"control_successes": 2, "control_total": 1},
        {"control_successes": -1, "control_total": 10},
        {"control_successes": 1, "control_total": 0},
        {"treatment_successes": 1, "treatment_total": -1},
        {"treatment_successes": 11, "treatment_total": 10},
        {"confidence_level": 0.0},
        {"confidence_level": 1.0},
    ],
)
def test_invalid_proportion_inputs_fail_early(kwargs: dict[str, int | float]) -> None:
    arguments: dict[str, int | float] = {
        "control_successes": 1,
        "control_total": 10,
        "treatment_successes": 1,
        "treatment_total": 10,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        analyze_proportion(**arguments)  # type: ignore[arg-type]


def test_boolean_count_is_rejected() -> None:
    with pytest.raises(TypeError):
        analyze_proportion(
            control_successes=True,  # type: ignore[arg-type]
            control_total=10,
            treatment_successes=1,
            treatment_total=10,
        )


def test_unknown_alternative_is_rejected() -> None:
    with pytest.raises(ValueError, match="alternative"):
        analyze_proportion(
            control_successes=1,
            control_total=10,
            treatment_successes=1,
            treatment_total=10,
            alternative="up",  # type: ignore[arg-type]
        )
