import numpy as np
import pytest
from scipy import stats

from experimentation_toolkit import analyze_continuous


def test_welch_analysis_matches_scipy_reference() -> None:
    control = np.array([10.2, 9.8, 11.1, 10.4, 9.5, 10.7])
    treatment = np.array([11.5, 10.9, 12.1, 11.7, 10.8, 12.4, 11.2])
    result = analyze_continuous(control=control, treatment=treatment)
    reference = stats.ttest_ind(treatment, control, equal_var=False)
    reference_ci = reference.confidence_interval(confidence_level=0.95)
    assert result.test_statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.degrees_of_freedom == pytest.approx(reference.df)
    assert result.confidence_interval.lower == pytest.approx(reference_ci.low)
    assert result.confidence_interval.upper == pytest.approx(reference_ci.high)
    assert result.control_standard_deviation == pytest.approx(np.std(control, ddof=1))
    assert result.treatment_standard_deviation == pytest.approx(np.std(treatment, ddof=1))


def test_equal_samples_have_zero_effect_and_unit_p_value() -> None:
    values = [1.0, 2.0, 4.0, 8.0]
    result = analyze_continuous(control=values, treatment=values)
    assert result.absolute_effect == 0.0
    assert result.test_statistic == 0.0
    assert result.p_value == 1.0


def test_unequal_variances_and_sizes_use_welch() -> None:
    control = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([2.0, 8.0, 11.0, 14.0, 18.0, 22.0, 25.0])
    result = analyze_continuous(control=control, treatment=treatment)
    pooled = stats.ttest_ind(treatment, control, equal_var=True)
    assert result.p_value != pytest.approx(pooled.pvalue)
    assert "Welch" in result.method


def test_directional_continuous_p_values_use_direction() -> None:
    control = [4.0, 5.0, 6.0, 7.0]
    treatment = [1.0, 2.0, 3.0, 4.0]
    greater = analyze_continuous(control=control, treatment=treatment, alternative="greater")
    less = analyze_continuous(control=control, treatment=treatment, alternative="less")
    assert greater.p_value is not None and greater.p_value > 0.5
    assert less.p_value is not None and less.p_value < 0.05
    assert greater.p_value + less.p_value == pytest.approx(1.0)
    assert greater.confidence_interval.upper is None
    assert less.confidence_interval.lower is None


def test_two_sided_continuous_swap_invariance() -> None:
    control = [2.0, 3.0, 5.0, 7.0]
    treatment = [4.0, 6.0, 8.0, 9.0, 11.0]
    original = analyze_continuous(control=control, treatment=treatment)
    swapped = analyze_continuous(control=treatment, treatment=control)
    assert swapped.absolute_effect == pytest.approx(-original.absolute_effect)
    assert swapped.test_statistic == pytest.approx(-original.test_statistic)
    assert swapped.p_value == pytest.approx(original.p_value)
    assert swapped.confidence_interval.lower == pytest.approx(
        -original.confidence_interval.upper  # type: ignore[operator]
    )
    assert swapped.confidence_interval.upper == pytest.approx(
        -original.confidence_interval.lower  # type: ignore[operator]
    )


def test_one_sided_continuous_direction_reversal() -> None:
    control = [1.0, 3.0, 4.0, 9.0]
    treatment = [3.0, 5.0, 8.0, 10.0, 13.0]
    greater = analyze_continuous(control=control, treatment=treatment, alternative="greater")
    reversed_less = analyze_continuous(control=treatment, treatment=control, alternative="less")
    assert reversed_less.test_statistic == pytest.approx(-greater.test_statistic)
    assert reversed_less.p_value == pytest.approx(greater.p_value)
    assert reversed_less.confidence_interval.upper == pytest.approx(
        -greater.confidence_interval.lower  # type: ignore[operator]
    )


def test_nonfinite_values_raise_by_default() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        analyze_continuous(control=[1.0, np.nan, 2.0], treatment=[1.0, 2.0])


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_each_nonfinite_value_is_rejected(bad_value: float) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        analyze_continuous(control=[1.0, bad_value, 2.0], treatment=[1.0, 2.0])


def test_nonrepresentable_extreme_mean_difference_is_rejected() -> None:
    with pytest.raises(ValueError, match="mean difference"):
        analyze_continuous(control=[-1e308, -1e308], treatment=[1e308, 1e308])


def test_large_representable_magnitudes_are_scaled_stably() -> None:
    spacing = np.spacing(1e300)
    control = np.array([1e300, 1e300 + spacing, 1e300 + 2 * spacing])
    treatment = control + 4 * spacing
    result = analyze_continuous(control=control, treatment=treatment)
    assert result.absolute_effect > 0.0
    assert result.test_statistic is not None and np.isfinite(result.test_statistic)
    assert result.p_value is not None and np.isfinite(result.p_value)


def test_tiny_representable_magnitudes_do_not_underflow_welch_standard_error() -> None:
    control = np.array([1e-200, 2e-200, 4e-200])
    treatment = np.array([2e-200, 4e-200, 8e-200, 10e-200])
    result = analyze_continuous(control=control, treatment=treatment)
    assert result.control_standard_deviation is not None
    assert result.control_standard_deviation > 0.0
    assert result.test_statistic is not None and np.isfinite(result.test_statistic)
    assert result.p_value is not None and 0.0 <= result.p_value <= 1.0


def test_explicit_nonfinite_omission_is_reported() -> None:
    result = analyze_continuous(
        control=[1.0, np.nan, 2.0, np.inf],
        treatment=[1.0, 2.0, 3.0],
        nonfinite_policy="omit",
    )
    assert result.control_sample_size == 2
    diagnostic = next(
        item for item in result.diagnostics if item.code == "NONFINITE_OBSERVATIONS_OMITTED"
    )
    assert diagnostic.details["control_omitted"] == 2


def test_both_constant_samples_return_no_invalid_inference() -> None:
    result = analyze_continuous(control=[2.0, 2.0], treatment=[3.0, 3.0])
    assert result.test_statistic is None
    assert result.p_value is None
    assert result.degrees_of_freedom is None
    assert result.confidence_interval.lower == 1.0
    assert result.confidence_interval.upper == 1.0
    assert any(item.code == "DEGENERATE_ZERO_VARIANCE" for item in result.diagnostics)


def test_all_zero_samples_follow_explicit_zero_standard_error_policy() -> None:
    result = analyze_continuous(control=[0.0, 0.0], treatment=[0.0, 0.0])
    assert result.absolute_effect == 0.0
    assert result.test_statistic is None
    assert result.p_value is None
    assert result.relative_effect is None


def test_one_constant_sample_is_computable_and_warned() -> None:
    result = analyze_continuous(control=[2.0, 2.0, 2.0], treatment=[1.0, 2.0, 4.0])
    assert result.p_value is not None
    assert any(item.code == "ONE_ZERO_VARIANCE_SAMPLE" for item in result.diagnostics)


def test_nonpositive_control_mean_omits_relative_effect() -> None:
    result = analyze_continuous(control=[-2.0, -1.0], treatment=[1.0, 2.0])
    assert result.relative_effect is None


def test_sign_crossing_omits_relative_effect() -> None:
    result = analyze_continuous(control=[1.0, 2.0], treatment=[-2.0, -1.0])
    assert result.relative_effect is None


def test_cancellation_near_zero_control_mean_omits_relative_effect() -> None:
    result = analyze_continuous(
        control=[-1.0, 1.0 + np.finfo(float).eps],
        treatment=[1.0, 2.0],
    )
    assert result.control_estimate > 0.0
    assert result.relative_effect is None


def test_genuinely_small_ratio_scale_values_can_report_relative_effect() -> None:
    result = analyze_continuous(
        control=[1e-12, 2e-12, 3e-12],
        treatment=[2e-12, 3e-12, 4e-12],
    )
    assert result.relative_effect == pytest.approx(0.5)


def test_larger_continuous_sample_reduces_interval_width() -> None:
    control = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([2.0, 3.0, 4.0, 5.0])
    small = analyze_continuous(control=control, treatment=treatment)
    large = analyze_continuous(control=np.tile(control, 10), treatment=np.tile(treatment, 10))
    small_width = small.confidence_interval.upper - small.confidence_interval.lower  # type: ignore[operator]
    large_width = large.confidence_interval.upper - large.confidence_interval.lower  # type: ignore[operator]
    assert large_width < small_width


@pytest.mark.parametrize(
    ("control", "treatment"),
    [([1.0], [1.0, 2.0]), ([[1.0, 2.0]], [1.0, 2.0]), (["x", "y"], [1.0, 2.0])],
)
def test_invalid_continuous_samples_fail(control: object, treatment: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        analyze_continuous(control=control, treatment=treatment)  # type: ignore[arg-type]


def test_invalid_nonfinite_policy_fails() -> None:
    with pytest.raises(ValueError, match="nonfinite_policy"):
        analyze_continuous(
            control=[1.0, 2.0],
            treatment=[1.0, 2.0],
            nonfinite_policy="bad",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("confidence_level", [0.0, 1.0, np.nan])
def test_invalid_continuous_confidence_level_fails(confidence_level: float) -> None:
    with pytest.raises(ValueError, match="confidence_level"):
        analyze_continuous(
            control=[1.0, 2.0],
            treatment=[1.0, 2.0],
            confidence_level=confidence_level,
        )
