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


def test_nonfinite_values_raise_by_default() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        analyze_continuous(control=[1.0, np.nan, 2.0], treatment=[1.0, 2.0])


def test_overflowing_sample_moments_are_rejected() -> None:
    with pytest.raises(ValueError, match="sample moments"):
        analyze_continuous(control=[1e308, -1e308], treatment=[1.0, 2.0])


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


def test_one_constant_sample_is_computable_and_warned() -> None:
    result = analyze_continuous(control=[2.0, 2.0, 2.0], treatment=[1.0, 2.0, 4.0])
    assert result.p_value is not None
    assert any(item.code == "ONE_ZERO_VARIANCE_SAMPLE" for item in result.diagnostics)


def test_nonpositive_control_mean_omits_relative_effect() -> None:
    result = analyze_continuous(control=[-2.0, -1.0], treatment=[1.0, 2.0])
    assert result.relative_effect is None


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
