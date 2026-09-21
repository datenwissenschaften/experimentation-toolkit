import pytest
from scipy import stats

from experimentation_toolkit import DiagnosticStatus, check_sample_ratio_mismatch


def test_srm_matches_scipy_reference_and_fails() -> None:
    result = check_sample_ratio_mismatch(
        observed_counts={"control": 48_000, "treatment": 52_000},
        expected_proportions={"control": 0.5, "treatment": 0.5},
    )
    reference = stats.chisquare([48_000, 52_000], f_exp=[50_000, 50_000])
    assert result.chi_square_statistic == pytest.approx(160.0)
    assert result.chi_square_statistic == pytest.approx(reference.statistic)
    assert result.p_value == pytest.approx(reference.pvalue)
    assert result.degrees_of_freedom == 1
    assert result.status is DiagnosticStatus.FAIL


def test_balanced_assignment_passes() -> None:
    result = check_sample_ratio_mismatch(
        observed_counts={"control": 500, "treatment": 500},
        expected_proportions={"control": 0.5, "treatment": 0.5},
    )
    assert result.p_value == 1.0
    assert result.status is DiagnosticStatus.PASS


def test_unequal_expected_allocation() -> None:
    result = check_sample_ratio_mismatch(
        observed_counts={"control": 250, "treatment": 750},
        expected_proportions={"control": 0.25, "treatment": 0.75},
    )
    assert result.expected_counts == {"control": 250.0, "treatment": 750.0}
    assert result.status is DiagnosticStatus.PASS


def test_three_group_srm_has_two_degrees_of_freedom() -> None:
    result = check_sample_ratio_mismatch(
        observed_counts={"a": 200, "b": 300, "c": 500},
        expected_proportions={"a": 0.2, "b": 0.3, "c": 0.5},
    )
    assert result.degrees_of_freedom == 2
    assert result.p_value == 1.0


def test_zero_observed_count_is_valid_when_expected_counts_are_adequate() -> None:
    result = check_sample_ratio_mismatch(
        observed_counts={"control": 0, "treatment": 100},
        expected_proportions={"control": 0.5, "treatment": 0.5},
    )
    assert result.status is DiagnosticStatus.FAIL


def test_small_expected_count_returns_approximation_diagnostic() -> None:
    result = check_sample_ratio_mismatch(
        observed_counts={"control": 1, "treatment": 9},
        expected_proportions={"control": 0.1, "treatment": 0.9},
    )
    assert result.expected_counts == {"control": 1.0, "treatment": 9.0}
    assert any(item.code == "SMALL_EXPECTED_ASSIGNMENT_COUNT" for item in result.diagnostics)


def test_p_value_equal_to_threshold_passes() -> None:
    reference = check_sample_ratio_mismatch(
        observed_counts={"control": 45, "treatment": 55},
        expected_proportions={"control": 0.5, "treatment": 0.5},
    )
    result = check_sample_ratio_mismatch(
        observed_counts={"control": 45, "treatment": 55},
        expected_proportions={"control": 0.5, "treatment": 0.5},
        significance_level=reference.p_value,
    )
    assert result.status is DiagnosticStatus.PASS


@pytest.mark.parametrize(
    ("observed", "expected"),
    [
        ({"control": 10}, {"control": 1.0}),
        ({"control": 10, "treatment": 10}, {"control": 0.5}),
        ({"control": -1, "treatment": 11}, {"control": 0.5, "treatment": 0.5}),
        ({"control": 10, "treatment": 10}, {"control": 0.4, "treatment": 0.4}),
        ({"control": 10, "treatment": 10}, {"control": 0.0, "treatment": 1.0}),
        ({"control": 0, "treatment": 0}, {"control": 0.5, "treatment": 0.5}),
    ],
)
def test_invalid_srm_inputs(observed: dict[str, int], expected: dict[str, float]) -> None:
    with pytest.raises((TypeError, ValueError)):
        check_sample_ratio_mismatch(observed_counts=observed, expected_proportions=expected)


def test_negative_proportion_tolerance_is_rejected() -> None:
    with pytest.raises(ValueError, match="proportion_tolerance"):
        check_sample_ratio_mismatch(
            observed_counts={"control": 10, "treatment": 10},
            expected_proportions={"control": 0.5, "treatment": 0.5},
            proportion_tolerance=-1.0,
        )


@pytest.mark.parametrize("significance_level", [0.0, 1.0, float("nan")])
def test_invalid_srm_significance_level_is_rejected(significance_level: float) -> None:
    with pytest.raises(ValueError, match="significance_level"):
        check_sample_ratio_mismatch(
            observed_counts={"control": 10, "treatment": 10},
            expected_proportions={"control": 0.5, "treatment": 0.5},
            significance_level=significance_level,
        )
