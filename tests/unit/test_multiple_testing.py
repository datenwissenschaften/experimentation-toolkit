import pytest
from scipy import stats

from experimentation_toolkit import CorrectionMethod, adjust_p_values


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        (CorrectionMethod.BONFERRONI, (0.04, 0.16, 0.12, 0.008)),
        (CorrectionMethod.HOLM, (0.03, 0.06, 0.06, 0.008)),
        (CorrectionMethod.BENJAMINI_HOCHBERG, (0.02, 0.04, 0.04, 0.008)),
    ],
)
def test_reference_adjusted_p_values(method: CorrectionMethod, expected: tuple[float, ...]) -> None:
    result = adjust_p_values([0.01, 0.04, 0.03, 0.002], method=method)
    assert result.adjusted_p_values == pytest.approx(expected)


def test_original_order_and_labels_are_preserved() -> None:
    labels = ("third", "first", "second")
    result = adjust_p_values([0.8, 0.001, 0.03], method="holm", hypothesis_labels=labels)
    assert result.original_p_values == (0.8, 0.001, 0.03)
    assert result.hypothesis_labels == labels
    assert result.adjusted_p_values[1] < result.adjusted_p_values[2] < result.adjusted_p_values[0]


@pytest.mark.parametrize("method", list(CorrectionMethod))
def test_adjusted_values_are_probabilities(method: CorrectionMethod) -> None:
    result = adjust_p_values([0.0, 0.2, 0.9, 1.0], method=method)
    assert all(0.0 <= value <= 1.0 for value in result.adjusted_p_values)
    assert result.rejected == tuple(value <= result.alpha for value in result.adjusted_p_values)


def test_bh_reports_fdr_not_fwer() -> None:
    result = adjust_p_values([0.01, 0.2], method="benjamini-hochberg")
    assert result.error_rate_control == "false discovery rate"


def test_bh_matches_scipy_reference_for_unsorted_duplicates() -> None:
    p_values = [0.6, 0.01, 0.01, 1.0, 0.0, 0.2]
    result = adjust_p_values(p_values, method="benjamini-hochberg")
    reference = stats.false_discovery_control(p_values, method="bh")
    assert result.adjusted_p_values == pytest.approx(reference)
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    sorted_adjusted = [result.adjusted_p_values[index] for index in order]
    assert sorted_adjusted == sorted(sorted_adjusted)


@pytest.mark.parametrize("method", list(CorrectionMethod))
def test_single_hypothesis_is_unchanged(method: CorrectionMethod) -> None:
    result = adjust_p_values([0.03], method=method, hypothesis_labels=["only"])
    assert result.adjusted_p_values == pytest.approx((0.03,))
    assert result.hypothesis_labels == ("only",)


@pytest.mark.parametrize("p_values", [[], [-0.1], [1.1], [float("nan")], [float("inf")]])
def test_invalid_p_values_are_rejected(p_values: list[float]) -> None:
    with pytest.raises(ValueError):
        adjust_p_values(p_values, method="holm")


def test_label_length_must_match() -> None:
    with pytest.raises(ValueError):
        adjust_p_values([0.1, 0.2], method="holm", hypothesis_labels=["only one"])


def test_empty_label_and_unknown_method_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        adjust_p_values([0.1], method="holm", hypothesis_labels=[""])
    with pytest.raises(ValueError, match="method must be"):
        adjust_p_values([0.1], method="unknown")
