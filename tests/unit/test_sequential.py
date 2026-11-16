import numpy as np
import numpy.typing as npt
import pytest
from pydantic import ValidationError
from scipy import stats

from experimentation_toolkit import (
    AlternativeHypothesis,
    SequentialTestResult,
    SpendingFunction,
    alpha_spent,
    analyze_sequential_continuous,
    group_sequential_boundaries,
)

FIVE_LOOKS = (0.2, 0.4, 0.6, 0.8, 1.0)
HOURLY_LOOKS = tuple(look / 72 for look in range(1, 73))


def crossing_rate(
    fractions: tuple[float, ...], boundaries: tuple[float, ...], seed: int, size: int
) -> float:
    information = np.asarray(fractions)
    increments = np.diff(np.concatenate(([0.0], information)))
    rng = np.random.default_rng(seed)
    scores = np.cumsum(rng.standard_normal((size, information.size)) * np.sqrt(increments), 1)
    statistics = scores / np.sqrt(information)
    return float(np.mean(np.any(statistics >= np.asarray(boundaries), axis=1)))


def monitor(
    control: npt.ArrayLike,
    treatment: npt.ArrayLike,
    treatment_looks: list[int],
    alternative: str,
) -> SequentialTestResult:
    control_size = len(np.asarray(control))
    return analyze_sequential_continuous(
        control=control,
        treatment=treatment,
        control_looks=[control_size] * len(treatment_looks),
        treatment_looks=treatment_looks,
        maximum_control_size=control_size,
        maximum_treatment_size=treatment_looks[-1],
        alternative=alternative,
        metric_name="frames_saved_per_hour",
    )


@pytest.mark.parametrize(
    ("spending", "expected"),
    [
        (SpendingFunction.O_BRIEN_FLEMING, (4.877, 3.357, 2.680, 2.290, 2.031)),
        (SpendingFunction.POCOCK, (2.438, 2.427, 2.410, 2.397, 2.386)),
    ],
)
def test_boundaries_match_published_lan_demets_values(
    spending: SpendingFunction, expected: tuple[float, ...]
) -> None:
    boundaries = group_sequential_boundaries(FIVE_LOOKS, alpha=0.025, spending=spending)
    assert boundaries == pytest.approx(expected, abs=2e-3)


@pytest.mark.parametrize("alpha", [0.01, 0.025, 0.05, 0.2])
def test_a_single_final_look_is_the_fixed_horizon_test(alpha: float) -> None:
    assert group_sequential_boundaries([1.0], alpha=alpha) == pytest.approx(
        (stats.norm.isf(alpha),)
    )


def test_a_single_interim_look_spends_its_own_alpha() -> None:
    spent = alpha_spent(0.5, alpha=0.05, spending="pocock")
    boundary = group_sequential_boundaries([0.5], alpha=0.05, spending="pocock")
    assert boundary == pytest.approx((stats.norm.isf(spent),))


@pytest.mark.parametrize(
    ("fractions", "spending", "alpha"),
    [
        (HOURLY_LOOKS, SpendingFunction.O_BRIEN_FLEMING, 0.05),
        (HOURLY_LOOKS, SpendingFunction.POCOCK, 0.05),
        ((0.05, 0.1, 0.5, 0.51, 0.52, 1.0), SpendingFunction.O_BRIEN_FLEMING, 0.1),
    ],
)
def test_null_crossing_probability_equals_alpha(
    fractions: tuple[float, ...], spending: SpendingFunction, alpha: float
) -> None:
    boundaries = group_sequential_boundaries(fractions, alpha=alpha, spending=spending)
    rate = crossing_rate(fractions, boundaries, seed=11, size=200_000)
    assert rate == pytest.approx(alpha, abs=4 * np.sqrt(alpha * (1 - alpha) / 200_000))


def test_obrien_fleming_boundaries_fall_and_stay_finite_for_tiny_increments() -> None:
    boundaries = group_sequential_boundaries([0.001, 0.03, 0.5, 0.5000001, 1.0], alpha=0.05)
    assert all(np.isfinite(boundaries))
    assert boundaries[0] > 40.0
    assert boundaries[0] > boundaries[1] > boundaries[2] > boundaries[4]


def test_alpha_spending_endpoints_and_shapes() -> None:
    for spending in SpendingFunction:
        assert alpha_spent(0.0, alpha=0.05, spending=spending) == 0.0
        assert alpha_spent(1.0, alpha=0.05, spending=spending) == 0.05
        spent = [alpha_spent(fraction, alpha=0.05, spending=spending) for fraction in FIVE_LOOKS]
        assert spent == sorted(spent)
    pocock = alpha_spent(0.3, alpha=0.05, spending="pocock")
    assert pocock == pytest.approx(0.05 * np.log(1 + (np.e - 1) * 0.3))
    obrien = alpha_spent(0.3, alpha=0.05, spending="obrien-fleming")
    assert obrien == pytest.approx(2 - 2 * stats.norm.cdf(stats.norm.isf(0.025) / np.sqrt(0.3)))


def test_clear_harm_crosses_early_and_stops_monitoring() -> None:
    rng = np.random.default_rng(3)
    result = monitor(rng.normal(100, 20, 72), rng.normal(40, 20, 30), list(range(2, 31)), "less")
    assert result.crossed
    assert result.crossed_at_look is not None
    assert result.crossed_at_look < 28
    assert len(result.z_statistics) == result.crossed_at_look + 1
    assert result.z_statistics[-1] <= result.boundaries[-1] < 0.0
    assert result.alternative is AlternativeHypothesis.LESS
    assert result.metric_name == "frames_saved_per_hour"


def test_a_benefit_crosses_the_upper_boundary() -> None:
    rng = np.random.default_rng(5)
    result = monitor(rng.normal(0, 1, 50), rng.normal(2, 1, 50), [10, 20, 30, 40, 50], "greater")
    assert result.crossed
    assert result.z_statistics[-1] >= result.boundaries[-1] > 0.0


def test_no_effect_monitors_every_look_without_crossing() -> None:
    control = [3.0, 5.0, 4.0, 6.0, 5.0, 4.0, 3.0, 5.0]
    treatment = [4.0, 5.0, 3.0, 6.0, 4.0, 5.0, 4.0, 5.0]
    result = monitor(control, treatment, [4, 6, 8], "less")
    assert not result.crossed
    assert result.crossed_at_look is None
    assert len(result.z_statistics) == 3
    assert result.information_fractions[-1] == pytest.approx(1.0)
    assert result.cumulative_alpha_spent[-1] == pytest.approx(0.05)


def test_look_statistics_match_welch_tail_probabilities() -> None:
    control = np.array([10.2, 9.8, 11.1, 10.4, 9.5, 10.7])
    treatment = np.array([9.5, 9.9, 9.1, 9.7, 10.8, 9.2, 9.4])
    result = monitor(control, treatment, [4, 7], "less")
    for size, statistic in zip((4, 7), result.z_statistics, strict=False):
        reference = stats.ttest_ind(treatment[:size], control, equal_var=False, alternative="less")
        assert stats.norm.cdf(statistic) == pytest.approx(reference.pvalue)


def test_null_monitoring_keeps_its_error_rate() -> None:
    rng = np.random.default_rng(13)
    simulations = 300
    crossed = sum(
        monitor(rng.normal(0, 1, 20), rng.normal(0, 1, 20), [5, 10, 15, 20], "greater").crossed
        for _ in range(simulations)
    )
    assert 0.02 < crossed / simulations < 0.09


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"alternative": "two-sided"}, "one-sided"),
        ({"alpha": 0.5}, "below 0.5"),
        ({"spending": "haybittle"}, "spending must be one of"),
        ({"treatment_looks": [], "control_looks": []}, "must not be empty"),
        ({"treatment_looks": [1, 6]}, "at least two"),
        ({"treatment_looks": [5, 3, 6]}, "must not decrease"),
        ({"treatment_looks": [3, 5]}, "cover all"),
        ({"maximum_treatment_size": 5}, "maximum size"),
        ({"control_looks": [6, 6, 6]}, "same length"),
        ({"control": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], "treatment": [2.0] * 6}, "zero variance"),
    ],
)
def test_invalid_monitoring_inputs_are_rejected(changes: dict[str, object], message: str) -> None:
    arguments: dict[str, object] = {
        "control": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "treatment": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        "control_looks": [6, 6],
        "treatment_looks": [3, 6],
        "maximum_control_size": 6,
        "maximum_treatment_size": 6,
        "alternative": "less",
    }
    with pytest.raises(ValueError, match=message):
        analyze_sequential_continuous(**{**arguments, **changes})  # type: ignore[arg-type]


def test_information_must_grow_between_looks() -> None:
    with pytest.raises(ValueError, match="information must increase"):
        monitor([1.0, 2.0, 3.0, 4.0], [5.0, 5.0, 5.0, 5.0], [3, 4], "less")


def test_a_look_without_variance_is_rejected() -> None:
    with pytest.raises(ValueError, match="zero variance in both arms"):
        analyze_sequential_continuous(
            control=[1.0, 1.0, 2.0, 3.0],
            treatment=[1.0, 1.0, 4.0, 6.0],
            control_looks=[2, 4],
            treatment_looks=[2, 4],
            maximum_control_size=4,
            maximum_treatment_size=4,
            alternative="greater",
        )


@pytest.mark.parametrize(
    ("fractions", "message"),
    [
        ([], "must not be empty"),
        ([0.0, 1.0], r"in \(0, 1\]"),
        ([0.5, 1.2], r"in \(0, 1\]"),
        ([0.5, 0.5], "increase strictly"),
        ([float("nan")], "finite"),
    ],
)
def test_invalid_information_fractions_are_rejected(fractions: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        group_sequential_boundaries(fractions)


def test_invalid_spending_arguments_are_rejected() -> None:
    with pytest.raises(ValueError, match="information_fraction"):
        alpha_spent(1.5, alpha=0.05, spending="pocock")
    with pytest.raises(ValueError, match="alpha"):
        alpha_spent(0.5, alpha=0.0, spending="pocock")


def result_fields() -> dict[str, object]:
    return {
        "alternative": "less",
        "spending": "obrien-fleming",
        "alpha": 0.05,
        "information_fractions": (0.5, 1.0),
        "z_statistics": (-1.0, -2.0),
        "boundaries": (-2.5, -1.9),
        "cumulative_alpha_spent": (0.005, 0.05),
        "crossed": True,
        "crossed_at_look": 1,
        "method": "test",
    }


def test_a_coherent_result_validates() -> None:
    result = SequentialTestResult(**result_fields())  # type: ignore[arg-type]
    assert result.crossed_at_look == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"information_fractions": (), "z_statistics": (), "boundaries": ()},
        {"z_statistics": (-1.0,)},
        {"alternative": "two-sided"},
        {"information_fractions": (0.0, 1.0)},
        {"information_fractions": (0.6, 0.5)},
        {"cumulative_alpha_spent": (0.05, 0.01)},
        {"cumulative_alpha_spent": (0.005, 0.2)},
        {"crossed": False, "crossed_at_look": None},
        {"z_statistics": (-3.0, -2.0)},
        {"crossed_at_look": 0},
    ],
)
def test_incoherent_results_are_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SequentialTestResult(**{**result_fields(), **changes})  # type: ignore[arg-type]
