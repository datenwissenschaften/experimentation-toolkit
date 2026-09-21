import numpy as np
import pytest
from scipy import stats

from experimentation_toolkit import proportion_achieved_power, proportion_sample_size


def test_absolute_mde_sample_size_reaches_target_power() -> None:
    result = proportion_sample_size(
        baseline_rate=0.10,
        minimum_detectable_effect=0.01,
        mde_type="absolute",
        power=0.80,
    )
    assert result.treatment_rate == pytest.approx(0.11)
    assert result.control_sample_size == 14_751
    assert result.treatment_sample_size == 14_751
    assert result.achieved_power == pytest.approx(0.8000065290879333)
    assert result.achieved_power >= 0.80
    previous = proportion_achieved_power(
        baseline_rate=0.10,
        treatment_rate=0.11,
        control_sample_size=result.control_sample_size - 1,
        treatment_sample_size=result.treatment_sample_size - 1,
    )
    assert previous.power < 0.80


def test_relative_mde_is_explicit_fraction_of_baseline() -> None:
    result = proportion_sample_size(
        baseline_rate=0.10,
        minimum_detectable_effect=0.10,
        mde_type="relative",
    )
    assert result.treatment_rate == pytest.approx(0.11)


def test_less_alternative_subtracts_positive_mde_magnitude() -> None:
    result = proportion_sample_size(
        baseline_rate=0.20,
        minimum_detectable_effect=0.02,
        mde_type="absolute",
        alternative="less",
    )
    assert result.treatment_rate == pytest.approx(0.18)
    assert result.achieved_power >= result.target_power


def test_allocation_ratio_is_treatment_over_control() -> None:
    result = proportion_sample_size(
        baseline_rate=0.2,
        minimum_detectable_effect=0.03,
        mde_type="absolute",
        allocation_ratio=2.0,
    )
    assert result.treatment_sample_size == 2 * result.control_sample_size


def test_power_increases_with_sample_size() -> None:
    small = proportion_achieved_power(
        baseline_rate=0.1,
        treatment_rate=0.12,
        control_sample_size=500,
        treatment_sample_size=500,
    )
    large = proportion_achieved_power(
        baseline_rate=0.1,
        treatment_rate=0.12,
        control_sample_size=5000,
        treatment_sample_size=5000,
    )
    assert large.power > small.power


def test_achieved_power_matches_independent_normal_reference() -> None:
    baseline = 0.12
    treatment = 0.145
    control_size = 1400
    treatment_size = 2100
    alpha = 0.05
    result = proportion_achieved_power(
        baseline_rate=baseline,
        treatment_rate=treatment,
        control_sample_size=control_size,
        treatment_sample_size=treatment_size,
        alpha=alpha,
    )
    pooled = (control_size * baseline + treatment_size * treatment) / (
        control_size + treatment_size
    )
    null_se = (pooled * (1 - pooled) * (1 / control_size + 1 / treatment_size)) ** 0.5
    alternative_se = (
        baseline * (1 - baseline) / control_size + treatment * (1 - treatment) / treatment_size
    ) ** 0.5
    mean = (treatment - baseline) / null_se
    scale = alternative_se / null_se
    critical = stats.norm.ppf(1 - alpha / 2)
    expected = stats.norm.sf((critical - mean) / scale) + stats.norm.cdf((-critical - mean) / scale)
    assert result.power == pytest.approx(expected)


def test_wrong_direction_has_low_one_sided_power() -> None:
    result = proportion_achieved_power(
        baseline_rate=0.2,
        treatment_rate=0.18,
        control_sample_size=1000,
        treatment_sample_size=1000,
        alternative="greater",
    )
    assert result.power < 0.05


def test_less_achieved_power_uses_lower_tail() -> None:
    less = proportion_achieved_power(
        baseline_rate=0.2,
        treatment_rate=0.18,
        control_sample_size=1000,
        treatment_sample_size=1000,
        alternative="less",
    )
    greater = proportion_achieved_power(
        baseline_rate=0.2,
        treatment_rate=0.18,
        control_sample_size=1000,
        treatment_sample_size=1000,
        alternative="greater",
    )
    assert less.power == pytest.approx(0.3067647429536998)
    assert less.power > greater.power


@pytest.mark.parametrize(
    ("baseline", "mde", "alternative"),
    [(0.0001, 0.00005, "greater"), (0.9999, 0.00005, "less")],
)
def test_sample_size_near_probability_boundaries(
    baseline: float, mde: float, alternative: str
) -> None:
    result = proportion_sample_size(
        baseline_rate=baseline,
        minimum_detectable_effect=mde,
        mde_type="absolute",
        alternative=alternative,
    )
    assert result.achieved_power >= result.target_power
    assert 0.0 < result.treatment_rate < 1.0


def test_unequal_allocation_result_is_locally_minimal() -> None:
    result = proportion_sample_size(
        baseline_rate=0.2,
        minimum_detectable_effect=0.025,
        mde_type="absolute",
        allocation_ratio=1.7,
    )
    previous_control = result.control_sample_size - 1
    previous_treatment = max(2, int(np.ceil(1.7 * previous_control)))
    previous = proportion_achieved_power(
        baseline_rate=0.2,
        treatment_rate=0.225,
        control_sample_size=previous_control,
        treatment_sample_size=previous_treatment,
    )
    assert result.achieved_power >= result.target_power
    assert previous.power < result.target_power


def test_low_target_power_returns_smallest_supported_design_quickly() -> None:
    result = proportion_sample_size(
        baseline_rate=0.2,
        minimum_detectable_effect=0.02,
        mde_type="absolute",
        power=0.01,
    )
    assert result.control_sample_size == 2
    assert result.treatment_sample_size == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_rate": 0.0},
        {"baseline_rate": float("nan")},
        {"minimum_detectable_effect": 0.0},
        {"minimum_detectable_effect": float("nan")},
        {"allocation_ratio": 0.0},
        {"allocation_ratio": float("inf")},
        {"alpha": 0.0},
        {"power": 1.0},
        {"minimum_detectable_effect": 1.0},
    ],
)
def test_invalid_sample_size_domains(kwargs: dict[str, float]) -> None:
    arguments = {
        "baseline_rate": 0.1,
        "minimum_detectable_effect": 0.01,
        "mde_type": "absolute",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        proportion_sample_size(**arguments)  # type: ignore[arg-type]


def test_unknown_mde_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="mde_type"):
        proportion_sample_size(
            baseline_rate=0.1,
            minimum_detectable_effect=0.01,
            mde_type="points",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum_detectable_effect": 1e-300},
        {"allocation_ratio": 1e16},
    ],
)
def test_unrepresentable_sample_size_design_is_rejected(kwargs: dict[str, float]) -> None:
    arguments = {
        "baseline_rate": 0.1,
        "minimum_detectable_effect": 0.01,
        "mde_type": "absolute",
    }
    arguments.update(kwargs)
    with pytest.raises(ValueError, match="float64 precision"):
        proportion_sample_size(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_rate": 0.0},
        {"treatment_rate": 1.0},
        {"control_sample_size": 0},
        {"treatment_sample_size": 0},
        {"alpha": 1.0},
        {"alternative": "up"},
    ],
)
def test_invalid_achieved_power_domains(kwargs: dict[str, object]) -> None:
    arguments: dict[str, object] = {
        "baseline_rate": 0.1,
        "treatment_rate": 0.11,
        "control_sample_size": 100,
        "treatment_sample_size": 100,
    }
    arguments.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        proportion_achieved_power(**arguments)  # type: ignore[arg-type]
