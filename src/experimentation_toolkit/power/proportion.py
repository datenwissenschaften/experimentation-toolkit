"""Normal-approximation power and sample size for independent proportions."""

from math import ceil, sqrt

import numpy as np
from scipy import stats

from experimentation_toolkit.models.enums import AlternativeHypothesis, MDEType
from experimentation_toolkit.models.results import AchievedPowerResult, ProportionSampleSizeResult
from experimentation_toolkit.validation.inputs import (
    normalize_alternative,
    validate_positive_integer,
    validate_probability,
)

_METHOD = "Normal approximation for a pooled-null two-sample proportion score test"


def _normal_approximation_power(
    baseline_rate: float,
    treatment_rate: float,
    control_size: int,
    treatment_size: int,
    alpha: float,
    alternative: AlternativeHypothesis,
) -> float:
    pooled_rate = (control_size * baseline_rate + treatment_size * treatment_rate) / (
        control_size + treatment_size
    )
    null_standard_error = sqrt(
        pooled_rate * (1.0 - pooled_rate) * (1.0 / control_size + 1.0 / treatment_size)
    )
    alternative_standard_error = sqrt(
        baseline_rate * (1.0 - baseline_rate) / control_size
        + treatment_rate * (1.0 - treatment_rate) / treatment_size
    )
    if null_standard_error == 0.0 or alternative_standard_error == 0.0:
        raise ValueError("power approximation is undefined at degenerate boundary rates")
    mean = (treatment_rate - baseline_rate) / null_standard_error
    scale = alternative_standard_error / null_standard_error
    if alternative is AlternativeHypothesis.TWO_SIDED:
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        power = stats.norm.sf((critical - mean) / scale) + stats.norm.cdf(
            (-critical - mean) / scale
        )
    elif alternative is AlternativeHypothesis.GREATER:
        critical = float(stats.norm.ppf(1.0 - alpha))
        power = stats.norm.sf((critical - mean) / scale)
    else:
        critical = float(stats.norm.ppf(alpha))
        power = stats.norm.cdf((critical - mean) / scale)
    return float(np.clip(power, 0.0, 1.0))


def proportion_achieved_power(
    *,
    baseline_rate: float,
    treatment_rate: float,
    control_sample_size: int,
    treatment_sample_size: int,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis | str = AlternativeHypothesis.TWO_SIDED,
) -> AchievedPowerResult:
    """Estimate achieved power for a fixed two-proportion design."""
    baseline_rate = validate_probability(baseline_rate, "baseline_rate", open_interval=True)
    treatment_rate = validate_probability(treatment_rate, "treatment_rate", open_interval=True)
    control_sample_size = validate_positive_integer(control_sample_size, "control_sample_size")
    treatment_sample_size = validate_positive_integer(
        treatment_sample_size, "treatment_sample_size"
    )
    alpha = validate_probability(alpha, "alpha", open_interval=True)
    alternative = normalize_alternative(alternative)
    power = _normal_approximation_power(
        baseline_rate,
        treatment_rate,
        control_sample_size,
        treatment_sample_size,
        alpha,
        alternative,
    )
    return AchievedPowerResult(
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
        control_sample_size=control_sample_size,
        treatment_sample_size=treatment_sample_size,
        alpha=alpha,
        power=power,
        alternative=alternative,
        method=_METHOD,
    )


def proportion_sample_size(
    *,
    baseline_rate: float,
    minimum_detectable_effect: float,
    mde_type: MDEType | str,
    alpha: float = 0.05,
    power: float = 0.8,
    allocation_ratio: float = 1.0,
    alternative: AlternativeHypothesis | str = AlternativeHypothesis.TWO_SIDED,
) -> ProportionSampleSizeResult:
    """Plan group sizes for a two-proportion experiment.

    ``allocation_ratio`` is treatment/control. The MDE is a positive magnitude:
    it is added to the baseline for two-sided/greater alternatives and subtracted
    for a less alternative. A relative MDE is a fraction of the baseline (``0.1``
    means a 10% relative change), never percentage points.
    """
    baseline_rate = validate_probability(baseline_rate, "baseline_rate", open_interval=True)
    effect = float(minimum_detectable_effect)
    if not np.isfinite(effect) or effect <= 0.0:
        raise ValueError("minimum_detectable_effect must be finite and greater than zero")
    try:
        mde_type = MDEType(mde_type)
    except ValueError as error:
        raise ValueError("mde_type must be 'absolute' or 'relative'") from error
    alpha = validate_probability(alpha, "alpha", open_interval=True)
    power = validate_probability(power, "power", open_interval=True)
    allocation_ratio = float(allocation_ratio)
    if not np.isfinite(allocation_ratio) or allocation_ratio <= 0.0:
        raise ValueError("allocation_ratio must be finite and greater than zero")
    alternative = normalize_alternative(alternative)

    absolute_effect = effect if mde_type is MDEType.ABSOLUTE else baseline_rate * effect
    sign = -1.0 if alternative is AlternativeHypothesis.LESS else 1.0
    treatment_rate = baseline_rate + sign * absolute_effect
    if not 0.0 < treatment_rate < 1.0:
        raise ValueError("the MDE implies a treatment rate outside the open interval (0, 1)")

    pooled_rate = (baseline_rate + allocation_ratio * treatment_rate) / (1.0 + allocation_ratio)
    null_variance_factor = pooled_rate * (1.0 - pooled_rate) * (1.0 + 1.0 / allocation_ratio)
    alternative_variance_factor = baseline_rate * (1.0 - baseline_rate) + (
        treatment_rate * (1.0 - treatment_rate) / allocation_ratio
    )
    alpha_quantile = float(
        stats.norm.ppf(
            1.0 - alpha / 2.0 if alternative is AlternativeHypothesis.TWO_SIDED else 1.0 - alpha
        )
    )
    power_quantile = float(stats.norm.ppf(power))
    initial_control_size = ceil(
        (
            alpha_quantile * sqrt(null_variance_factor)
            + power_quantile * sqrt(alternative_variance_factor)
        )
        ** 2
        / absolute_effect**2
    )
    control_size = max(2, initial_control_size)
    treatment_size = max(2, ceil(allocation_ratio * control_size))
    achieved = _normal_approximation_power(
        baseline_rate,
        treatment_rate,
        control_size,
        treatment_size,
        alpha,
        alternative,
    )
    while achieved < power:
        control_size += 1
        treatment_size = max(2, ceil(allocation_ratio * control_size))
        achieved = _normal_approximation_power(
            baseline_rate,
            treatment_rate,
            control_size,
            treatment_size,
            alpha,
            alternative,
        )

    return ProportionSampleSizeResult(
        baseline_rate=baseline_rate,
        treatment_rate=treatment_rate,
        minimum_detectable_effect=effect,
        mde_type=mde_type,
        alpha=alpha,
        target_power=power,
        achieved_power=achieved,
        allocation_ratio=allocation_ratio,
        alternative=alternative,
        control_sample_size=control_size,
        treatment_sample_size=treatment_size,
        total_sample_size=control_size + treatment_size,
        method=_METHOD,
    )
