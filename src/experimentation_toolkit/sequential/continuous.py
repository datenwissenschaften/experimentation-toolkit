"""Group-sequential monitoring of a continuous treatment-minus-control mean difference."""

from collections.abc import Sequence
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from scipy import special, stats

from experimentation_toolkit.analysis.continuous import _welch_degrees_of_freedom
from experimentation_toolkit.models.enums import AlternativeHypothesis, SpendingFunction
from experimentation_toolkit.models.results import SequentialTestResult
from experimentation_toolkit.sequential.boundaries import (
    alpha_spent,
    group_sequential_boundaries,
    normalize_spending,
    validate_sequential_alpha,
)
from experimentation_toolkit.validation.inputs import (
    normalize_alternative,
    normalize_numeric_sample,
    validate_positive_integer,
)

_MINIMUM_LOOK_SIZE = 2


def _looks(looks: Sequence[int], name: str, observed: int, maximum: int) -> npt.NDArray[np.int64]:
    sizes = np.asarray([validate_positive_integer(size, name) for size in looks], dtype=np.int64)
    if sizes.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.any(sizes < _MINIMUM_LOOK_SIZE):
        raise ValueError(f"every look in {name} needs at least two observations")
    if np.any(np.diff(sizes) < 0):
        raise ValueError(f"{name} must not decrease")
    if sizes[-1] != observed:
        raise ValueError(f"the last look in {name} must cover all {observed} observations")
    if maximum < observed:
        raise ValueError(f"the maximum size must be at least the {observed} observed values")
    return sizes


def _look_z(control: npt.NDArray[np.float64], treatment: npt.NDArray[np.float64]) -> float:
    control_error = float(np.std(control, ddof=1)) / np.sqrt(control.size)
    treatment_error = float(np.std(treatment, ddof=1)) / np.sqrt(treatment.size)
    standard_error = float(np.hypot(control_error, treatment_error))
    if standard_error == 0.0:
        raise ValueError("a look has zero variance in both arms; its statistic is undefined")
    statistic = (float(np.mean(treatment)) - float(np.mean(control))) / standard_error
    freedom = _welch_degrees_of_freedom(
        control_error, control.size, treatment_error, treatment.size
    )
    if statistic >= 0.0:
        z = -float(special.ndtri_exp(stats.t.logsf(statistic, freedom)))
    else:
        z = float(special.ndtri_exp(stats.t.logcdf(statistic, freedom)))
    if not np.isfinite(z):
        raise ValueError("a look statistic is non-finite; rescale extreme observations")
    return z


def _fractions(
    control: npt.NDArray[np.float64],
    treatment: npt.NDArray[np.float64],
    control_sizes: npt.NDArray[np.int64],
    treatment_sizes: npt.NDArray[np.int64],
    maximum_sizes: tuple[int, int],
) -> tuple[float, ...]:
    control_variance = float(np.var(control, ddof=1))
    treatment_variance = float(np.var(treatment, ddof=1))
    if control_variance == 0.0 and treatment_variance == 0.0:
        raise ValueError("both arms have zero variance; information is undefined")
    final = control_variance / maximum_sizes[0] + treatment_variance / maximum_sizes[1]
    looks = control_variance / control_sizes + treatment_variance / treatment_sizes
    fractions = tuple(float(value) for value in final / looks)
    if any(later <= earlier for earlier, later in pairwise(fractions)):
        raise ValueError("information must increase strictly between looks")
    return fractions


def analyze_sequential_continuous(
    *,
    control: npt.ArrayLike,
    treatment: npt.ArrayLike,
    control_looks: Sequence[int],
    treatment_looks: Sequence[int],
    maximum_control_size: int,
    maximum_treatment_size: int,
    alpha: float = 0.05,
    alternative: AlternativeHypothesis | str,
    spending: SpendingFunction | str = SpendingFunction.O_BRIEN_FLEMING,
    metric_name: str | None = None,
) -> SequentialTestResult:
    """Monitor a mean difference with a one-sided Lan-DeMets group-sequential test.

    ``control`` and ``treatment`` are ordered observations; look ``k`` analyzes the first
    ``control_looks[k]`` and ``treatment_looks[k]`` of them, and the last look covers all of
    them. Each look's Welch statistic is mapped to a z statistic with the same tail
    probability. Information fractions use the full-data arm variances and the planned
    maximum sizes. Monitoring stops at the first look that crosses its boundary.
    """
    control_values, _ = normalize_numeric_sample(control, "control", nonfinite_policy="raise")
    treatment_values, _ = normalize_numeric_sample(treatment, "treatment", nonfinite_policy="raise")
    alpha = validate_sequential_alpha(alpha)
    alternative = normalize_alternative(alternative)
    if alternative is AlternativeHypothesis.TWO_SIDED:
        raise ValueError("sequential monitoring is one-sided: use greater or less")
    spending = normalize_spending(spending)
    maximum_control = validate_positive_integer(maximum_control_size, "maximum_control_size")
    maximum_treatment = validate_positive_integer(maximum_treatment_size, "maximum_treatment_size")
    control_sizes = _looks(control_looks, "control_looks", control_values.size, maximum_control)
    treatment_sizes = _looks(
        treatment_looks, "treatment_looks", treatment_values.size, maximum_treatment
    )
    if control_sizes.size != treatment_sizes.size:
        raise ValueError("control_looks and treatment_looks must have the same length")
    fractions = _fractions(
        control_values,
        treatment_values,
        control_sizes,
        treatment_sizes,
        (maximum_control, maximum_treatment),
    )
    sign = 1.0 if alternative is AlternativeHypothesis.GREATER else -1.0
    boundaries = tuple(
        sign * boundary
        for boundary in group_sequential_boundaries(fractions, alpha=alpha, spending=spending)
    )
    statistics: list[float] = []
    for control_size, treatment_size, boundary in zip(
        control_sizes, treatment_sizes, boundaries, strict=True
    ):
        statistic = _look_z(control_values[:control_size], treatment_values[:treatment_size])
        statistics.append(statistic)
        if sign * statistic >= sign * boundary:
            break
    looks = len(statistics)
    crossed = sign * statistics[-1] >= sign * boundaries[looks - 1]
    return SequentialTestResult(
        metric_name=metric_name,
        alternative=alternative,
        spending=spending,
        alpha=alpha,
        information_fractions=fractions[:looks],
        z_statistics=tuple(statistics),
        boundaries=boundaries[:looks],
        cumulative_alpha_spent=tuple(
            alpha_spent(fraction, alpha=alpha, spending=spending) for fraction in fractions[:looks]
        ),
        crossed=crossed,
        crossed_at_look=looks - 1 if crossed else None,
        method=(
            "Lan-DeMets group-sequential test with exact boundaries on Welch statistics "
            "mapped to z by tail probability"
        ),
    )
