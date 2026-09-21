"""Sample-ratio-mismatch diagnostics."""

from collections.abc import Mapping

import numpy as np
from scipy import stats

from experimentation_toolkit.models.enums import DiagnosticStatus
from experimentation_toolkit.models.results import SRMResult
from experimentation_toolkit.validation.inputs import (
    validate_nonnegative_integer,
    validate_probability,
)


def check_sample_ratio_mismatch(
    *,
    observed_counts: Mapping[str, int],
    expected_proportions: Mapping[str, float],
    significance_level: float = 0.01,
    proportion_tolerance: float = 1e-12,
) -> SRMResult:
    """Run Pearson's chi-square goodness-of-fit test for assignment counts.

    A failing result flags a potential randomization or instrumentation problem. It
    does not identify the cause. The chi-square approximation assumes independent
    assignments and sufficiently large positive expected counts.
    """
    significance_level = validate_probability(
        significance_level, "significance_level", open_interval=True
    )
    if not np.isfinite(proportion_tolerance) or proportion_tolerance < 0.0:
        raise ValueError("proportion_tolerance must be finite and non-negative")
    if len(observed_counts) < 2:
        raise ValueError("at least two variants are required for an SRM check")
    if set(observed_counts) != set(expected_proportions):
        raise ValueError("observed_counts and expected_proportions must have identical keys")

    observed = {
        name: validate_nonnegative_integer(count, f"observed_counts[{name!r}]")
        for name, count in observed_counts.items()
    }
    proportions = {name: float(expected_proportions[name]) for name in observed}
    if any(not np.isfinite(value) or value <= 0.0 for value in proportions.values()):
        raise ValueError("all expected allocation proportions must be finite and positive")
    if not np.isclose(sum(proportions.values()), 1.0, rtol=0.0, atol=proportion_tolerance):
        raise ValueError("expected allocation proportions must sum to 1")
    total = sum(observed.values())
    if total <= 0:
        raise ValueError("the total observed assignment count must be greater than zero")
    expected = {name: total * proportion for name, proportion in proportions.items()}
    if min(expected.values()) < 5.0:
        raise ValueError(
            "every expected assignment count must be at least 5 for the chi-square test"
        )

    result = stats.chisquare(
        f_obs=np.asarray(tuple(observed.values()), dtype=np.float64),
        f_exp=np.asarray(tuple(expected.values()), dtype=np.float64),
    )
    statistic = float(result.statistic)
    p_value = float(result.pvalue)
    failed = p_value < significance_level
    return SRMResult(
        observed_counts=observed,
        expected_counts={name: float(value) for name, value in expected.items()},
        chi_square_statistic=statistic,
        p_value=p_value,
        significance_level=significance_level,
        status=DiagnosticStatus.FAIL if failed else DiagnosticStatus.PASS,
        message=(
            "Assignment counts are inconsistent with the configured allocation at the "
            "selected threshold; investigate randomization or instrumentation."
            if failed
            else "No sample ratio mismatch was detected at the selected threshold."
        ),
    )
