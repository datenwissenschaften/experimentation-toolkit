"""Multiple-testing corrections with stable input-order mapping."""

from collections.abc import Iterable, Sequence

import numpy as np

from experimentation_toolkit.models.enums import CorrectionMethod
from experimentation_toolkit.models.results import MultipleTestingResult
from experimentation_toolkit.validation.inputs import validate_p_values, validate_probability


def adjust_p_values(
    p_values: Iterable[float],
    *,
    method: CorrectionMethod | str,
    alpha: float = 0.05,
    hypothesis_labels: Sequence[str] | None = None,
) -> MultipleTestingResult:
    """Adjust p-values with Bonferroni, Holm, or Benjamini-Hochberg.

    Bonferroni and Holm control family-wise error rate. Benjamini-Hochberg
    controls false discovery rate under independence or positive dependence;
    it does not generally control family-wise error rate.
    """
    p_values = validate_p_values(p_values)
    alpha = validate_probability(alpha, "alpha", open_interval=True)
    try:
        method = CorrectionMethod(method)
    except ValueError as error:
        allowed = ", ".join(item.value for item in CorrectionMethod)
        raise ValueError(f"method must be one of: {allowed}") from error
    labels = tuple(hypothesis_labels) if hypothesis_labels is not None else None
    if labels is not None and len(labels) != len(p_values):
        raise ValueError("hypothesis_labels must have the same length as p_values")
    if labels is not None and any(not label for label in labels):
        raise ValueError("hypothesis labels must be non-empty")

    values = np.asarray(p_values, dtype=np.float64)
    count = values.size
    if method is CorrectionMethod.BONFERRONI:
        adjusted = np.minimum(values * count, 1.0)
        error_rate_control = "family-wise error rate"
    else:
        order = np.argsort(values, kind="stable")
        sorted_values = values[order]
        if method is CorrectionMethod.HOLM:
            scaled = (count - np.arange(count)) * sorted_values
            sorted_adjusted = np.minimum(np.maximum.accumulate(scaled), 1.0)
            error_rate_control = "family-wise error rate"
        else:
            scaled = count * sorted_values / np.arange(1, count + 1)
            sorted_adjusted = np.minimum(np.minimum.accumulate(scaled[::-1])[::-1], 1.0)
            error_rate_control = "false discovery rate"
        adjusted = np.empty(count, dtype=np.float64)
        adjusted[order] = sorted_adjusted
    rejected = adjusted <= alpha
    return MultipleTestingResult(
        original_p_values=p_values,
        adjusted_p_values=tuple(float(value) for value in adjusted),
        rejected=tuple(bool(value) for value in rejected),
        alpha=alpha,
        method=method,
        hypothesis_labels=labels,
        error_rate_control=error_rate_control,
    )
