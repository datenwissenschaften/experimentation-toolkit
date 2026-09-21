"""Reusable validation and normalization helpers."""

from collections.abc import Iterable
from typing import Literal

import numpy as np
import numpy.typing as npt

from experimentation_toolkit.models.enums import AlternativeHypothesis


def validate_probability(value: float, name: str, *, open_interval: bool = False) -> float:
    """Return a finite probability after validating its domain."""
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name} must be finite")
    valid = 0.0 < normalized < 1.0 if open_interval else 0.0 <= normalized <= 1.0
    if not valid:
        interval = "(0, 1)" if open_interval else "[0, 1]"
        raise ValueError(f"{name} must be in {interval}")
    return normalized


def validate_positive_integer(value: int, name: str) -> int:
    """Reject booleans, non-integral values, and non-positive counts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return normalized


def validate_nonnegative_integer(value: int, name: str) -> int:
    """Reject booleans, non-integral values, and negative counts."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return normalized


def normalize_alternative(value: AlternativeHypothesis | str) -> AlternativeHypothesis:
    """Normalize a public alternative-hypothesis argument."""
    try:
        return AlternativeHypothesis(value)
    except ValueError as error:
        allowed = ", ".join(item.value for item in AlternativeHypothesis)
        raise ValueError(f"alternative must be one of: {allowed}") from error


def normalize_numeric_sample(
    values: npt.ArrayLike,
    name: str,
    *,
    nonfinite_policy: Literal["raise", "omit"],
) -> tuple[npt.NDArray[np.float64], int]:
    """Return a one-dimensional float64 sample and omitted-value count."""
    if nonfinite_policy not in ("raise", "omit"):
        raise ValueError("nonfinite_policy must be 'raise' or 'omit'")
    try:
        sample = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be numeric array-like") from error
    if sample.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    finite = np.isfinite(sample)
    omitted = int(sample.size - np.count_nonzero(finite))
    if omitted and nonfinite_policy == "raise":
        raise ValueError(
            f"{name} contains {omitted} non-finite observation(s); "
            "use nonfinite_policy='omit' to remove them explicitly"
        )
    if nonfinite_policy == "omit":
        sample = sample[finite]
    if sample.size < 2:
        raise ValueError(f"{name} must contain at least two finite observations")
    return sample, omitted


def validate_p_values(p_values: Iterable[float]) -> tuple[float, ...]:
    """Normalize a non-empty iterable of finite p-values."""
    normalized = tuple(float(value) for value in p_values)
    if not normalized:
        raise ValueError("p_values must not be empty")
    if any(not np.isfinite(value) or value < 0.0 or value > 1.0 for value in normalized):
        raise ValueError("every p-value must be finite and in [0, 1]")
    return normalized
