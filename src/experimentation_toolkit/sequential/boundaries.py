"""Lan-DeMets alpha spending and exact one-sided group-sequential boundaries."""

from collections.abc import Sequence
from itertools import pairwise
from math import ceil, e, floor, log, sqrt

import numpy as np
import numpy.typing as npt
from scipy import optimize, special

from experimentation_toolkit.models.enums import SpendingFunction
from experimentation_toolkit.validation.inputs import validate_probability

_LOWER_SCORE = -10.0
_KERNEL_WIDTH = 10.0
_GRID_PER_INCREMENT = 4.0
_COARSEST_GRID = 0.02
_FINEST_GRID = 0.005
_SOLVER_LOWER = -40.0
_SOLVER_TOLERANCE = 1e-10
_BRACKET_MARGIN = 1.0
_NEGLIGIBLE_MASS = 1e-250


def validate_sequential_alpha(alpha: float) -> float:
    """Return a one-sided sequential significance level in (0, 0.5)."""
    alpha = validate_probability(alpha, "alpha", open_interval=True)
    if alpha >= 0.5:
        raise ValueError("alpha must be below 0.5 for one-sided sequential monitoring")
    return alpha


def normalize_spending(spending: SpendingFunction | str) -> SpendingFunction:
    """Normalize a public spending-function argument."""
    try:
        return SpendingFunction(spending)
    except ValueError as error:
        allowed = ", ".join(item.value for item in SpendingFunction)
        raise ValueError(f"spending must be one of: {allowed}") from error


def validate_information_fractions(information_fractions: Sequence[float]) -> tuple[float, ...]:
    """Require a non-empty, strictly increasing sequence of fractions in (0, 1]."""
    fractions = tuple(float(value) for value in information_fractions)
    if not fractions:
        raise ValueError("information_fractions must not be empty")
    if any(not np.isfinite(value) for value in fractions):
        raise ValueError("information fractions must be finite")
    if fractions[0] <= 0.0 or fractions[-1] > 1.0:
        raise ValueError("information fractions must be in (0, 1]")
    if any(later <= earlier for earlier, later in pairwise(fractions)):
        raise ValueError("information fractions must increase strictly")
    return fractions


def _log_alpha_spent(fraction: float, alpha: float, spending: SpendingFunction) -> float:
    if spending is SpendingFunction.O_BRIEN_FLEMING:
        critical = -float(special.ndtri(alpha / 2.0))
        return log(2.0) + float(special.log_ndtr(-critical / sqrt(fraction)))
    return log(alpha) + log(float(np.log1p((e - 1.0) * fraction)))


def alpha_spent(
    information_fraction: float,
    *,
    alpha: float,
    spending: SpendingFunction | str,
) -> float:
    """Return the cumulative one-sided alpha spent at an information fraction in [0, 1].

    O'Brien-Fleming-type spending is ``2 - 2 Phi(z_{1-alpha/2} / sqrt(t))`` and Pocock-type
    spending is ``alpha ln(1 + (e - 1) t)``; both spend exactly ``alpha`` at ``t = 1``.
    """
    alpha = validate_sequential_alpha(alpha)
    spending = normalize_spending(spending)
    fraction = validate_probability(information_fraction, "information_fraction")
    if fraction == 0.0:
        return 0.0
    if fraction == 1.0:
        return alpha
    return float(np.exp(_log_alpha_spent(fraction, alpha, spending)))


def _log_increment(
    previous: float, current: float, alpha: float, spending: SpendingFunction
) -> float:
    log_current = _log_alpha_spent(current, alpha, spending) if current < 1.0 else log(alpha)
    if previous == 0.0:
        return log_current
    log_previous = _log_alpha_spent(previous, alpha, spending)
    return log_current + float(np.log1p(-np.exp(log_previous - log_current)))


def _grid_step(fractions: tuple[float, ...]) -> float:
    increments = np.diff(np.asarray((0.0, *fractions)))
    finest = float(np.sqrt(np.min(increments))) / _GRID_PER_INCREMENT
    return min(_COARSEST_GRID, max(_FINEST_GRID, finest))


def _unconditional_boundary(log_target: float) -> float:
    return float(
        optimize.brentq(
            lambda z: float(special.log_ndtr(-z)) - log_target,
            _SOLVER_LOWER,
            -_SOLVER_LOWER - log_target,
            xtol=_SOLVER_TOLERANCE,
        )
    )


def _log_exit(
    masses: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    score_boundary: float,
    spread: float,
) -> float:
    held = masses > 0.0
    tails = np.log(masses[held]) + special.log_ndtr((scores[held] - score_boundary) / spread)
    largest = float(np.max(tails))
    return largest + float(np.log(np.sum(np.exp(tails - largest))))


def _kernel(spread: float, step: float) -> npt.NDArray[np.float64]:
    reach = ceil(_KERNEL_WIDTH * spread / step)
    edges = (np.arange(-reach, 2) - 0.5) * step / spread
    left = np.diff(special.ndtr(edges))
    return np.asarray(np.concatenate((left, left[-2::-1])), dtype=np.float64)


def _continue(
    masses: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    score_boundary: float,
    spread: float,
    step: float,
) -> npt.NDArray[np.float64]:
    kernel = _kernel(spread, step)
    reach = (kernel.size - 1) // 2
    moved = np.convolve(masses, kernel)[reach : reach + masses.size]
    kept = np.clip((score_boundary - (scores - step / 2.0)) / step, 0.0, 1.0)
    continued = moved * kept
    return np.asarray(np.where(continued < _NEGLIGIBLE_MASS, 0.0, continued), dtype=np.float64)


def _solve_boundary(
    masses: npt.NDArray[np.float64],
    scores: npt.NDArray[np.float64],
    root: float,
    spread: float,
    log_target: float,
    ceiling: float,
) -> float:
    return float(
        optimize.brentq(
            lambda z: _log_exit(masses, scores, z * root, spread) - log_target,
            _SOLVER_LOWER,
            ceiling + _BRACKET_MARGIN,
            xtol=_SOLVER_TOLERANCE,
        )
    )


def group_sequential_boundaries(
    information_fractions: Sequence[float],
    *,
    alpha: float = 0.05,
    spending: SpendingFunction | str = SpendingFunction.O_BRIEN_FLEMING,
) -> tuple[float, ...]:
    """Return one-sided upper z boundaries for Lan-DeMets alpha spending.

    The z statistics follow the canonical joint distribution with
    ``Corr(Z_i, Z_j) = sqrt(t_i / t_j)``. Each boundary is solved so that the probability of
    first crossing at that look under the null equals the alpha-spending increment, using
    recursive numerical integration of the score process on a grid.
    """
    fractions = validate_information_fractions(information_fractions)
    alpha = validate_sequential_alpha(alpha)
    spending = normalize_spending(spending)
    targets = [
        _log_increment(previous, current, alpha, spending)
        for previous, current in zip((0.0, *fractions), fractions, strict=False)
    ]
    ceilings = [
        _unconditional_boundary(target) * sqrt(fraction)
        for target, fraction in zip(targets, fractions, strict=True)
    ]
    step = _grid_step(fractions)
    upper = ceil((max(ceilings) + _BRACKET_MARGIN) / step) + 2
    scores = np.arange(floor(_LOWER_SCORE / step), upper, dtype=np.float64) * step
    masses = np.zeros_like(scores)
    masses[int(np.argmin(np.abs(scores)))] = 1.0
    boundaries: list[float] = []
    previous = 0.0
    for fraction, target, ceiling in zip(fractions, targets, ceilings, strict=True):
        spread = sqrt(fraction - previous)
        root = sqrt(fraction)
        boundary = _solve_boundary(masses, scores, root, spread, target, ceiling / root)
        boundaries.append(boundary)
        masses = _continue(masses, scores, boundary * root, spread, step)
        previous = fraction
    return tuple(boundaries)
