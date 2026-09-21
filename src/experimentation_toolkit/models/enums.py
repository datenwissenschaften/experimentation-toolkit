"""Enumerations shared by the public API."""

from enum import StrEnum


class AlternativeHypothesis(StrEnum):
    """Direction of the treatment-minus-control hypothesis test."""

    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"


class MetricType(StrEnum):
    """Supported metric families."""

    PROPORTION = "proportion"
    CONTINUOUS = "continuous"


class DiagnosticStatus(StrEnum):
    """Status of a factual diagnostic check."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


class CorrectionMethod(StrEnum):
    """Supported multiple-testing procedures."""

    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini-hochberg"


class MDEType(StrEnum):
    """Scale on which a minimum detectable effect is supplied."""

    ABSOLUTE = "absolute"
    RELATIVE = "relative"
