"""Statistically explicit tools for classical controlled A/B experiments."""

from experimentation_toolkit.analysis import analyze_continuous, analyze_proportion
from experimentation_toolkit.diagnostics import check_sample_ratio_mismatch
from experimentation_toolkit.models import (
    AchievedPowerResult,
    AlternativeHypothesis,
    AnalysisResult,
    ConfidenceInterval,
    CorrectionMethod,
    DiagnosticResult,
    DiagnosticStatus,
    Experiment,
    ExperimentReport,
    MDEType,
    Metric,
    MetricType,
    MultipleTestingResult,
    ProportionSampleSizeResult,
    SequentialTestResult,
    SpendingFunction,
    SRMResult,
    Variant,
)
from experimentation_toolkit.multiple_testing import adjust_p_values
from experimentation_toolkit.power import proportion_achieved_power, proportion_sample_size
from experimentation_toolkit.sequential import (
    alpha_spent,
    analyze_sequential_continuous,
    group_sequential_boundaries,
)

__all__ = [
    "AchievedPowerResult",
    "AlternativeHypothesis",
    "AnalysisResult",
    "ConfidenceInterval",
    "CorrectionMethod",
    "DiagnosticResult",
    "DiagnosticStatus",
    "Experiment",
    "ExperimentReport",
    "MDEType",
    "Metric",
    "MetricType",
    "MultipleTestingResult",
    "ProportionSampleSizeResult",
    "SRMResult",
    "SequentialTestResult",
    "SpendingFunction",
    "Variant",
    "adjust_p_values",
    "alpha_spent",
    "analyze_continuous",
    "analyze_proportion",
    "analyze_sequential_continuous",
    "check_sample_ratio_mismatch",
    "group_sequential_boundaries",
    "proportion_achieved_power",
    "proportion_sample_size",
]

__version__ = "0.2.0"
