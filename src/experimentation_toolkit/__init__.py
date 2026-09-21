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
    SRMResult,
    Variant,
)
from experimentation_toolkit.multiple_testing import adjust_p_values
from experimentation_toolkit.power import proportion_achieved_power, proportion_sample_size

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
    "Variant",
    "adjust_p_values",
    "analyze_continuous",
    "analyze_proportion",
    "check_sample_ratio_mismatch",
    "proportion_achieved_power",
    "proportion_sample_size",
]

__version__ = "0.1.0"
