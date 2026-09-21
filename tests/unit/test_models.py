import json

import numpy as np
import pytest
from pydantic import ValidationError

from experimentation_toolkit import (
    AnalysisResult,
    ConfidenceInterval,
    CorrectionMethod,
    DiagnosticStatus,
    Experiment,
    ExperimentReport,
    Metric,
    MetricType,
    MultipleTestingResult,
    ProportionSampleSizeResult,
    SRMResult,
    analyze_proportion,
)


def test_experiment_validates_distinct_variants_and_metrics() -> None:
    with pytest.raises(ValidationError):
        Experiment(control="same", treatment="same")
    with pytest.raises(ValidationError):
        Experiment(
            control="a",
            treatment="b",
            metrics=(
                Metric(name="x", metric_type=MetricType.PROPORTION),
                Metric(name="x", metric_type=MetricType.CONTINUOUS),
            ),
        )
    with pytest.raises(ValidationError):
        Experiment(control=" ", treatment="treatment")


def test_report_serializes_to_standards_compliant_json() -> None:
    result = analyze_proportion(
        control_successes=np.int64(10),
        control_total=np.int64(100),
        treatment_successes=np.int64(12),
        treatment_total=np.int64(100),
        alternative="greater",
    )
    report = ExperimentReport(
        experiment=Experiment(control="control", treatment="treatment"),
        analyses=(result,),
        metadata={"numpy_scalar": np.int64(3)},
    )
    payload = report.model_dump_json()
    decoded = json.loads(payload, parse_constant=lambda value: pytest.fail(value))
    assert decoded["analyses"][0]["confidence_interval"]["upper"] is None
    assert decoded["metadata"]["numpy_scalar"] == 3


def test_models_reject_nonfinite_json_numbers() -> None:
    with pytest.raises(ValidationError):
        ExperimentReport(
            experiment=Experiment(control="control", treatment="treatment"),
            analyses=(),
            metadata={"bad": float("nan")},
        )


def test_confidence_interval_rejects_reversed_bounds() -> None:
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=1.0, upper=0.0, confidence_level=0.95)


def test_confidence_interval_requires_a_bounded_endpoint() -> None:
    with pytest.raises(ValidationError):
        ConfidenceInterval(lower=None, upper=None, confidence_level=0.95)


def test_models_are_immutable() -> None:
    interval = ConfidenceInterval(lower=0.0, upper=1.0, confidence_level=0.95)
    with pytest.raises(ValidationError):
        interval.lower = -1.0  # type: ignore[misc]


def test_srm_model_rejects_incoherent_degrees_of_freedom() -> None:
    with pytest.raises(ValidationError, match="degrees of freedom"):
        SRMResult(
            observed_counts={"a": 10, "b": 10},
            expected_counts={"a": 10.0, "b": 10.0},
            chi_square_statistic=0.0,
            p_value=1.0,
            significance_level=0.01,
            degrees_of_freedom=2,
            status=DiagnosticStatus.PASS,
            message="reference",
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_counts": {"a": 10.0, "c": 10.0}},
        {"observed_counts": {"a": -1, "b": 21}},
        {"expected_counts": {"a": 0.0, "b": 20.0}},
        {"status": DiagnosticStatus.WARN},
    ],
)
def test_srm_model_rejects_structural_incoherence(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "observed_counts": {"a": 10, "b": 10},
        "expected_counts": {"a": 10.0, "b": 10.0},
        "chi_square_statistic": 0.0,
        "p_value": 1.0,
        "significance_level": 0.01,
        "degrees_of_freedom": 1,
        "status": DiagnosticStatus.PASS,
        "message": "reference",
    }
    values.update(changes)
    with pytest.raises(ValidationError):
        SRMResult(**values)  # type: ignore[arg-type]


def test_multiple_testing_model_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValidationError, match="equal lengths"):
        MultipleTestingResult(
            original_p_values=(0.1, 0.2),
            adjusted_p_values=(0.2,),
            rejected=(False, False),
            alpha=0.05,
            method=CorrectionMethod.HOLM,
            error_rate_control="family-wise error rate",
        )


def test_multiple_testing_model_rejects_incoherent_decision() -> None:
    with pytest.raises(ValidationError, match="rejection decisions"):
        MultipleTestingResult(
            original_p_values=(0.01,),
            adjusted_p_values=(0.01,),
            rejected=(False,),
            alpha=0.05,
            method=CorrectionMethod.HOLM,
            error_rate_control="family-wise error rate",
        )


def test_multiple_testing_model_rejects_wrong_error_rate_description() -> None:
    with pytest.raises(ValidationError, match="error-rate"):
        MultipleTestingResult(
            original_p_values=(0.1,),
            adjusted_p_values=(0.1,),
            rejected=(False,),
            alpha=0.05,
            method=CorrectionMethod.BENJAMINI_HOCHBERG,
            error_rate_control="family-wise error rate",
        )


def test_sample_size_model_rejects_incorrect_total() -> None:
    with pytest.raises(ValidationError, match="total sample size"):
        ProportionSampleSizeResult(
            baseline_rate=0.1,
            treatment_rate=0.11,
            minimum_detectable_effect=0.01,
            mde_type="absolute",
            alpha=0.05,
            target_power=0.8,
            achieved_power=0.8,
            allocation_ratio=1.0,
            alternative="two-sided",
            control_sample_size=100,
            treatment_sample_size=100,
            total_sample_size=199,
            method="reference",
        )


def test_sample_size_model_requires_achieved_target_power() -> None:
    with pytest.raises(ValidationError, match="achieved power"):
        ProportionSampleSizeResult(
            baseline_rate=0.1,
            treatment_rate=0.11,
            minimum_detectable_effect=0.01,
            mde_type="absolute",
            alpha=0.05,
            target_power=0.8,
            achieved_power=0.79,
            allocation_ratio=1.0,
            alternative="two-sided",
            control_sample_size=100,
            treatment_sample_size=100,
            total_sample_size=200,
            method="reference",
        )


def test_json_boundary_values_remain_standard_compliant() -> None:
    result = analyze_proportion(
        control_successes=0,
        control_total=10,
        treatment_successes=10,
        treatment_total=10,
        alternative="less",
    )
    payload = ExperimentReport(
        experiment=Experiment(control="control", treatment="treatment"),
        analyses=(result,),
        metadata={"small": np.nextafter(0.0, 1.0), "large": np.finfo(float).max},
    ).model_dump_json()
    decoded = json.loads(payload, parse_constant=lambda value: pytest.fail(value))
    assert decoded["analyses"][0]["confidence_interval"]["lower"] is None
    assert decoded["metadata"]["small"] > 0.0


@pytest.mark.parametrize(
    ("field", "value"),
    [("p_value", 1.1), ("control_sample_size", 0), ("confidence_level", 1.0)],
)
def test_analysis_model_rejects_invalid_public_invariants(field: str, value: float) -> None:
    valid = analyze_proportion(
        control_successes=10,
        control_total=100,
        treatment_successes=12,
        treatment_total=100,
    ).model_dump()
    valid[field] = value
    with pytest.raises(ValidationError):
        AnalysisResult(**valid)


def test_analysis_model_rejects_effect_and_interval_direction_mismatch() -> None:
    valid = analyze_proportion(
        control_successes=10,
        control_total=100,
        treatment_successes=12,
        treatment_total=100,
    ).model_dump()
    wrong_effect = {**valid, "absolute_effect": 0.5}
    with pytest.raises(ValidationError, match="treatment estimate minus control"):
        AnalysisResult(**wrong_effect)
    wrong_interval = {
        **valid,
        "alternative": "greater",
        "confidence_interval": ConfidenceInterval(lower=-0.1, upper=0.1, confidence_level=0.95),
    }
    with pytest.raises(ValidationError, match="lower-bounded"):
        AnalysisResult(**wrong_interval)


def test_analysis_model_rejects_confidence_and_relative_metadata_mismatch() -> None:
    valid = analyze_proportion(
        control_successes=10,
        control_total=100,
        treatment_successes=12,
        treatment_total=100,
    ).model_dump()
    wrong_confidence = {
        **valid,
        "confidence_interval": ConfidenceInterval(lower=-0.1, upper=0.1, confidence_level=0.9),
    }
    with pytest.raises(ValidationError, match="confidence levels"):
        AnalysisResult(**wrong_confidence)
    wrong_relative = {**valid, "relative_effect": None}
    with pytest.raises(ValidationError, match="relative effect"):
        AnalysisResult(**wrong_relative)
