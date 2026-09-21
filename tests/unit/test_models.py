import json

import numpy as np
import pytest
from pydantic import ValidationError

from experimentation_toolkit import (
    ConfidenceInterval,
    Experiment,
    ExperimentReport,
    Metric,
    MetricType,
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
