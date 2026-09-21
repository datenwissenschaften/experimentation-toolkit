import json

import numpy as np

from experimentation_toolkit import (
    DiagnosticStatus,
    Experiment,
    ExperimentReport,
    Metric,
    MetricType,
    adjust_p_values,
    analyze_continuous,
    analyze_proportion,
    check_sample_ratio_mismatch,
)


def _build_report() -> ExperimentReport:
    rng = np.random.default_rng(20260921)
    control_size = 10_050
    treatment_size = 9_950
    srm = check_sample_ratio_mismatch(
        observed_counts={"existing onboarding": control_size, "new onboarding": treatment_size},
        expected_proportions={"existing onboarding": 0.5, "new onboarding": 0.5},
    )

    control_completion = rng.binomial(1, 0.62, control_size)
    treatment_completion = rng.binomial(1, 0.645, treatment_size)
    control_retention = rng.binomial(1, 0.31, control_size)
    treatment_retention = rng.binomial(1, 0.318, treatment_size)
    control_duration = np.maximum(rng.normal(34.0, 12.0, control_size), 0.0)
    treatment_duration = np.maximum(rng.normal(35.2, 13.5, treatment_size), 0.0)

    analyses = (
        analyze_proportion(
            control_successes=int(control_completion.sum()),
            control_total=control_size,
            treatment_successes=int(treatment_completion.sum()),
            treatment_total=treatment_size,
            metric_name="onboarding completion rate",
        ),
        analyze_proportion(
            control_successes=int(control_retention.sum()),
            control_total=control_size,
            treatment_successes=int(treatment_retention.sum()),
            treatment_total=treatment_size,
            metric_name="7-day retention",
        ),
        analyze_continuous(
            control=control_duration,
            treatment=treatment_duration,
            metric_name="session duration (minutes)",
        ),
    )
    correction = adjust_p_values(
        (result.p_value for result in analyses if result.p_value is not None),
        method="holm",
        hypothesis_labels=tuple(result.metric_name or "unnamed" for result in analyses),
    )
    experiment = Experiment(
        name="Synthetic game onboarding redesign",
        control="existing onboarding",
        treatment="new onboarding",
        metrics=(
            Metric(
                name="onboarding completion rate",
                metric_type=MetricType.PROPORTION,
                primary=True,
            ),
            Metric(name="7-day retention", metric_type=MetricType.PROPORTION),
            Metric(name="session duration (minutes)", metric_type=MetricType.CONTINUOUS),
        ),
    )
    return ExperimentReport(
        experiment=experiment,
        analyses=analyses,
        diagnostics=(srm,),
        multiple_testing=correction,
        metadata={"synthetic": True, "random_seed": 20260921},
    )


def test_synthetic_workflow_is_deterministic_and_complete() -> None:
    first = _build_report()
    second = _build_report()
    assert first == second
    assert len(first.analyses) == 3
    assert first.diagnostics[0].status is DiagnosticStatus.PASS
    assert first.multiple_testing is not None
    assert len(first.multiple_testing.adjusted_p_values) == 3
    assert first.metadata["synthetic"] is True
    assert json.loads(first.model_dump_json())["experiment"]["name"]
