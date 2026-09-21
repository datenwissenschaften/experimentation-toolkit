import json

from examples.game_onboarding import build_report

from experimentation_toolkit import DiagnosticStatus


def test_synthetic_workflow_is_deterministic_and_complete() -> None:
    first = build_report()
    second = build_report()
    assert first == second
    assert len(first.analyses) == 3
    assert first.diagnostics[0].status is DiagnosticStatus.PASS
    assert first.multiple_testing is not None
    assert len(first.multiple_testing.adjusted_p_values) == 3
    assert first.metadata["synthetic"] is True
    assert json.loads(first.model_dump_json())["experiment"]["name"]
