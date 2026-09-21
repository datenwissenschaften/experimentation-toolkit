"""Experiment metadata models."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from experimentation_toolkit.models.enums import MetricType


class ToolkitModel(BaseModel):
    """Strict, immutable base model with standards-compliant JSON output."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Variant(ToolkitModel):
    """A named experiment arm."""

    name: str = Field(min_length=1)


class Metric(ToolkitModel):
    """Metadata for one experiment outcome."""

    name: str = Field(min_length=1)
    metric_type: MetricType
    primary: bool = False
    description: str | None = None


class Experiment(ToolkitModel):
    """Metadata for a two-arm controlled experiment."""

    name: str = "A/B experiment"
    control: Variant | str
    treatment: Variant | str
    metrics: tuple[Metric, ...] = ()

    @model_validator(mode="after")
    def validate_distinct_variants(self) -> "Experiment":
        """Require distinct non-empty arm names and unique metric names."""
        control_name = self.control.name if isinstance(self.control, Variant) else self.control
        treatment_name = (
            self.treatment.name if isinstance(self.treatment, Variant) else self.treatment
        )
        if not control_name.strip() or not treatment_name.strip():
            raise ValueError("variant names must be non-empty")
        if control_name == treatment_name:
            raise ValueError("control and treatment must have different names")
        metric_names = [metric.name for metric in self.metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("metric names must be unique within an experiment")
        return self
