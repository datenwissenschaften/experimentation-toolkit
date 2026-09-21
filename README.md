# experimentation-toolkit

`experimentation-toolkit` is a typed Python library for classical frequentist analysis of
independent, two-arm controlled experiments. It connects assignment diagnostics, metric-level
inference, multiplicity correction, power planning, and a serializable report without making a
DataFrame the core abstraction.

The package reports estimates, uncertainty, and statistical evidence. It deliberately does not
turn a small p-value into a claim that an effect is important, beneficial, or worth shipping.

## Quick example

```python
from experimentation_toolkit import (
    Experiment,
    ExperimentReport,
    analyze_proportion,
    check_sample_ratio_mismatch,
)

experiment = Experiment(control="control", treatment="treatment")
srm = check_sample_ratio_mismatch(
    observed_counts={"control": 10_000, "treatment": 10_050},
    expected_proportions={"control": 0.5, "treatment": 0.5},
)
conversion = analyze_proportion(
    control_successes=1_200,
    control_total=10_000,
    treatment_successes=1_280,
    treatment_total=10_050,
    confidence_level=0.95,
    alternative="two-sided",
    metric_name="conversion",
)
report = ExperimentReport(
    experiment=experiment,
    analyses=(conversion,),
    diagnostics=(srm,),
)
print(report.model_dump_json(indent=2))
```

The absolute effect is a risk difference. A move from 10% to 11% is `0.01`, or **one
percentage point**. Its relative lift is `0.10`, or **10% relative**. Results expose these as
separate fields with an explicit relative-effect definition.

## Supported analyses

- Independent binary outcomes from success counts and totals
- Independent continuous outcomes from numeric array-like samples
- Two-sided, treatment-greater, and treatment-less alternatives
- Sample-ratio-mismatch checks for two or more assignment groups
- Proportion sample-size planning and achieved-power approximation
- Bonferroni, Holm, and Benjamini-Hochberg p-value adjustment
- Immutable Pydantic models and standards-compliant JSON reports

## Statistical methods

Proportion inference uses a pooled two-sample score z-test for the point null
`p_treatment - p_control = 0`. The risk-difference confidence interval is Newcombe's hybrid
score interval formed from Wilson limits without continuity correction. It remains useful at
zero and one, while an explicit warning flags sparse expected cells where the test's normal
approximation may be inaccurate.

Continuous inference uses Welch's unequal-variance two-sample t-test and the
Welch-Satterthwaite degrees of freedom. It does not assume equal group variances. Both analyses
define effects as treatment minus control and align one-sided confidence bounds with the chosen
alternative. An unbounded endpoint is serialized as `null`, never infinity.

See [docs/statistical-methods.md](docs/statistical-methods.md) for formulas, assumptions, and
references.

## Assumptions

Inference assumes randomized or otherwise exchangeable groups, independent experimental units,
a metric definition fixed independently of observed treatment results, and no interference
between units. Binary observations must be Bernoulli outcomes at the analysis unit. Welch
inference assumes finite variance and is exact under normal sampling; with larger samples it
uses the sampling distribution of the mean. The library cannot establish that these assumptions
hold from arrays or counts alone.

## Sample ratio mismatch

`check_sample_ratio_mismatch` applies Pearson's chi-square goodness-of-fit test to observed arm
counts and configured allocation probabilities. `FAIL` means the allocation is statistically
inconsistent with the plan at the selected threshold. It indicates a potential randomization,
eligibility, logging, or instrumentation problem; it does not identify the cause. Inspect SRM
before interpreting treatment effects.

## Power and sample size

```python
from experimentation_toolkit import proportion_sample_size

plan = proportion_sample_size(
    baseline_rate=0.10,
    minimum_detectable_effect=0.01,
    mde_type="absolute",  # +1 percentage point
    alpha=0.05,
    power=0.80,
    allocation_ratio=1.0,  # treatment / control
)
```

Set `mde_type="relative"` to express the MDE as a fraction of baseline: `0.10` at a 10%
baseline targets an 11% treatment rate. The calculation is a fixed-horizon normal approximation,
and the returned achieved power reflects the integer group sizes. Planning inputs are assumptions,
not estimates guaranteed to hold after launch.

## Multiple testing

`adjust_p_values` preserves original hypothesis order and optionally carries labels. Bonferroni
and Holm control the family-wise error rate. Benjamini-Hochberg controls the false discovery rate
under independence or positive dependence and must not be interpreted as FWER control. The caller
defines the hypothesis family before looking at results.

## Interpretation

Statistical significance quantifies incompatibility with a null model under its assumptions.
Practical significance requires domain context, costs, guardrails, and a pre-specified decision
rule. Read the absolute effect and its confidence interval first. Relative effects are omitted
when the control denominator is zero or when continuous means do not support a conservative,
non-negative and numerically resolved ratio-scale interpretation.

## Installation

Python 3.12 is required.

```bash
python -m pip install experimentation-toolkit
```

For a source checkout:

```bash
python -m pip install -e '.[dev]'
```

To execute the visual notebook, install the separate example dependencies:

```bash
python -m pip install -e '.[dev,examples]'
```

## Testing and development

```bash
ruff check .
ruff format --check .
mypy
pytest
pytest --cov=experimentation_toolkit --cov-report=term-missing
```

The deterministic [game onboarding notebook](examples/game_onboarding.ipynb) demonstrates
assignment counts, SRM, a primary completion metric, retention and session-duration metrics, Holm
correction, interactive Plotly uncertainty plots, and report serialization. It is self-contained,
uses a fixed seed, and labels all generated observations as synthetic.

## Limitations

Version 0.1 supports independent, two-arm, fixed-horizon analyses only. It does not implement
paired or clustered designs, exact unconditional proportion tests, covariate adjustment, CUPED,
sequential monitoring, alpha spending, Bayesian analysis, causal identification, heterogeneous
effects, feature flags, or platform infrastructure. The proportion test and power methods use
large-sample approximations; warnings do not convert them into exact procedures.

## Roadmap

Statistically useful next steps include stratified analyses with explicit estimands, cluster-aware
standard errors and design effects, robust or transformed continuous outcomes, exact methods for
sparse binary data, and simulation-based validation of planning assumptions. Each belongs behind
a distinct API that states its sampling unit and assumptions.

## License

MIT
