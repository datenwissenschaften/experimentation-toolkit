# Statistical methods

This document specifies the conventions used by `experimentation-toolkit` 0.2. Effects are always
defined as treatment minus control. Let `alpha = 1 - confidence_level`.

## Binary outcomes

For arm `j`, the observed rate is

\[
\hat p_j = x_j / n_j.
\]

The absolute effect is the risk difference
`d = p_treatment - p_control`. When the control rate is nonzero, relative lift is
`d / p_control`. The two quantities are never substituted for one another. Relative lift is
omitted at a zero control rate.

### Hypothesis test

The null is `H0: p_treatment - p_control = 0`. The pooled estimate under the null is

\[
\hat p = (x_c + x_t)/(n_c+n_t),
\]

and the score statistic is

\[
z = \frac{\hat p_t-\hat p_c}
{\sqrt{\hat p(1-\hat p)(1/n_t+1/n_c)}}.
\]

Two-sided p-values use `2 Phi(-abs(z))`; directional p-values use the appropriate normal CDF or
survival function for the signed statistic. When both samples are entirely failures or entirely
successes, the null standard error is zero and identical observed rates carry `z = 0, p = 1`.
An approximation warning is attached when any expected success/failure count under the pooled null
is below five.

### Confidence interval

The package uses Newcombe's hybrid score interval (method 10) for the difference between two
independent proportions. Wilson score limits are computed separately without continuity correction
and combined by the square-and-add method. This avoids the most serious boundary behavior of the
unpooled Wald interval. For a greater alternative the result is `[lower, +infinity)`; for a less
alternative it is `(-infinity, upper]`. The unbounded endpoint is represented as `None` in Python
and `null` in JSON.

Reference: Robert G. Newcombe, “Interval estimation for the difference between independent
proportions: comparison of eleven methods,” *Statistics in Medicine* 17 (1998), 873–890,
[doi:10.1002/(SICI)1097-0258(19980430)17:8<873::AID-SIM779>3.0.CO;2-I](https://doi.org/10.1002/(SICI)1097-0258(19980430)17:8%3C873::AID-SIM779%3E3.0.CO;2-I).

## Continuous outcomes

The absolute effect is `mean(treatment) - mean(control)`. Sample standard deviations use Bessel's
correction (`ddof=1`). The Welch statistic is

\[
t = \frac{\bar x_t-\bar x_c}{\sqrt{s_t^2/n_t+s_c^2/n_c}},
\]

with Welch-Satterthwaite degrees of freedom

\[
\nu = \frac{(s_t^2/n_t+s_c^2/n_c)^2}
{(s_t^2/n_t)^2/(n_t-1)+(s_c^2/n_c)^2/(n_c-1)}.
\]

The test and interval use SciPy's t distribution; reference tests compare the results with
`scipy.stats.ttest_ind(equal_var=False)`. No equal-variance assumption is made. The two-sided
interval uses the `1-alpha/2` t quantile; one-sided intervals use `1-alpha` and an unbounded
opposite endpoint.

Each arm requires at least two observations. NaN and infinity raise by default. With the explicit
`nonfinite_policy="omit"`, they are removed and a diagnostic records counts. If both sample
variances are zero, Welch's reference distribution is undefined: the result reports the observed
point interval but no t statistic, degrees of freedom, or p-value. One zero-variance arm remains
computable and is flagged. Means and sample standard deviations are evaluated after scale
normalization, and the combined standard error uses a stable Euclidean norm; this avoids
unnecessary overflow and underflow for representable extreme magnitudes.

A relative mean difference is returned only when the treatment mean is non-negative and the
positive control mean exceeds `sqrt(machine epsilon)` times the largest absolute observation.
This scale-aware guard suppresses ratios dominated by floating-point cancellation while retaining
genuinely small ratio-scale measurements. It is a conservative runtime check, not proof of
ratio-scale measurement; callers must still decide whether ratios are scientifically meaningful.

Reference: B. L. Welch, “The generalization of ‘Student's’ problem when several different
population variances are involved,” *Biometrika* 34 (1947), 28–35,
[doi:10.1093/biomet/34.1-2.28](https://doi.org/10.1093/biomet/34.1-2.28).

## Directional alternatives

The alternatives are `two-sided`, `greater`, and `less`, always applied to treatment minus control.
Directional p-values are computed from the signed statistic; the implementation does not blindly
halve a two-sided p-value. Confidence intervals use the same direction and tail allocation as the
test. Welch intervals invert the corresponding Welch test. The pooled binary score test and
Newcombe interval are distinct score-based constructions, so their rejection and exclusion
boundaries need not agree exactly in finite samples.

## Sample ratio mismatch

Given observed counts `O_i`, total `N`, and strictly positive planned proportions `pi_i` summing to
one, expected counts are `E_i = N pi_i`. Pearson's statistic is

\[
X^2 = \sum_i (O_i-E_i)^2/E_i,
\]

compared with a chi-square distribution on `k-1` degrees of freedom through
`scipy.stats.chisquare`. The result exposes `k-1` degrees of freedom. If any expected count is below
five, the test is still computed but carries a diagnostic that the chi-square reference
approximation may be inaccurate. `p < significance_level` produces `FAIL`; equality does not. SRM
flags inconsistency with allocation but cannot diagnose its cause.

Reference: `scipy.stats.chisquare` in the
[SciPy reference guide](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html).

## Proportion power and sample size

Planning uses a fixed-horizon normal approximation. The allocation ratio `r = n_t/n_c`. Under the
alternative, rates are `p_c` and `p_t`; the pooled planning rate is
`p_bar = (p_c + r p_t)/(1+r)`. For a positive effect magnitude `delta`, the starting control size is

\[
n_c = \left\lceil
\frac{[z_{1-\alpha^*}\sqrt{\bar p(1-\bar p)(1+1/r)} +
z_{power}\sqrt{p_c(1-p_c)+p_t(1-p_t)/r}]^2}{\delta^2}
\right\rceil,
\]

where `alpha*` is `alpha/2` for two-sided and `alpha` for one-sided planning. Treatment size is
`ceil(r n_c)`. The implementation evaluates achieved power after integer rounding and increments
the control size until the requested approximate power is met, then checks smaller adjacent integer
designs so the returned control size is locally minimal for the configured allocation rule.

An absolute MDE is a probability-point difference. A relative MDE is multiplied by baseline. MDE
is a positive magnitude, added for `two-sided`/`greater` and subtracted for `less`. Rates must stay
strictly between zero and one. Achieved power models the null-standardized score statistic as a
normal variable using its alternative mean and variance. These calculations do not account for
attrition, clustering, repeated looks, or baseline-rate estimation uncertainty. An MDE that does
not change the target rate at float64 precision, or a design requiring group sizes above the exact
integer range of float64, is rejected rather than returned with spurious precision.

## Multiple testing

Bonferroni returns `min(m p_i, 1)`. Holm sorts p-values, scales ordered values by `m-i+1`, applies a
cumulative maximum, caps at one, and restores input order. Both control family-wise error rate.

Benjamini-Hochberg sorts p-values, computes `m p_(i)/i`, applies a reverse cumulative minimum,
caps at one, and restores input order. It controls false discovery rate under independence or
positive dependence. It does not generally control family-wise error rate.

References:

- Sture Holm, “A Simple Sequentially Rejective Multiple Test Procedure,” *Scandinavian Journal of
  Statistics* 6 (1979), 65–70, [JSTOR 4615733](https://www.jstor.org/stable/4615733).
- Yoav Benjamini and Yosef Hochberg, “Controlling the False Discovery Rate: A Practical and
  Powerful Approach to Multiple Testing,” *Journal of the Royal Statistical Society, Series B* 57
  (1995), 289–300, [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).

## Group-sequential monitoring

Monitoring uses one-sided Lan-DeMets alpha spending. With information fractions
`0 < t_1 < ... < t_K <= 1`, the z statistics follow the canonical joint distribution

\[
\operatorname{Corr}(Z_i, Z_j) = \sqrt{t_i / t_j}, \quad i \le j,
\]

equivalently a score process `S_k = Z_k sqrt(t_k)` with independent increments
`N(0, t_k - t_{k-1})` under the null. The O'Brien-Fleming-type spending function is

\[
\alpha(t) = 2 - 2\Phi\left(z_{1-\alpha/2}/\sqrt{t}\right)
\]

and the Pocock-type function is `alpha ln(1 + (e - 1) t)`; both spend exactly `alpha` at `t = 1`.
The boundary `c_k` solves

\[
P(Z_1 < c_1, \ldots, Z_{k-1} < c_{k-1}, Z_k \ge c_k) = \alpha(t_k) - \alpha(t_{k-1}).
\]

The probability is evaluated by recursive numerical integration of the score process
(Armitage, McPherson and Rowe; Jennison and Turnbull, chapter 19). The continuation distribution
is held as probability masses on a uniform score grid whose step is at most a quarter of the
smallest increment's standard deviation, within `[0.005, 0.02]`. Each step convolves it with the
exact cell probabilities of the Gaussian increment and truncates it at the boundary. Exit
probabilities use the exact normal tail and are solved in log space, so boundaries stay finite
when early O'Brien-Fleming increments are as small as `1e-30`. Masses below `1e-250` are dropped.
A single look reproduces the fixed-horizon critical value; for five equally spaced looks at
one-sided `alpha = 0.025` the boundaries are 4.877, 3.357, 2.680, 2.290 and 2.031
(O'Brien-Fleming) and 2.438, 2.427, 2.410, 2.397 and 2.386 (Pocock). Tests check the error rate
by simulation.

`analyze_sequential_continuous` analyzes the first `n_c,k` control and `n_t,k` treatment
observations at look `k`. The Welch statistic `t_k` with Welch-Satterthwaite degrees of freedom
`nu_k` is mapped to `z_k = Phi^{-1}(F_t(t_k; nu_k))`, which preserves its tail probability.
Information fractions use the arm variances of all supplied observations and the planned maximum
sizes:

\[
t_k = \frac{s_c^2/N_c + s_t^2/N_t}{s_c^2/n_{c,k} + s_t^2/n_{t,k}}.
\]

A look whose subsamples are both constant has no defined statistic and raises. Monitoring stops
at the first crossing, and the result reports only the looks analyzed. Monitoring for harm uses
`alternative="less"` (or `"greater"` for a lower-is-better metric). The procedure is a test; the
mean difference at a crossing look is biased away from zero and is not reported as an estimate.

References:

- K. K. Gordon Lan and David L. DeMets, “Discrete sequential boundaries for clinical trials,”
  *Biometrika* 70 (1983), 659–663, [doi:10.1093/biomet/70.3.659](https://doi.org/10.1093/biomet/70.3.659).
- Christopher Jennison and Bruce W. Turnbull, *Group Sequential Methods with Applications to
  Clinical Trials*, Chapman & Hall/CRC (2000), chapters 7 and 19.
- P. Armitage, C. K. McPherson and B. C. Rowe, “Repeated significance tests on accumulating data,”
  *Journal of the Royal Statistical Society, Series A* 132 (1969), 235–244,
  [doi:10.2307/2343787](https://doi.org/10.2307/2343787).

## Serialization and numerical policy

Computations use NumPy float64 and SciPy primitives. Inputs are validated before inference and are
never rounded internally. Pydantic result models reject NaN and infinity. NumPy scalar inputs are
normalized to Python scalar fields. Optional and unbounded quantities serialize as JSON `null`.
