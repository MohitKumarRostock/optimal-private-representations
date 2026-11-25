# Data dictionary

All numerical references in [data/reference](../data/reference) describe synthetic laws, fitted-model measurements, or aggregate diagnostics. No individual training observations are stored. See [experiments.md](experiments.md) for the laws, sampling, seed recipe, methods, and interpretation.

The 20 frozen files total **27,207,118 bytes**. [reference_data_manifest.json](../provenance/reference_data_manifest.json) records SHA-256 hashes, byte lengths, ordered CSV column names, row counts, and JSON top-level keys. Row counts below exclude headers.

## File inventory

| File | Rows / type | Observation or purpose |
|---|---:|---|
| `learning_protocol.json` | JSON object | Main configuration grid, seed, method list, original source hashes, environment |
| `learning_raw.csv` | 76,800 | One fitted method on one configuration–replicate sample |
| `learning_run.json` | JSON object | Original run duration, new-fit count, protocol/raw hashes |
| `learning_summary.csv` | 300 | One configuration–method summary |
| `paired_comparisons.csv` | 250 | One configuration–comparator paired difference |
| `near_tie_display_summary.csv` | 15 | Equal-sign normalized excess for $c=1$, by sample size and method |
| `summary_integrity.json` | JSON object | Learning counts, ranges, and interval interpretation |
| `near_tie_decomposition_raw.csv` | 15,360 | Diagnostic replay of one existing near-tie fit |
| `near_tie_decomposition_summary.csv` | 30 | Equal-sign diagnostic summary by $n,c,$ method |
| `near_tie_decomposition_integrity.json` | JSON object | Replay equality, risk identities, hashes, and post-run status |
| `geometry_three_point.csv` | 5 | One exact law–statistic comparison |
| `geometry_exact_four_state.csv` | 28 | One dyadic-parameter exact law–statistic comparison |
| `geometry_modulus_grid.csv` | 897 | One privacy-ratio–reconstruction-budget law |
| `geometry_checks.json` | JSON object | Exact/float comparison checks and numerical interpretation |
| `output_count.csv` | 6 | One privacy ratio in the additional output-count diagnostic |
| `output_count.json` | JSON object | Diagnostic status, exact conclusions, hashes, and duplicate structured rows |
| `runtime_raw.csv` | 120 | One timing or separate memory-trace call |
| `runtime_summary.csv` | 12 | One alphabet–method timing/allocation summary |
| `runtime_metadata.json` | JSON object | Benchmark protocol and platform |
| `reported_results.json` | JSON object | Selected comparisons used by the summary table |

## Parsing conventions

CSV files have one header row, comma separators, standard CSV quoting, and decimal float strings. Treat `config_id`, `config_minus`, and `config_plus` as strings: leading zeros matter. Integer-valued configuration columns may be written as `2048.0` in processed tables; parse numerically and validate integrality if integer types are needed.

State labels and replicate numbers are zero-based. Privacy budgets are dimensionless and use natural logarithms. Risk, variance, squared-error, and deficiency fields are in squared target units; the target is bounded in $[0,1]$. Entropies are in nats, time is in seconds, and memory is in bytes unless a rendered table explicitly converts to milliseconds or MiB ($2^{20}$ bytes).

Exact rational scalars are strings such as `"37/240"` or `"0"`, parsed with `fractions.Fraction`. Array-valued CSV cells are JSON text. Parse the CSV first, then `json.loads` the cell, then convert exact rational elements if needed. Partition arrays contain integer category labels.

CSV logical flags are the strings `True` and `False`. Replay category indicators are integers `0` or `1`. JSON uses native booleans and null. Some numeric values inside `reported_results.json` remain strings because they were copied from CSV records.

There are only two intentional kinds of blank CSV values:

- In `geometry_modulus_grid.csv`, the two `deficiency_over_gamma_*` ratios are blank when their denominator is zero (131 rows in each column).
- In `runtime_raw.csv`, `seconds` is blank on 12 memory-trace rows and `peak_traced_bytes` is blank on 108 timed rows. A blank is not a measured zero.

In decomposition summaries, a `*_welch_df` of `inf` denotes zero observed within-sign variance and a zero-width moment-based interval. It does not establish that an event has exactly zero or one population probability. Confidence interval limits are not clipped to the admissible range of a risk or frequency.

## Learning measurements

### `learning_raw.csv`

Unique key: `(config_id, replicate, method)`. There are 50 configurations, 256 replicates per configuration, and six methods. The six rows for one `(config_id, replicate)` share a synthetic training sample.

| Columns | Type and meaning |
|---|---|
| `config_id` | Three-digit string joining to `learning_protocol.json` |
| `study` | `sample_size`, `training_budget`, `query_budget`, `alphabet_size`, `mean_calibration`, or `near_tie` |
| `family` | `permuted_gradient`, `rare_low_mean`, or `near_tie` |
| `n`, `d` | Public training sample size and full declared input alphabet size |
| `alpha`, `epsilon` | Requested central training and local query privacy budgets; `nonprivate_fit` ignores `alpha` |
| `amplitude`, `sign` | Near-tie $c\in\{0.5,1\}$ and $s\in\{-1,+1\}$; both zero outside that family |
| `replicate` | Integer `0`–`255` |
| `method` | One of `attaining`, `direct_histogram`, `private_mean`, `no_mean_calibration`, `public_order`, `nonprivate_fit` |
| `risk` | Analytically integrated float64 population risk of the fitted randomized predictor |
| `oracle_risk` | Known-law, query-private population optimum computed with true conditional-mean order |
| `excess` | `risk - oracle_risk` |
| `k`, `oracle_k` | Fitted and known-law RR output counts |
| `fit_seconds` | Elapsed fit-call time during the main simulation; internal diagnostic, distinct from the serial benchmark |
| `a` | $\tanh^2(\varepsilon/2)$ |
| `rate_scale` | Numerical theoretical comparison scale defined below; not a fitted or exact risk prediction |
| `target_mean` | Known population $\sum_i p_i m_i$ |
| `missing_cells` | Number of declared states absent from this training sample; same across its six methods |

The saved scale is

$$
\frac1n+\frac a{\sqrt n}
+\min\{a,a\sqrt{d/n},(ad/n)^{2/3}\}
+\min\{1,t^{-2}\}
+a\min\{1,d/t\},
\qquad t=n\tanh(\alpha/2).
$$

When $t=0$, the final two contributions are defined as $1$ and $a$. This field remains a common configuration descriptor on `nonprivate_fit` rows; it is not a training-private guarantee for that comparator.

### `learning_summary.csv`

Unique key: `(config_id, method)`. Configuration descriptors, `oracle_risk`, `oracle_k`, `a`, `rate_scale`, and `target_mean` retain their meanings above. `replicates` is 256.

For each metric in `risk`, `excess`, and `k`, the four columns `<metric>_mean`, `<metric>_se`, `<metric>_ci_low`, and `<metric>_ci_high` give the replicate mean, empirical standard error, and pointwise 95% Student-$t$ interval. `constant_output_fraction` is the proportion with `k == 1`. `mean_missing_cells` averages the count of absent states.

### `paired_comparisons.csv`

Unique key: `(config_id, method)`, excluding `attaining` itself.

| Columns | Meaning |
|---|---|
| `config_id`, `study`, `method` | Configuration and comparator |
| `comparison` | Literal `method minus attaining` |
| `mean`, `se`, `ci_low`, `ci_high` | Mean and pointwise Student-$t$ interval for the within-replicate risk difference |
| `replicates` | Number of paired differences, 256 |

Positive differences mean the comparator has higher risk than `attaining`.

### `near_tie_display_summary.csv`

Unique key: `(n, method)`. This small display table fixes **amplitude $c=1$**, uses the three methods `attaining`, `direct_histogram`, and `nonprivate_fit`, and combines both signs equally.

`mean`, `se`, `lo`, and `hi` describe $\sqrt n\,\mathrm{excess}/a$. `df` is the Welch–Satterthwaite degrees of freedom for fixed sign strata. `replicates` is 512 in total, with 256 per sign. The file does not contain all six methods or both amplitudes.

## Diagnostic replay and risk decomposition

### `near_tie_decomposition_raw.csv`

Unique key: `(config_id, replicate, method)`, joining directly to the corresponding learning row. The fields `n`, `d`, `alpha`, `epsilon`, `amplitude`, `sign`, `method`, `a`, and `k` have the learning meanings. `c` is a redundant copy of `amplitude`.

Let $V$ denote the population oracle, $g$ the fitted decoder, and $g_P$ the population Bayes decoder for the **same fitted channel**.

| Columns | Meaning |
|---|---|
| `risk`, `original_risk` | Replayed fitted risk and value read from the original learning row |
| `replay_risk_error` | `risk - original_risk`; zero for all reference rows |
| `oracle_risk` | $V$ |
| `population_redecoded_risk` | $R(Q,g_P)$, using the known population solely for evaluation |
| `total_excess` | $R(Q,g)-V$ |
| `channel_excess` | $R(Q,g_P)-V$ |
| `decoder_error` | $R(Q,g)-R(Q,g_P)$ |
| `decoder_squared_distance` | Independently evaluated $\sum_z P_Q(z)(g_z-g_{P,z})^2$ |
| `decomposition_identity_error` | `total_excess - channel_excess - decoder_squared_distance` |
| `decoder_identity_error` | `decoder_error - decoder_squared_distance` |
| `normalized_total_excess`, `normalized_channel_excess`, `normalized_decoder_error` | Corresponding quantity multiplied by $\sqrt n/a$ |
| `canonical_assignment` | String of input-to-bin labels, relabeled by first occurrence; e.g. `0011`. Preserve leading zeros |
| `partition_category` | One of the five categories below |
| `binary_correct`, `binary_wrong`, `binary_other`, `constant`, `multibin` | Exclusive `0/1` category indicators summing to one |
| `source_wrong_gap` | Population risk of the opposite source partition minus $V$; equals $|\theta|/60$ in exact arithmetic |
| `normalized_source_wrong_gap` | Previous field multiplied by $\sqrt n/a$ |
| `fitted_decoder`, `population_decoder` | JSON arrays in the original channel-column order |

The correct source assignment is `0111` for positive sign and `0001` for negative sign. `binary_wrong` is the opposite source assignment, `binary_other` covers other two-output assignments, `constant` means one output, and `multibin` means more than two outputs.

**Column-order caution:** `canonical_assignment` deliberately removes output-label permutations. The two decoder arrays retain the original fitted channel's output order. The canonical labels and decoder arrays cannot always be combined directly to reconstruct the fitted predictor; replay the fit to recover the original channel ordering.

### `near_tie_decomposition_summary.csv`

Unique key: `(n, c, method)`. Each row combines two sign configurations, 256 fits per sign.

| Columns | Meaning |
|---|---|
| `n`, `c`, `amplitude`, `method`, `alpha`, `epsilon`, `a` | Configuration descriptors |
| `config_minus`, `config_plus` | Source IDs for the two signs |
| `replicates_per_sign`, `replayed_fits` | 256 and 512 |
| `source_wrong_gap_mean`, `normalized_source_wrong_gap_mean` | Equal-sign mean reference costs |

For each metric in

```text
risk, population_redecoded_risk,
total_excess, channel_excess, decoder_error,
normalized_total_excess, normalized_channel_excess, normalized_decoder_error,
k, binary_correct, binary_wrong, binary_other, constant, multibin
```

the columns `<metric>_mean`, `<metric>_se`, `<metric>_ci_low`, `<metric>_ci_high`, `<metric>_welch_df`, `<metric>_minus_mean`, and `<metric>_plus_mean` give the equal-sign estimate, uncertainty, and the two separate sign means. The suffix `<category>_count` for each of the five categories counts its occurrences among 512 fits.

Indicator intervals use the same approximate moment calculation as continuous metrics. The saved `inf` degrees of freedom and zero-width intervals for unobserved categories reflect zero empirical variance only.

## Exact representation geometry

All three `geometry_*.csv` files share the following columns. The suffix `h` refers to the full input; `t` refers to the retained/coarsened statistic. Here $\mu=\mathbb E Y$, $\Phi=\operatorname{Var}(\mathbb E[Y\mid Z])$, and risk is $\mathbb E Y^2-\mu^2-\Phi$.

### Common geometry columns

| Columns | Meaning |
|---|---|
| `case_id` | Unique case string within the combined geometry data |
| `target_model` | `Y=M(H)` for deterministic targets or `Y\|H~Bernoulli(M(H))` for Bernoulli targets |
| `epsilon`, `exp_epsilon_exact` | Float $\log r$ and exact rational $r$; the latter defines the exact privacy primitive |
| `a`, `a_exact` | Float and exact $((r-1)/(r+1))^2$ |
| `input_states`, `statistic_states` | Original and retained alphabet sizes |
| `probabilities_exact`, `conditional_means_exact` | JSON arrays of exact rational strings for $p_i,m_i$ |
| `coarsening` | JSON integer array defining the deterministic retained category of each input |
| `statistic_probabilities_exact`, `statistic_means_exact` | Exact retained-law probabilities and conditional means, in retained-label order |
| `target_second_moment`, `target_second_moment_exact` | Float and exact unconditional $\mathbb E Y^2$ |
| `mse`, `mse_exact` | Actual $\mathbb E(M-\mathbb E[M\mid T])^2$ |
| `l1`, `l1_exact` | Actual $\mathbb E|M-\mathbb E[M\mid T]|$ |
| `linf` | Actual maximum absolute conditional-mean reconstruction error over positive-mass states |
| `risk_h_reference`, `risk_t_reference` | Float renderings of the independently enumerated exact optimal risks |
| `risk_h_exact`, `risk_t_exact` | Exact optimal risks |
| `phi_h_reference`, `phi_t_reference` | Float renderings of exact optimal explained variances |
| `phi_h_exact`, `phi_t_exact` | Exact explained variances |
| `deficiency_reference`, `deficiency_exact` | Float rendering and exact full-versus-retained oracle risk gap |
| `deficiency_exact_status` | `exact_zero` or `exact_positive` |
| `risk_h_float64`, `risk_t_float64`, `phi_h_float64`, `phi_t_float64` | Raw ordered-optimizer values using float approximations of the law |
| `deficiency_float64` | Raw `phi_h_float64 - phi_t_float64`; not clipped to nonnegative values |
| `deficiency_float64_resolution` | $64\,\epsilon_{\rm machine}\max(1,\mathbb E Y^2)$ |
| `deficiency_float64_resolved` | Whether the absolute float gap exceeds that conservative threshold |
| `k_h_reference`, `k_t_reference` | Fewest RR outputs among tied exact optima |
| `k_h_float64`, `k_t_float64` | Outputs selected by the float optimizer, potentially a different tie choice |
| `partition_h_reference`, `partition_t_reference` | Selected exact partitions, as JSON arrays |
| `optimal_partitions_h_exact`, `optimal_partitions_t_exact` | JSON lists of all exactly tied optimum partitions |
| `decoder_h_exact`, `decoder_t_exact` | JSON arrays of exact rational Bayes decoder values |
| `partition_h_float64`, `partition_t_float64` | Selected float-optimizer assignments |
| `decoder_h_float64`, `decoder_t_float64` | Selected float decoder arrays |
| `partitions_checked_h`, `partitions_checked_t` | Number of set partitions enumerated: Bell numbers 1, 2, 5, or 15 |
| `dp_exhaustive_phi_error_h`, `dp_exhaustive_phi_error_t` | Absolute float-versus-exact explained-variance errors |
| `dp_exhaustive_check_pass` | Both errors within $2\times10^{-13}$ |
| `max_rr_row_sum_error` | Largest stochastic-row sum discrepancy across full/coarse float channels |
| `max_rr_column_ratio` | Largest columnwise probability ratio across full/coarse float channels |

Exact tied partitions use canonical restricted-growth labels and select the fewest outputs, then lexicographically smallest labels. A numerical tie can choose a different optimizer without a material objective discrepancy.

Nine positive exact gaps have `deficiency_float64_resolved=False`. Those are the four-state cases with dyadic powers 22 through 30. The flag describes subtraction resolution, not underflow and not exact equality. Use `deficiency_exact` or its `reference` rendering for these laws.

### Additional three-point columns

| Columns | Meaning |
|---|---|
| `statistic` | `identity`, `sufficient_left`, `sufficient_right`, `boundary_crossing`, or `constant` |
| `j_left`, `j_left_exact` | Float/exact fixed-codebook Jensen penalty for decoder $(3/8,3/5)$ |
| `j_right`, `j_right_exact` | Same penalty for decoder $(2/5,5/8)$ |
| `statistic_entropy_nats` | $-\sum_t P(T=t)\log P(T=t)$ |

The fixed-codebook penalties are not in general the reoptimized deficiency.

### Additional exact four-state columns

| Columns | Meaning |
|---|---|
| `t`, `t_exact`, `dyadic_power` | Parameter $t=2^{-\text{dyadic_power}}$, with powers 3–30 |
| `deficiency_over_mse_two_thirds` | Exact gap divided by actual MSE to the $2/3$ power, rendered as float |
| `deficiency_over_t_squared_exact` | The same ratio as an exact rational, since MSE=$t^3$ |
| `asymptotic_coefficient`, `asymptotic_coefficient_exact` | $35/324$ |
| `coefficient_relative_error` | Absolute relative error of the observed ratio from $35/324$ |
| `exact_formula_check_pass` | Exact oracle and explicit finite-$t$ formula agree |

### Additional modulus-grid columns

| Columns | Meaning |
|---|---|
| `budget_v`, `budget_v_exact` | Allowed reconstruction-error budget, not necessarily the actual `mse` |
| `w`, `h`, `w_exact`, `h_exact` | Float/exact parameters of the constructed four-state law |
| `construction_regime` | `zero_budget`, `small_budget_two_thirds_construction`, `middle_budget_square_root_construction`, or `saturated_construction` |
| `gamma_branch` | Minimizing branch of $\Gamma_a(v)$: `zero`, `constant`, `square_root`, or `two_thirds` |
| `gamma_budget`, `gamma_actual_mse` | $\Gamma_a$ evaluated at allowed budget or actual MSE |
| `deficiency_over_gamma_budget`, `deficiency_over_gamma_actual_mse` | Gap divided by the corresponding scale; blank at zero denominator |
| `theorem_lower_budget` | Numerical lower comparison $(37/1920)\Gamma_a(v)$; historical field name retained |
| `theorem_upper_actual_mse` | $\min\{a/4,a\sqrt{\mathrm{mse}},6(a\,\mathrm{mse})^{2/3}\}$; historical field name retained |
| `source_feasible_phi_h`, `source_feasible_phi_h_exact` | Feasible original-input comparator $a((1-w)/4+wh)^2$; need not be globally optimal |
| `source_lower_bound`, `source_lower_bound_exact` | $(37/120)awh$ |
| `source_phi_h_comparator_globally_optimal` | Exact equality of the feasible comparator with the optimized full-input value |
| `source_phi_t_formula_check_pass` | Exact retained oracle agrees with $a(1-w)^2/[16(1-aw^2)]$ |
| `source_lower_bound_check_pass` | Exact gap satisfies the recorded source lower comparison |
| `actual_mse_within_budget_exact` | Exact $wh^2\le v$ |

The regime labels and exact laws are fully reproducible from [experiments.md](experiments.md). The construction saturates at actual MSE $1/64$; entries with larger budgets are not examples having those larger actual errors.

## Output-count comparison

`output_count.csv` has one row for each exact ratio $r\in\{1,5/4,3/2,2,3,4\}$.

| Columns | Meaning |
|---|---|
| `exp_epsilon_exact`, `epsilon` | Exact $r$ and float $\log r$ |
| `probabilities_exact`, `conditional_means_exact`, `target_model`, `target_variance_exact` | Exact population law and $\operatorname{Var}(Y)=1/4$ |
| `optimal_k`, `best_at_most_two_k` | Minimum output counts among tied unrestricted/all-RR or at-most-two-RR optima |
| `optimal_partition`, `best_at_most_two_partition` | Selected partitions |
| `optimal_partitions_exact`, `best_at_most_two_optimal_partitions` | All exactly tied partitions in each optimization |
| `optimal_decoder_exact`, `best_at_most_two_decoder_exact` | Exact Bayes decoders |
| `phi_unrestricted_exact`, `phi_at_most_two_exact` | Exact explained variances |
| `risk_unrestricted_exact`, `risk_at_most_two_exact` | Exact risks |
| `binary_restriction_gap_exact` | Restricted risk minus unrestricted risk |
| `phi_unrestricted`, `phi_at_most_two`, `risk_unrestricted`, `risk_at_most_two`, `binary_restriction_gap` | Float renderings of the corresponding exact quantities |
| `partitions_checked_unrestricted`, `partitions_checked_at_most_two` | 15 and 8 |
| `exact_enumeration_check_pass` | Exact assertions passed |

“Unrestricted” uses the scalar RR structural reduction. “At most two” restricts the RR-partition family; it is not an additional optimization over arbitrary binary-output private channels.

## Runtime and memory

### `runtime_raw.csv`

Unique key: `(d, method, replicate)`. `d` is one of 4, 8, 16, 32, 64, 128; `n` is 8192; `method` is `attaining` or `direct_histogram`.

| Columns | Meaning |
|---|---|
| `replicate` | String `0`–`8` for timed calls, or `memory` for a separate trace |
| `seconds` | Fit-call wall-clock duration; blank for memory rows |
| `k` | Output count from that fit |
| `peak_traced_bytes` | Peak traced fitting allocations; populated only for memory rows |

Warm-up calls are excluded. The memory-traced call uses separate fitting randomness and is never used as a timed repetition.

### `runtime_summary.csv`

Unique key: `(d, method)`. `median_seconds`, `q25_seconds`, `q75_seconds`, `min_seconds`, and `max_seconds` summarize the nine timed calls. `repeats` is nine. `peak_traced_bytes` is copied from the separate memory call. Quartiles are empirical distribution summaries, not confidence limits.

## JSON metadata

JSON provenance is preserved verbatim, including original source basenames and development labels. Those names are historical metadata, not filesystem dependencies. [code_migration.json](../provenance/code_migration.json) links original source hashes to the portable package.

| File | Important keys and interpretation |
|---|---|
| `learning_protocol.json` | `master_seed`, `replicates_per_configuration`, `methods`, and `configurations` define the experiment. `law_definition`, `risk`, `coupling`, `interval`, and `interpretation` describe semantics. `source_hashes` identifies the original learner and calibration sources; `environment` records Python/NumPy/platform. `title` and `date` describe the original run |
| `learning_run.json` | `new_fits`, `elapsed_seconds`, and `resumed` describe that invocation. `protocol_sha256` and `raw_sha256` bind it to the saved files. Duration is not a portable performance promise |
| `summary_integrity.json` | `expected_rows`, `observed_rows`, `unique_fits`, `groups`, `replicates_per_group`, `minimum_excess`, `maximum_risk`, `intervals`, and `paired_sign` |
| `near_tie_decomposition_integrity.json` | `analysis_status` records post-run status. Count fields distinguish replayed fits from zero additional independent models. `maximum_errors`, minimum component values, `identity`, `decoder`, `normalization`, partition definitions, `intervals`, `elapsed_seconds`, and `hashes` document replay and analysis |
| `geometry_checks.json` | `deterministic=true` and `random_seed=null`. `source_provenance` and hashes identify original calculations. `exact_reference_method`, `oracle_scope`, `floating_method`, `file_row_counts`, `checks`, `interpretation_limits`, and `columns` document scope and numerical status |
| `output_count.json` | `status`, `purpose`, `provenance`, `method`, `unrestricted_oracle_justification`, `all_checks_pass`, and `limits` describe the added diagnostic. `exp_epsilon_4_conclusion` contains exact headline values. Hashes bind code and CSV; `rows` duplicates the CSV information |
| `runtime_metadata.json` | `repeats`, `n`, `alpha`, `epsilon`, `methods`, `alphabet_sizes`, `timing`, `memory`, `scheduling`, and recorded platform fields define the uncapped benchmark |
| `reported_results.json` | `main_configuration_ids` selects `004` and `028` for the comparison table. `balanced_oracle`, `low_mean_oracle`, `low_target_mean`, and `paired_results` are selected summaries, not a new experiment |

The raw learning and replay CSVs are the basis for recalculating their summaries. Exact rational geometry fields are the basis for interpreting tiny population gaps. The frozen metadata preserve what was actually run; newly generated outputs should record their own environment and source revision rather than replace that history.

