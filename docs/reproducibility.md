# Reproducing the experiments

This guide distinguishes three operations: verifying the supplied reference artifacts, regenerating analyses from those artifacts, and generating new simulated data and fitted models. Each operation has a different computational cost and evidential purpose.

## Environment and installation

The recorded numerical environment is Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8 on Linux x86_64. Install the pinned dependencies and the package in an isolated environment using the commands in the root README. Execute the following commands from the repository root. The editable installation retains access to the reference files distributed with this checkout.

Record the Git commit, Python version, dependency versions, operating system, and relevant hardware when reporting a new run. Fixed seeds control pseudorandom inputs; they do not make runtime measurements, PDF creation metadata, or floating-point behavior identical across all platforms and dependency versions.

## 1. Check the installation and reference artifacts

```bash
opr smoke
opr validate
```

The smoke run checks that the package can fit a model and exercise its numerical interfaces. The validation run checks the frozen reference records and their integrity. Numerical checks support implementation consistency; the mathematical privacy argument concerns every neighboring data set and cannot be established by a finite collection of tests.

Reference inputs are read from `data/reference/`. Generated files belong under a separate `outputs/` directory. Validation must not silently repair a frozen input or replace it with newly computed values. A mismatch should be investigated against the protocol, source revision, and recorded environment.

## 2. Regenerate summaries, figures, and tables

```bash
opr reproduce --output-dir outputs/reference-reproduction
```

This workflow uses the recorded fit-level and runtime measurements, recomputes the statistical summaries, and regenerates the six figure families and two summary tables. It also recomputes the small exact population and output-count calculations. The numerical comparisons concern measured risks, exact oracle values, interval estimates, and plotted quantities; a PDF file hash is not an appropriate test of numerical equality.

The principal expected learning outputs are 300 method summaries and 250 paired comparisons. The full protocol contains 50 configurations and six methods. Each configuration has 256 independent replicates; paired comparisons use the same replicate for the two methods being compared. Repeated parameter settings in different sweeps have distinct configuration identifiers and independent training samples.

The six figure families are:

| Figure stem | Scientific quantity |
|---|---|
| `representation_geometry` | Exact coarsening loss and privacy-dependent approximation scales |
| `learning_curves` | Learning behavior under sample-size, alphabet-size, and privacy-budget sweeps |
| `ablation_curves` | Effects of pilot ordering and separate mean calibration |
| `risk_and_local_alternatives` | Absolute risk and normalized excess for sample-size-dependent local alternatives |
| `near_tie_decomposition` | Population channel excess and decoder-estimation components |
| `runtime_and_memory` | Uncapped fitting time and traced allocation across alphabet sizes |

The reference images use the same numerical data as the experiment scripts, with presentation adjustments to improve printed labels. See the generated numerical tables when an exact value is needed.

## 3. Generate a new learning run

```bash
opr simulate --replicates 256 --output-dir outputs/full-simulation
opr summarize --data-dir outputs/full-simulation/data --output-dir outputs/full-analysis
```

The complete protocol generates $50\times256=12{,}800$ independent training samples and evaluates six fitted methods on each sample, for 76,800 fits. A smaller replicate count is useful for checking the workflow, but its outputs are a new reduced simulation and do not reproduce the uncertainty of the 256-replicate reference study.

`summarize` reads the new run's protocol and raw fit measurements, writes recomputed method summaries and paired comparisons, and generates the three learning/ablation/absolute-risk figure families in a separate output directory. It does not require a frozen-reference hash match and does not modify the input run. In contrast, `reproduce` verifies and rebuilds the distributed reference study. Population calculations, decomposition, and runtime measurements remain separate workflows.

All algorithms in the recorded comparison use the uncapped experiment implementation. The public learner API uses the capped optimizer. Changing the experiment to the capped implementation is a new implementation comparison, even when its optimized risk agrees numerically. In particular, the historical timing data cannot be relabeled as measurements of the capped optimizer.

The root seed, configuration identifiers, seed spawning, and per-method coupling are specified in the protocol. Synthetic methods share both the training sample and the initial fitting-noise seed within a replicate. This coupling supports paired numerical comparisons. It is not a privacy claim for jointly releasing all six fits on confidential data.

The targets are conditionally Bernoulli. The fitted model is evaluated by integrating the true finite law and randomized query channel analytically, rather than adding an additional test-set Monte Carlo error. The saved raw records therefore contain fit-level risk measurements, not the individual synthetic training observations.

## 4. Recompute exact population calculations and diagnostics

The `population` command recomputes exact finite-law representation-loss and output-count calculations. The `decompose` command reconstructs the local-alternative fitted models from their recorded seeds and computes the population Bayes redecoding diagnostic. Consult `opr population --help` and `opr decompose --help` for their input and output options.

The main population calculation evaluates 930 law–statistic–budget cases and compares the original and coarsened oracles with an independent floating-point optimizer, yielding 1,860 comparisons. Positive rational losses below float64 resolution are retained as exact references and used in the corresponding figures.

The decomposition replays 15,360 existing fits from three methods on the 20 local-alternative configurations. These replays are not new independent samples. The population decoder is used only for diagnostic evaluation and is not supplied to the fitted learner.

## 5. Measure runtime separately

The `benchmark` command performs serial fitting measurements with data generation excluded, one discarded warm-up, and nine timed repetitions. Run it after the scientific calculations and avoid concurrent computational work. Consult `opr benchmark --help` for the output option.

The reference benchmark fixes $n=8192$, $\alpha=1$, and $\varepsilon=\log4$, and varies the input alphabet size. The allocation trace is a separate measurement of allocations during fitting; it is not total resident process memory. Runtime and allocations depend on the implementation, platform, numerical libraries, and measurement procedure.

Timing fields are internal experimental diagnostics. They are not included in the claimed private model release.

## Comparison criteria

| Artifact or property | Appropriate comparison |
|---|---|
| Frozen downloaded reference file | Exact SHA-256 equality |
| Configuration identifiers, methods, replicates | Exact equality and uniqueness of expected keys |
| Rational oracle values | Exact equality of rational values |
| Regenerated risk summaries and intervals | Numerical agreement with stated tolerances |
| Same-seed fitted scientific outputs | Agreement of risk, output count, and model quantities within numerical tolerance |
| PDF or PNG output | Plot content, labels, and numerical quantities; binary metadata may vary |
| Runtime measurements | Protocol and distribution of measurements; exact equality is not expected |

The reference protocol and data dictionary define the expected counts and field meanings. The maintenance guide specifies how a scientifically justified change becomes a new version rather than silently replacing the frozen study.
