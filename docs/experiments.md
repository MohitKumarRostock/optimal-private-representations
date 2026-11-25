# Experiment design and interpretation

This repository contains controlled synthetic studies of learning a randomized private channel for one bounded scalar prediction task. The input encoder and its finite alphabet are public. Training records are complete input–target pairs; the central training budget is `alpha`, and the local privacy budget for each randomized fresh-input release is `epsilon`. All logarithms are natural.

The experiments address population information loss, finite-sample learning, model components, and implementation cost. They are examples and implementation checks, not an empirical minimax supremum or a numerical proof of privacy. Distribution-dependent rankings are expected: the pilot-and-calibration method is not uniformly best on the recorded laws.

## What is saved

The [frozen numerical references](../data/reference) contain **76,800 distinct fitted-model measurements**, from **50 configurations × 256 independent training samples × 6 methods**. There are 12,800 distinct training samples. Each row of `learning_raw.csv` is a fit, not an individual training record.

The individual simulated observations and complete fitted channel matrices are not stored. They can be regenerated from the public law, configuration identifier, replicate number, seed recipe, and recorded implementation. A separate diagnostic replays 15,360 of the existing fits; these do not increase the number of independent samples or distinct models in the main study.

The [data dictionary](data_dictionary.md) gives every file's schema and units. The [reproducibility guide](reproducibility.md) gives installation and execution instructions. The [methods guide](methods.md) defines the algorithms and privacy model.

## Main learning protocol

For each configuration, draw states $H_\ell$ independently with probabilities $p_i$, then targets $Y_\ell\mid H_\ell=i\sim\operatorname{Bernoulli}(m_i)$. State indices run from zero to $d-1$. The recorded grid is fully specified in [learning_protocol.json](../data/reference/learning_protocol.json); it was saved before the main run.

### Balanced graded family

Input probabilities are uniform. Let $b_d(i)$ reverse the bits of the zero-padded binary expansion of $i$, using $\log_2 d$ bits. The conditional means are

$$
m_i=0.05+\frac{0.90\,b_d(i)}{d-1}.
$$

The tested alphabet sizes are powers of two. The permutation prevents numerical input order from coinciding with conditional-mean order.

The default is $(n,d,\alpha,\varepsilon)=(2048,16,1,\log4)$. Each row below changes one parameter and holds the others at their defaults.

| Configuration IDs | Study field | Parameter values | Number of configurations |
|---|---|---|---:|
| `000`–`007` | `sample_size` | $n=128,256,512,1024,2048,4096,8192,16384$ | 8 |
| `008`–`013` | `training_budget` | $\alpha=0,0.05,0.1,0.25,0.5,1$ | 6 |
| `014`–`020` | `query_budget` | $\varepsilon=0,0.1,0.25,0.5,\log2,1,\log4$ | 7 |
| `021`–`025` | `alphabet_size` | $d=4,8,16,32,64$ | 5 |

Repeated defaults in different sweeps have distinct configuration IDs and independent samples. Their estimates need not coincide.

### Uneven low-mean family

Configurations `026`–`029` use 32 states with

$$
p_i=\frac{(i+1)^{-1.3}}{\sum_{j=0}^{31}(j+1)^{-1.3}},
\qquad
m_i=0.02+\frac{0.36\,b_{32}(i)}{31}.
$$

The sample sizes are $128,512,2048,8192$, with $\alpha=0.25$ and $\varepsilon=\log2$. This law combines uneven cell frequencies and a target mean near $0.1306350949$. Its known-law query-private oracle risk is approximately $0.1125124219$. The family tests the practical effect of separate mean calibration, including settings in which a private constant predictor is competitive.

### Local alternatives near a partition tie

Configurations `030`–`049` use

$$
p=(1/3,1/6,1/6,1/3),\qquad
m_\theta=(1/4,1/2+\theta,1/2+\theta,3/4),
\qquad \theta=\frac{s c}{\sqrt n}.
$$

Here $s\in\{-1,+1\}$, $c\in\{0.5,1\}$, $n\in\{256,1024,4096,16384,65536\}$, $\alpha=1$, and $\varepsilon=\log2$. IDs advance first over increasing $n$, then $c=0.5,1$, then $s=-1,+1$.

The oracle risk has the closed form

$$
V=\frac14-\frac{\theta^2}{9}
-\frac{(3+4|\theta|)^2}{2880}.
$$

The optimal source partition is `0111` for positive $\theta$ and `0001` for negative $\theta$. Using the opposite source partition and its population decoder costs $|\theta|/60$ in risk. Other binary partitions are possible and are retained separately in the diagnostic.

These laws change with $n$. A curve of normalized excess on this family does not establish a limiting exponent or a worst-case rate over all laws.

## Six fitted methods

All recorded learning and timing results use the **uncapped experiment implementation**, which searches all output counts up to the number of distinct sanitized score groups. The public estimator also offers a capped optimizer; its timings were not measured in the frozen study. See [methods.md](methods.md) for that distinction and the complete accounting.

Write $\bar\alpha=\min(1,\alpha)$, $n_1=\lfloor n/2\rfloor$, and $N=n-n_1$.

| Method identifier | Records and fitting procedure | Interpretation |
|---|---|---|
| `attaining` | Private pilot on the first $n_1$ records; histogram and separate mean calibration on the remaining $N$ | Reference pilot-and-calibration learner |
| `direct_histogram` | All-$n$ normalized fractional histogram, independent Laplace scale $2/(n\bar\alpha)$, clipping and normalization; order and optimize using sanitized means | Avoids the sample split |
| `private_mean` | All-$n$ mean plus Laplace scale $1/(n\bar\alpha)$, clipped to $[0,1]$; constant query channel | Ignores the input |
| `no_mean_calibration` | Same pilot and split as `attaining`; calibration histogram uses scale $2/(N\bar\alpha)$ and no separate mean release | Changes both mean treatment and calibration-budget allocation |
| `public_order` | Numerical input-index order; same calibration block and mean recipe as `attaining`; discard the pilot block | A restricted-order ablation, not an oracle order |
| `nonprivate_fit` | All-$n$ empirical fractional histogram; order and optimize without training noise | Retains query privacy but has no training privacy |

For `attaining`, the pilot count-vector noise scale is $2/\bar\alpha$. The calibration histogram and mean have scales $4/(N\bar\alpha)$ and $2/(N\bar\alpha)$, respectively, splitting the calibration budget equally. The pilot and calibration blocks are disjoint. Every declared cell receives noise, including unseen states. Fixed fallback rules handle all-zero clipped histograms.

At $\alpha=0$, the five training-private methods return the data-independent constant $1/2$. `nonprivate_fit` ignores the training budget and remains nonprivate. At $\varepsilon=0$, channels reveal no input information, but their constant decoder can still be estimated subject to the training budget.

The methods have individual ideal privacy guarantees. Releasing all six fitted models together is not claimed to have a single shared training budget. Fixed public simulation seeds are not suitable hidden randomness for a private deployment.

## Seed recipe and common random numbers

The main master seed is **2026112507**. For each configuration and replicate, the implementation constructs:

```python
seed = np.random.SeedSequence(
    [2026112507, int(config_id), replicate]
)
data_seed, noise_seed = seed.spawn(2)
data_rng = np.random.default_rng(data_seed)
states = data_rng.choice(d, size=n, p=p)
labels = data_rng.binomial(1, m[states]).astype(float)
```

Replicate indices are `0` through `255` in the reference study. Each method receives the same `states` and `labels` and a newly initialized `np.random.default_rng(noise_seed)`. This is common initial fitting randomness, not a claim that methods consume identical random draws. In particular, the fixed-order ablation does not draw pilot noise. The pilot draw order is shared by `attaining` and `no_mean_calibration`.

The seeded generator, law code, and algorithm version matter together. A different NumPy version or altered draw order can change a replay. Original and migrated source hashes are mapped in [code_migration.json](../provenance/code_migration.json).

## Risk, oracles, and uncertainty

A fitted channel $Q$ has a row for every declared state, and a decoder has values $g_z\in[0,1]$. Risk is evaluated by integrating the known finite law:

$$
R(Q,g)=\sum_i p_i\left[
m_i(1-m_i)+\sum_z Q(z\mid i)(m_i-g_z)^2
\right].
$$

This evaluates the squared risk of the **randomized prediction** and includes both target and query noise. It is not the squared error of the deterministic expected prediction. No sampled test set or sampled query output adds evaluation noise. Except in the explicitly rational population calculations, “analytic evaluation” still uses float64 arithmetic.

The population oracle knows the true law and the true conditional-mean order. It is optimized separately for each query budget and has no training-privacy constraint. Saved `excess` is `risk - oracle_risk`. A smaller excess under stronger query privacy can reflect a less informative oracle; inspect absolute risk as well.

Each configuration–method mean uses its 256 independent fitting replicates. Pointwise 95% Monte Carlo intervals are

$$
\bar x\ \pm\ t_{0.975,R-1}\frac{s_x}{\sqrt R},
\qquad R=256.
$$

These quantify uncertainty in expected risk over training samples and fitting randomness. They are neither prediction intervals nor performance guarantees for a single fit. There is no multiplicity correction.

Paired comparisons use the within-replicate quantity
`comparator risk - attaining risk`. Positive values favor the pilot-and-calibration method. Computing differences before the empirical variance preserves the shared-sample coupling. The stored summaries comprise 300 configuration–method rows and 250 configuration–comparator rows.

For two near-tie signs, summaries weight signs equally. If $\bar x_-$ and $\bar x_+$ are the sign means and $v_s=s_s^2/R$, then

$$
\bar x=\frac{\bar x_-+\bar x_+}{2},\qquad
\operatorname{SE}=\frac{\sqrt{v_-+v_+}}2,\qquad
\nu=\frac{(v_-+v_+)^2}{v_-^2/(R-1)+v_+^2/(R-1)}.
$$

The interval uses the Student-$t$ critical value with $\nu$ degrees of freedom. This treats signs as fixed strata, rather than pooling their between-sign variation. Normalization is applied to each record before aggregation.

## Post-run decomposition of existing near-tie fits

After inspecting the initial learning curves, a diagnostic was added for `attaining`, `direct_histogram`, and `nonprivate_fit` on the 20 local-alternative configurations. It regenerates the original data and fitting randomness. All **15,360 replay risks and oracle values match the recorded entries exactly** in the original environment.

For each fitted channel, the diagnostic replaces its decoder only for evaluation by

$$
g_{P,Q}(z)=\frac{\sum_i p_i m_i Q(z\mid i)}
{\sum_i p_i Q(z\mid i)}.
$$

It records the decomposition

$$
R(Q,g)-V=
\underbrace{R(Q,g_{P,Q})-V}_{\text{channel excess}}
+\underbrace{\sum_z P_Q(z)(g_z-g_{P,Q}(z))^2}_{\text{decoder error}}.
$$

The population decoder is unavailable to the learner. This operation does not improve the fitted algorithm, add observations, or create independent replicates. The maximum recorded identity residual is $6.25\times10^{-17}$.

Raw and normalized components use $a=\tanh^2(\varepsilon/2)$ and normalization $\sqrt n/a$. Partition indicators distinguish the correct source partition, its opposite, other binary partitions, constant outputs, and more than two outputs. All replayed fits in the frozen study have two outputs; zero observed frequencies are not probability theorems. Approximate moment-based indicator intervals can extend outside $[0,1]$ and are not clipped.

The diagnostic explains why a declining normalized total can be driven by decoder estimation even when the channel component changes more slowly. It is explicitly a **post-run diagnostic**, not a preregistered empirical rate test.

## Exact population representation studies

These calculations use no random sampling. For every rational law on at most four states, an independent enumerator visits every canonical set partition, constructs its randomized-response channel, and computes its Bayes decoder with Python `Fraction` arithmetic. All RR partitions are included, including nonconsecutive ones. The scalar optimal-mechanism reduction justifies using this class for the unrestricted oracle; enumeration alone does not prove that reduction.

For a retained statistic $T$, both the original and coarsened channels and decoders are optimized. The recorded deficiency is $V(T)-V(H)$. Explained variance is $\Phi=\operatorname{Var}(\mathbb E[Y\mid Z])$, so risk is $\operatorname{Var}(Y)-\Phi$.

| File | Cases | Construction |
|---|---:|---|
| `geometry_three_point.csv` | 5 | $H=Y$ uniform on $\{0,1/2,1\}$, $e^\varepsilon=2$; identity, left/right optimal-bin coarsenings, boundary-crossing coarsening, constant |
| `geometry_exact_four_state.csv` | 28 | $t=2^{-j}$ for $j=3,\ldots,30$; probabilities $((1-t)/2,t/2,t/2,(1-t)/2)$, deterministic targets $(0,1/2-t,1/2+t,1)$; merge the middle states |
| `geometry_modulus_grid.csv` | 897 | Bernoulli four-state laws over seven rational privacy ratios and a reconstruction-budget grid; merge the middle states |

The three-point coarsenings have group maps `[0,1,2]`, `[0,1,1]`, `[0,0,1]`, `[1,0,1]`, and `[0,0,0]`. The first three have zero deficiency; the last two have deficiency $1/80$. The full risk is $37/240$.

The exact four-state family has actual reconstruction MSE $t^3$ and

$$
\Delta(t)=
\frac{t^2(35-34t+31t^2+4t^3-4t^4)}{36(9-t^2)}.
$$

Consequently $\Delta/(t^3)^{2/3}$ tends to $35/324$.

For the modulus grid, the exact primitive is
$r=e^\varepsilon\in\{1,51/50,21/20,11/10,5/4,2,4\}$.
Set $a=((r-1)/(r+1))^2$. The base budgets are the rationalized decimal values of 121 `np.geomspace(1e-16, 0.25, 121)` points, plus $0,1/64,1/4,1,4$. For positive $a$, also include $a^2/6400,a^2/64,a^2,\min(1/4,100a^2)$; duplicates are removed.

Each law uses probabilities $((1-w)/2,w/2,w/2,(1-w)/2)$ and means $(1/4,1/2-h,1/2+h,3/4)$. For positive budget $v\le a^2/64$, choose $w$ just below $(v/a^2)^{1/3}$ as a float-representable rational and set $h=aw$. For $a^2/64<v\le1/64$, use $w=1/4$ and rationalize $h=\sqrt{v/w}$ downward. Above $1/64$, use $w=h=1/4$; at zero budget, use $w=1/4,h=0$. Exact checks enforce $wh^2\le v$.

The comparison scale is $\Gamma_a(v)=\min\{a,a\sqrt v,(av)^{2/3}\}$. The field `budget_v` is an allowed reconstruction error; `mse` is the actual error. The family saturates at actual MSE $1/64$. Budgets above this value, including the formal plateau at $v\ge1$, are not larger actual reconstruction errors.

There are **930 law–statistic–budget cases and 1,860 full/coarsened exact-versus-float64 oracle comparisons**. The maximum explained-variance discrepancy is $2.78\times10^{-17}$. Nine exact positive deficiencies in the four-state family ($j=22,\ldots,30$) fall below the conservative float64 subtraction-resolution diagnostic. This is not floating-point underflow or evidence of exact zero. The rational values and statuses remain authoritative; plots use their float renderings. The diagnostic threshold is $64\,\epsilon_{\rm machine}\max(1,\mathbb E Y^2)$.

## Separate output-count diagnostic

A post-design interpretation check uses Bernoulli means $(0,1/2,1/2,1)$ with probabilities $(3/10,1/5,1/5,3/10)$. It enumerates all 15 four-state partitions and all 8 partitions with at most two cells at $r=1,5/4,3/2,2,3,4$.

The minimum optimal RR output counts are $1,2,2,2,2,3$. At $r=4$, the unrestricted optimum has risk $163/760$, while the best RR partition with at most two outputs has risk $127/589$, a difference of $27/23560$. This comparison prevents the main examples' two-output optima from suggesting that binary RR always suffices. The restricted comparator is specifically the at-most-two-output **RR-partition family**; this diagnostic does not separately solve every possible binary-output private mechanism.

## Runtime and allocation measurements

The recorded benchmark is separate and serial, with $n=8192$, $\alpha=1$, $\varepsilon=\log4$, the balanced graded law, $d\in\{4,8,16,32,64,128\}$, and methods `attaining` and `direct_histogram`.

Data generation is excluded. Each method–alphabet combination uses one discarded warm-up, nine timed fitting calls, and a separate allocation-traced call. The warm-up is not in the CSV. The trace measures peak Python/NumPy allocations during fitting, not resident process memory.

Benchmark seeds are:

- Data: `np.random.SeedSequence([2026112508, d])`.
- Fitting: `np.random.SeedSequence([2026112509, d, rep])`.
- Timed `rep` values: `0`–`8`; warm-up: `1000`; memory trace: `2000`.

The two methods share the generated sample and the initial fitting seed at each repetition. The 120 raw rows comprise 108 timed fits and 12 memory traces. Timing summaries report median, quartiles, minimum, and maximum; quartile ribbons are not confidence intervals.

The recorded environment is Linux x86_64, Python 3.12.14, NumPy 2.3.5, SciPy 1.17.0, and Matplotlib 3.10.8. Timings characterize the original **uncapped** implementation and this environment. They do not measure a capped-optimizer speedup, certify asymptotic complexity, or form part of a private model release.

## Outputs and provenance

Run commands from the repository root after installation. Each `--output-dir` is an output root; generated files do not overwrite the frozen references.

| Command | Work performed | Main output locations |
|---|---|---|
| `opr validate` | Hash checks, finite checks, and selected recorded-fit replays | `outputs/validate/validation.json` and check JSONs |
| `opr reproduce` | Recompute summaries, exact geometry and output counts; redraw six figure families and two tables using recorded learning/runtime/replay data | `outputs/reproduce/data/`, `figures/`, `tables/` |
| `opr simulate --replicates 256` | Generate the 50-configuration main learning run with the uncapped methods | `outputs/simulate/data/` |
| `opr summarize --data-dir outputs/simulate/data` | Analyze a completed new learning run, producing summaries and the three learning figure families | `outputs/summarize/data/`, `figures/` |
| `opr population` | Recompute deterministic exact population studies and output counts | `outputs/population/data/` |
| `opr decompose` | Replay the existing near-tie fits from a reference directory; regenerate diagnostic records and plot | `outputs/decompose/data/`, `figures/` |
| `opr benchmark` | Collect new serial uncapped timings and separate allocation traces | `outputs/benchmark/data/` |

`reproduce` performs **no new learning fits, no decomposition replays, and no new timings**. `summarize` analyzes new simulation outputs without requiring their hashes to match the frozen study, and leaves its input files unchanged. `decompose` is the explicit replay operation. The latter expects the recorded 20 near-tie configurations and 256 replicates. A smaller `simulate` run is useful for workflow checks, but it does not reproduce the reference uncertainty.

The frozen JSON metadata retain the original run descriptions and source hashes verbatim, including historical development labels. They are evidence of the original run, not instructions to locate old folders. The portable package uses [code_migration.json](../provenance/code_migration.json) to map original sources to packaged modules. Relocated imports and documentation changes mean the new modules are not generally byte-identical to the old sources.

[reference_data_manifest.json](../provenance/reference_data_manifest.json) records every frozen numerical file's SHA-256 hash, byte length, and CSV schema/count. The six final figure PDF/PNG pairs and two tables are separately provided under [results/reference](../results/reference). They use the same numerical records, with improved presentation labels. New timing values and rendered-file metadata are expected to differ across runs; numerical provenance and scientific comparisons should be assessed separately from image-file hashes.

