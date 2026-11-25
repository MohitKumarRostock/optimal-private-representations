# Methods and evaluation

This repository studies **task-specific private representations for finite-state regression**. A learner fits a randomized channel and a decoder from protected training records. At prediction time, the channel randomizes a fresh input state, and the decoder maps the released output to a prediction.

The initial feature encoder is fixed and public. The learning problem is to choose how its states are grouped, randomized, and decoded. The repository includes a reference estimator, comparison methods, population-oracle calculations, and controlled synthetic experiments. It does not train a general-purpose feature encoder or a representation intended for arbitrary downstream tasks.

All logarithms are natural. The implementation uses zero-based input indices.

## 1. Prediction problem

Let $`H=f(X)`$ be the output of a public encoder fixed independently of the fitting data. Training records are independent and identically distributed pairs

```math
D=((H_1,Y_1),\ldots,(H_n,Y_n)),\qquad
H_i\in\{0,\ldots,d-1\},\quad Y_i\in[0,1].
```

The alphabet size $`d`$, sample size $`n`$, and encoder are public. The full alphabet is declared in advance, including states absent from the sample. Input probabilities and conditional target means are unknown; there is no minimum-probability assumption on an input state.

A fitted model $`\Theta`$ specifies:

- a channel $`Q_\Theta(z\mid h)`$, a probability distribution over output indices for every input state;
- a decoder $`g_\Theta(z)\in[0,1]`$.

For a fresh input, draw $`Z\sim Q_\Theta(\cdot\mid H)`$ and predict $`g_\Theta(Z)`$. The risk of a fixed model is

```math
R_P(Q,g)=\mathbb E_{(H,Y)\sim P}
\sum_z Q(z\mid H)(Y-g(z))^2.
```

This expectation includes the channel's randomization. It is the risk of a sampled prediction, not the squared error of the deterministic mean prediction $`\sum_z Q(z\mid H)g(z)`$.

### Representation and release

An internal statistic $`T(H)`$ can restrict the information available to the channel. It need not be private by itself. The randomized output $`Z`$, or its decoded value $`g(Z)`$, is the query release. A deterministic state assignment and a randomized release are different objects.

## 2. Two privacy guarantees

The ideal algorithms use pure differential privacy: the additive failure parameter is zero.

### Privacy of fitting records

The model release has central privacy budget `alpha_train`, denoted by $`\alpha`$. Neighboring datasets have the same public size and differ by replacement of one **complete labeled record** $`(H_i,Y_i)`$. Either its input state, its target, or both can change. For every measurable model event $`E`$,

```math
\Pr\{\mathcal A(D)\in E\}
\le e^\alpha\Pr\{\mathcal A(D')\in E\}.
```

The learner operates in a trusted central environment. This is not label-only privacy, and it does not provide a privacy guarantee for training the original encoder.

### Privacy of a fresh input

For **every realized model**, its channel has all-pairs local privacy budget `epsilon_query`, denoted by $`\varepsilon`$:

```math
Q_\theta(B\mid h)\le e^\varepsilon Q_\theta(B\mid h')
\quad\text{for every }h,h'\text{ and output event }B.
```

The requirement covers the entire declared alphabet, including unseen states. It is conditional on the released model and applies to each individual randomized query release. Repeated releases for the same confidential input require composition; the two fitting/query budgets protect different objects.

### Which API calls implement a query release?

The public estimator is `fit_finite_private_task_learner`, returning a `FinitePrivateTaskModel`.

| Model method | Meaning | Query-release status |
|---|---|---|
| `sample_representations(states, rng=...)` | Draw one randomized output index per supplied state | Implements the randomized release |
| `predict(states, rng=...)` | Draw an output index and decode it | Implements the randomized release |
| `decode(private_indices)` | Decode indices that have already been released privately | Postprocessing |
| `bins(states)` | Compute deterministic input-to-output assignments | Internal/evaluation quantity |
| `probabilities(states)` | Return the corresponding channel rows | Internal/evaluation quantity |
| `expected_prediction(states)` | Compute the mean of the randomized prediction | Internal/evaluation quantity |

The last three calls are not private ways to release a confidential query state. A complete fitted channel table can be part of a training-private model: that release concerns the fitting records, rather than an input subsequently used to select one of its rows.

## 3. Population oracles and information loss

The population oracle knows the joint law $`P`$ and minimizes squared risk over every feasible query-private channel and decoder:

```math
V_\varepsilon(P;H)=\inf_{Q,g:\,Q\text{ is }\varepsilon\text{-private}}
R_P(Q,g).
```

It is unrestricted by training privacy or by a learned score order. It is a reference value, not a fitted baseline available to an ordinary learner.

For bounded scalar squared loss, the conditional mean $`m(h)=\mathbb E[Y\mid H=h]`$ contains the task-relevant information for this oracle. Optimal scalar privatization can be represented by randomized response on ordered bins of that mean. The scalar randomized-response structure is established in [Ghazi et al. (2023)](https://arxiv.org/abs/2212.06074); [Kairouz, Oh and Viswanath (2016)](https://www.jmlr.org/papers/volume17/15-135/15-135.pdf) provide broader foundations for optimal local-private mechanism design.

### Computing an oracle on a finite law

Write $`p_i=P(H=i)`$, $`m_i=\mathbb E[Y\mid H=i]`$, and $`q_i=p_i m_i`$. Supplying these population moments and the **true conditional-mean order** to the ordered optimizer gives the finite-law oracle in ideal arithmetic. Supplying a learned or prescribed order instead solves that order-constrained problem.

Two independent implementations support the experiments:

- [experiments/models.py](../src/optimal_private_representations/experiments/models.py) evaluates known-law oracles with a floating-point ordered optimizer.
- [experiments/population.py](../src/optimal_private_representations/experiments/population.py) enumerates every canonical set partition for laws with at most four positive-probability states, including nonconsecutive partitions, using rational arithmetic. It constructs each channel and Bayes decoder directly. Rational calculations require rational probabilities, means, second moments, and $`r=e^\varepsilon`$; $`\varepsilon`$ itself need not be rational.

The mathematical reduction justifies the oracle class. Finite enumeration and numerical comparisons check the implemented instances; enumeration alone does not prove the structural reduction.

### Deficiency of a retained statistic

The task-specific information loss of $`T`$ is

```math
\Delta_\varepsilon(T;P)
=V_\varepsilon(P;T)-V_\varepsilon(P;H)\ge0.
```

Both channels and both decoders are optimized separately. This is an intrinsic information loss, not the loss from routing through a fixed original codebook. Comparing optimal risks before and after a representation is a decision-theoretic feature-gap construction; see [van Rooyen and Williamson (2015)](https://arxiv.org/abs/1504.00083).

Preserving the bin assignment of **some** optimal private codebook is sufficient for zero deficiency. The same channel and decoder can then operate through the retained statistic, attaining the full-input oracle risk; composing any private channel with the statistic cannot improve on that oracle. This argument does not require reconstructing the entire conditional mean or choosing a unique optimal partition.

For example, take $`H=Y`$ uniform on $`\{0,1/2,1\}`$ and $`\varepsilon=\log 2`$. The full oracle risk is $`37/240`$.

| Retained statistic | Conditional-mean reconstruction MSE | Reoptimized private risk | Deficiency |
|---|---:|---:|---:|
| Isolate $`0`$, merge $`\{1/2,1\}`$ | $`1/24`$ | $`37/240`$ | $`0`$ |
| Merge $`\{0,1\}`$, isolate $`1/2`$ | $`1/6`$ | $`1/6`$ | $`1/80`$ |

These statistics have the same two category probabilities, but preserve different distinctions.

At $`\varepsilon=0`$, the oracle predicts the overall mean, its risk is $`\mathrm{Var}(Y)`$, and every statistic has zero deficiency. Learning that mean from protected fitting records can still incur estimation error.

Let $`M=\mathbb E[Y\mid H]`$, $`M_T=\mathbb E[M\mid T]`$, and $`v_T=\mathbb E(M-M_T)^2`$. The population experiments also use the reference scale

```math
\Gamma_a(v)=\min\{a,\ a\sqrt v,\ (av)^{2/3}\},
\qquad a=\tanh^2(\varepsilon/2).
```

For positive query budgets up to $`\log4`$, the recorded study compares exact deficiencies of explicit four-state coarsenings with $`\Gamma_a(v)`$. These ratios are diagnostics of the specified families, not a numerical proof of a uniform approximation law. The argument $`v`$ is an allowed error budget, not necessarily the actual MSE of an example; every actual $`v_T`$ is at most $`1/4`$. A positive reconstruction error need not imply a positive deficiency, as the table demonstrates.

One small-error family takes $`H=Y`$ in $`\{0,1/2-u,1/2+u,1\}`$ with probabilities $`((1-u)/2,u/2,u/2,(1-u)/2)`$ and merges the middle pair. Its reconstruction error is $`u^3`$. The rational oracle routine evaluates both sides of this coarsening directly, including gaps too small for a reliable float64 subtraction.

The exponent and geometric ingredients have established precedents in [margin-based classification](https://arxiv.org/pdf/0708.2321), [optimal quantization](https://www.numdam.org/item/JSFS_2018__159_1_1_0/), and [privacy contraction](https://arxiv.org/abs/2210.13386). These ingredients do not by themselves identify the complete optimized risk gap.

## 4. Private pilot and calibration estimator

The public estimator in [learner.py](../src/optimal_private_representations/learner.py) separates estimating an input order from calibrating a private channel for squared risk.

Let

```math
\bar\alpha=\min\{1,\alpha\},\qquad
\kappa=\tanh(\alpha/2).
```

The algorithm returns the data-independent prediction $`1/2`$ when $`n=1`$ or $`n\kappa<1`$, including $`\alpha=0`$. Otherwise, fixed public indices split the sample into a pilot of size $`n_1=\lfloor n/2\rfloor`$ and a calibration block of size $`N=\lceil n/2\rceil`$. The implementation uses $`\bar\alpha`$, so additional training budget above one is not spent.

### Pilot: estimate the input order

For every declared state, form fractional counts

```math
A_i=\sum_{\ell\le n_1}Y_\ell\,\mathbf1\{H_\ell=i\},
\qquad
B_i=\sum_{\ell\le n_1}(1-Y_\ell)\,\mathbf1\{H_\ell=i\}.
```

Add independent Laplace noise of scale $`2/\bar\alpha`$ to all $`2d`$ coordinates and clip negative values to zero. From the sanitized counts, compute

```math
\widetilde s_i=
\frac{\widetilde A_i+1/2}{\widetilde A_i+\widetilde B_i+1}.
```

Every state receives noise, whether or not it was observed. The complete count vector has replacement $`\ell_1`$ sensitivity two.

### Calibration: estimate cell moments and the global mean

On the calibration block, form normalized fractional counts $`\widehat z_{i0},\widehat z_{i1}`$ and the target mean $`\widehat\mu`$.

| Released statistic | Replacement sensitivity | Laplace scale | Allocated budget |
|---|---:|---:|---:|
| Entire normalized fractional histogram | $`2/N`$ | $`4/(N\bar\alpha)`$ | $`\bar\alpha/2`$ |
| Overall target mean | $`1/N`$ | $`2/(N\bar\alpha)`$ | $`\bar\alpha/2`$ |

Clip the noisy mean to $`[0,1]`$. Clip the histogram columns to be nonnegative, then rescale their total masses to $`1-\widetilde\mu`$ and $`\widetilde\mu`$. If a clipped column is entirely zero, place its required mass on public state zero. This produces a valid sanitized joint law.

Fractional counts support every real target in $`[0,1]`$. For squared loss, the first moments determine the part of risk affected by a predictor; the target second moment is an additive constant. Treating the sanitized moments as a binary-target law therefore supplies the correct optimization problem without assuming binary training targets.

Estimating the overall mean separately prevents its noise from accumulating across $`d`$ histogram cells. This component is evaluated explicitly by an ablation.

### Optimize and release

Order states by their sanitized pilot scores and optimize ordered randomized-response bins and their Bayes decoder under the sanitized calibration law. Return the fitted model and sanitized quantities, not raw counts, labels, observed support, or the random generator.

The pilot and calibration blocks are disjoint with fixed membership. A replacement affects one block: the pilot uses budget $`\bar\alpha`$, and the calibration's two releases compose to $`\bar\alpha`$. Consequently, the complete ideal model release uses at most $`\bar\alpha\le\alpha`$, rather than summing the budgets across the disjoint blocks. All later optimization is postprocessing.

## 5. Ordered randomized-response optimization

The low-level function is

~~~python
fit_ordered_private_dp(p, s, scores, epsilon, max_outputs=None)
~~~

Here `p[i]` is a cell probability, `s[i]` is its first target moment $`q_i`$, and `scores[i]` fixes the state ordering. In this function name, the suffix refers to **dynamic programming**. The optimizer is not a training-private release by itself when supplied with unsanitized empirical moments.

For a deterministic assignment $`b(h)`$ into $`k`$ bins and $`c=e^\varepsilon-1`$, randomized response uses

```math
Q(z\mid h)=\frac{1+c\,\mathbf1\{z=b(h)\}}{k+c}.
```

Every row is a distribution and each column's largest-to-smallest probability ratio is at most $`e^\varepsilon`$. This establishes all-pairs privacy for every defined input state.

Let $`\mu=\sum_i q_i`$. For a consecutive interval $`I`$, write $`p_I=\sum_{i\in I}p_i`$ and $`q_I=\sum_{i\in I}q_i`$. Its Bayes decoder and objective contribution are

```math
g(I)=\frac{\mu+cq_I}{1+cp_I},
\qquad
G(I)=\frac{(\mu+cq_I)^2}{1+cp_I}.
```

A partition's risk is the target second moment minus $`\sum_I G(I)/(k+c)`$. The dynamic program maximizes this objective while requiring nondecreasing decoder values across successive intervals. It considers all feasible last intervals and predecessor intervals, then backtracks the best partition.

Equal scores are aggregated. Zero-sanitized-mass groups need not participate in optimization, but thresholds extend the fitted assignment to **all** declared states. At a threshold, `searchsorted(..., side="right")` assigns the score to the interval on its right. At $`\varepsilon=0`$, one output with the sanitized mean suffices.

### Output cap and computation

In ideal arithmetic, an optimal ordered solution needs no more than

```math
K_\varepsilon=
\max\left\{2,\left\lceil\frac{e^{2\varepsilon}+3}{2}\right\rceil-1\right\}
```

outputs for $`\varepsilon>0`$; take $`K_0=1`$. This is a sufficient cap, not a claim that every optimum has this many outputs. For $`0<\varepsilon\le\log4`$, it is at most nine.

The public `output_count_cap` uses a conservative floor form with a floating-point guard. It may allow one extra output at an integer breakpoint of the symbolic expression; it still allows at most nine in the supported query-budget range. The default optimizer applies this cap. An explicit `max_outputs` instead defines the requested comparison class; a smaller user-supplied cap can lose optimality. Passing `max_outputs=len(p)` requests the uncapped search.

**Why the cap is sufficient.** Allow real nondecreasing decoder values and choose an ordered optimum with the fewest outputs and risk $`R`$. A constant prediction is feasible, so $`R\le\mathrm{Var}(Y)`$. Deleting a bin with zero input mass, or merging adjacent bins with the same decoder value $`b`$, gives

```math
R'=\frac{(k+c)R-\mathbb E(Y-b)^2}{k+c-1}\le R.
```

Thus a minimum-cardinality optimum has nonempty bins and strictly increasing decoder values. Existence follows by clipping decoders to $`[0,1]`$ and considering the finitely many nonempty ordered partitions. At a strict ordered optimum the order constraints are locally inactive, so each decoder is its bin's unconstrained Bayes value $`g(I)`$.

For its bin moments, put $`u_j=q_j-\mu p_j`$. The identity

```math
g_j-\mu=\frac{cu_j}{1+cp_j}
```

shows that negative $`u_j`$ form a prefix and positive $`u_j`$ form a suffix. If all are zero, a constant output is optimal. Otherwise let $`U=\sum_{u_j>0}u_j=-\sum_{u_j<0}u_j>0`$. The original channel's explained variance satisfies

```math
\Phi_k=\frac{c^2}{k+c}\sum_j\frac{u_j^2}{1+cp_j}
\le\frac{2c^2U^2}{k+c}.
```

Merge the negative and positive groups into two consecutive bins, attaching a possible zero group to either side. With merged masses $`p_-+p_+=1`$, this feasible ordered channel has explained variance

```math
\Phi_2=
\frac{c^2U^2}{(1+cp_-)(1+cp_+)}
\ge\frac{4c^2U^2}{(c+2)^2}.
```

For $`k>2`$ and $`k\ge(e^{2\varepsilon}+3)/2`$, this binary merge has no greater risk and fewer outputs, contradicting minimum cardinality. The displayed cap follows. This argument concerns the ideal optimization; it is separate from certifying its floating-point implementation.

If $`m\le d`$ distinct score groups have positive sanitized mass and $`B`$ is the search limit, the recurrence requires $`O(Bm^2\log(m+1))`$ arithmetic operations and $`O(Bm^2)`$ working storage. Grouping the full alphabet and constructing the full channel table add their corresponding sorting and output costs. The full estimator has bounds $`O(n+K_\varepsilon d^2\log(d+1))`$ and $`O(K_\varepsilon d^2)`$. In the supported query-budget range these become $`O(n+d^2\log(d+1))`$ time and $`O(d^2)`$ storage.

### Capped API versus recorded experiments

The top-level package exports the capped estimator and optimizer. The [legacy](../src/optimal_private_representations/legacy) package preserves the uncapped implementation used to generate the recorded studies, and the experiment modules import it explicitly. The cap preserves the ideal optimal objective; it need not preserve a unique minimizing partition.

Recorded runtime and memory results describe the **uncapped** implementation. They are not measured speedups or timings for the capped implementation. Replaying the recorded studies preserves their algorithm choice and random-number sequence.

## 6. Learning excess and reference scale

Learning is evaluated by expected oracle excess,

```math
\mathbb E_{D,\mathcal A}R_P(Q_{\mathcal A(D)},g_{\mathcal A(D)})
-V_\varepsilon(P;H).
```

The outer expectation averages training samples and fitting randomness; the risk already averages the fresh target and channel randomization.

The corresponding minimax question takes the infimum over learners satisfying both privacy requirements and the supremum over all joint laws on the declared input alphabet and $`[0,1]`$ targets.

For the recorded regimes $`d\ge4`$, $`n\ge1`$, finite $`\alpha\ge0`$, and $`0\le\varepsilon\le\log4`$, the experiments calculate the parameter-dependent reference expression

```math
S_{n,d}(\alpha,\varepsilon)=
\frac1n+\frac a{\sqrt n}+\Gamma_a(d/n)
+\min\{1,t^{-2}\}+a\min\{1,d/t\},
\qquad t=n\tanh(\alpha/2).
```

At $`t=0`$, the final two terms are defined as $`1`$ and $`a`$. The expression provides a common scale for examining sample size, alphabet size, and the two privacy budgets. Establishing matching minimax upper and lower bounds against arbitrary private channels is a separate theorem-and-proof claim; this documentation does not present the finite experiments as such a proof or attribute that complete claim to the references below.

The terms track overall-mean estimation, selection between nearly tied partitions, learning uncertain input assignments, private mean estimation, and private input-dependent estimation. They are not an exact additive decomposition of the risk on any one distribution. Established tools for analyzing such questions include the sharp empirical-distribution inequality of [Massart (1990)](https://doi.org/10.1214/aop/1176990746) and the private-testing methods of [Acharya, Sun and Zhang (2021)](https://proceedings.mlr.press/v132/acharya21a.html) and [Lalanne, Garivier and Gribonval (2023)](https://arxiv.org/abs/2210.02215).

For fixed positive privacy budgets and fixed $`d\ge4`$, the expression $`S_{n,d}`$ itself has order $`n^{-1/2}`$ as $`n`$ grows. This algebraic property does not determine an observed learning exponent on a particular law. A stronger query-privacy constraint can also reduce oracle excess by making the oracle less informative; this does not imply lower absolute prediction risk.

For a learner operating through a fixed public statistic $`T(H)`$, adding and subtracting $`V_\varepsilon(P;T)`$ gives the exact accounting identity

```math
\mathbb E R_P(\widehat Q,\widehat g)-V_\varepsilon(P;H)
=\Delta_\varepsilon(T;P)
+\left[\mathbb E R_P(\widehat Q,\widehat g)-V_\varepsilon(P;T)\right].
```

It separates population information loss from fitting excess on the retained alphabet without assuming a particular learning rate. Choosing the statistic adaptively from the same protected records requires additional privacy and statistical analysis.

## 7. Comparison methods

The recorded study uses the following method identifiers in [experiments/models.py](../src/optimal_private_representations/experiments/models.py).

| Identifier | Method | Interpretation |
|---|---|---|
| `attaining` | Private pilot and separate-mean calibration | Identifier for the reference estimator in the recorded files |
| `direct_histogram` | Privatize the full fractional histogram using all records; clip, normalize, order by sanitized means, and optimize | Uses the same individual training/query budgets without a sample split |
| `private_mean` | Add Laplace noise to the full-sample mean and clip; use a constant query channel | Input-independent private baseline |
| `no_mean_calibration` | Keep the private pilot; spend the full calibration budget on its histogram, with no separate mean release | Changes both mean treatment and calibration-budget allocation |
| `public_order` | Use numerical input-index order and the original calibration recipe; discard the pilot block | Tests a restricted public order, not knowledge of the true mean order |
| `nonprivate_fit` | Optimize the full empirical fractional-label law without training noise | Omits training privacy but retains query privacy |

The direct histogram has normalized replacement sensitivity $`2/n`$ and Laplace scale $`2/(n\bar\alpha)`$. The constant-mean baseline has sensitivity $`1/n`$ and scale $`1/(n\bar\alpha)`$. The no-separate-mean ablation uses scale $`2/(N\bar\alpha)`$ for the calibration histogram. All private methods use the data-independent constant $`1/2`$ at zero training budget.

The public-order ablation can have a persistent approximation gap: a useful grouping may not be expressible by consecutive intervals in that order. It therefore does not isolate the statistical cost of estimating the true order.

The known-law population oracle is separate from these fitted methods. Each private fitted method has its own ideal privacy accounting; jointly releasing all compared fits is not claimed to satisfy one common training budget.

## 8. Evaluation and uncertainty

For a known law, the evaluator integrates target and query randomness analytically. Given $`m_i=\mathbb E[Y\mid H=i]`$ and conditional second moments $`s_{2,i}=\mathbb E[Y^2\mid H=i]`$,

```math
R_P(Q,g)=
\sum_i p_i\left[
s_{2,i}-m_i^2+
\sum_zQ(z\mid i)(m_i-g_z)^2
\right].
```

For Bernoulli targets, $`s_{2,i}=m_i`$. This integration evaluates the randomized predictor's squared risk without drawing test-query samples. Floating-point integration is not exact rational arithmetic.

The recorded learning study has 50 configurations, 256 independent fitting samples per configuration, and six methods: 76,800 fits in total. The main distribution families are:

- **Balanced graded:** uniform input probabilities and means evenly spaced from 0.05 to 0.95, reordered by bit reversal of the input index. Separate sweeps vary sample size, alphabet size, and the two privacy budgets.
- **Uneven low-mean:** 32 states with probabilities proportional to $`i^{-1.3}`$, $`i=1,\ldots,32`$, and bit-permuted means from 0.02 to 0.38.
- **Local alternatives near a partition tie:** probabilities $`(1/3,1/6,1/6,1/3)`$ and means $`(1/4,1/2+\theta,1/2+\theta,3/4)`$, with $`\theta=\pm c/\sqrt n`$, $`c\in\{0.5,1\}`$, $`\varepsilon=\log2`$, and $`\alpha=1`$.

The full declared configuration grid and recorded results are in [data/reference](../data/reference). Different configurations use independent samples, even when their parameter values coincide. Within a replicate, methods share the fitting sample and the initial synthetic random-number seed for paired comparisons.

Pointwise intervals use empirical standard errors and Student-$`t`$ critical values. They quantify Monte Carlo uncertainty in **mean risk over fitting**, not a prediction interval or a guarantee for one trained model. Paired intervals use the within-replicate difference between a comparator and pilot/calibration. The two signs of a local alternative are equally weighted, using within-sign variances and Welch–Satterthwaite degrees of freedom. The intervals are not adjusted for multiple comparisons.

The local-alternative laws vary with $`n`$. A post-run diagnostic reuses fitted channels and replaces only their decoder by the population Bayes decoder to separate channel-selection error from decoder-estimation error. That population redecoding is unavailable to the fitted learner and does not add a training sample.

The study supports distribution-dependent rankings: the direct histogram is strong on the tested balanced family, separate mean calibration helps in some uneven-cell settings, and a private constant predictor can be preferable with little data. The curves do not identify a minimax supremum or an asymptotic learning exponent.

## 9. Numerical and scientific scope

- **Mathematical privacy versus implementation.** The privacy arguments assume ideal Laplace noise and arithmetic. NumPy floating-point calculations and random-number generation are research references, not certified finite-precision pure-DP implementations. Finite numerical tests cannot establish privacy over every neighboring dataset.
- **Random seeds.** Public fixed seeds support synthetic reproducibility. They must not be treated as hidden randomness for a private release on confidential records.
- **Diagnostic outputs.** Raw sufficient statistics, selected-query probability rows, elapsed fitting time, and internal memory traces are not covered as private releases merely because they accompany a private algorithm. Experiment outputs here concern synthetic data.
- **Small differences and ties.** Positive oracle gaps can fall below float64 resolution. The small-law reference calculations use exact fractions; floating-point optimizers can choose among numerical ties. An output-count cap is a computational bound, not a guarantee of a unique partition.
- **Range of the public estimator.** `epsilon_query` is restricted to $`[0,\log4]`$; `alpha_train` is finite and nonnegative. The fitting API accepts any positive declared alphabet size; the recorded learning study uses $`d\ge4`$.
- **One fixed task.** The target is scalar and bounded, the loss is squared error, and the encoder is fixed independently of fitting data. Multitask transfer, learned encoders, adaptive selection of a retained statistic, and an end-to-end deployment evaluation are outside the implemented study.

## References

The following primary sources provide established methods and theoretical ingredients used in this documentation. Their privacy models and objectives should be read with their own stated assumptions. BibTeX entries are available in [references.bib](references.bib).

1. Ghazi, B., Kamath, P., Kumar, R., Leeman, E., Manurangsi, P., Varadarajan, A. V., and Zhang, C. (2023). **Regression with Label Differential Privacy.** ICLR. [Paper](https://arxiv.org/abs/2212.06074).
2. Kairouz, P., Oh, S., and Viswanath, P. (2016). **Extremal Mechanisms for Local Differential Privacy.** JMLR 17(17), 1–51. [Paper](https://www.jmlr.org/papers/volume17/15-135/15-135.pdf).
3. van Rooyen, B., and Williamson, R. C. (2015). **A Theory of Feature Learning.** arXiv:1504.00083. [Paper](https://arxiv.org/abs/1504.00083).
4. Asoodeh, S., and Zhang, H. (2024). **Contraction of Locally Differentially Private Mechanisms.** IEEE Journal on Selected Areas in Information Theory 5, 385–395. [Paper](https://arxiv.org/abs/2210.13386).
5. Audibert, J.-Y., and Tsybakov, A. B. (2007). **Fast Learning Rates for Plug-in Classifiers.** Annals of Statistics 35(2), 608–633. [Paper](https://arxiv.org/pdf/0708.2321).
6. Levrard, C. (2018). **Quantization/Clustering: When and Why Does $`k`$-Means Work?** Journal de la Société Française de Statistique 159(1), 1–26. [Paper](https://www.numdam.org/item/JSFS_2018__159_1_1_0/).
7. Massart, P. (1990). **The Tight Constant in the Dvoretzky–Kiefer–Wolfowitz Inequality.** Annals of Probability 18(3), 1269–1283. [Paper](https://doi.org/10.1214/aop/1176990746).
8. Acharya, J., Sun, Z., and Zhang, H. (2021). **Differentially Private Assouad, Fano, and Le Cam.** Algorithmic Learning Theory, PMLR 132, 48–78. [Paper](https://proceedings.mlr.press/v132/acharya21a.html).
9. Lalanne, C., Garivier, A., and Gribonval, R. (2023). **On the Statistical Complexity of Estimation and Testing under Privacy Constraints.** TMLR. [Paper](https://arxiv.org/abs/2210.02215).
