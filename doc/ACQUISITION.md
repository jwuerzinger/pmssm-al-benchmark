# Acquisition: how a batch is chosen

Every pipeline chooses the points to label in the same three stages. The
code is `pmssm/selection.py` (candidates and strategies), `pmssm/uncertainty.py`
(the uncertainty of each surrogate) and `pmssm/heads.py` (what the surrogate's
output means).

## Stage 1: uncertainty

| surrogate | uncertainty | how |
|---|---|---|
| transformer, DNN | MC dropout | `--mc-samples` stochastic forward passes (default 30); the variance across passes |
| exact GP, deep GP | posterior variance | one pass through the GP likelihood gives mean and variance |
| TabPFN | in-context ensemble | the variance of `--n-ensemble-samples` forward passes of the pre-trained model |
| transformer, DNN with `--acq-uncertainty laplace` | last-layer Laplace posterior | linearised Gaussian posterior over the last layer, `--laplace-prior-precision` |
| transformer, DNN with `--n-ensemble K` | ensemble disagreement | K independently trained members per iteration |

## Stage 2: the shared pre-filter

1. **Candidates.** A pool of `--n-candidates` points (one million in
   production, 20,000 in the driver default) is drawn by Latin hypercube
   sampling or uniformly (`--candidate-generation`) over the free parameters.
   `--candidate-source mcmc` draws them from the reference posterior instead,
   the oracle configuration of the paper.
2. **Tolerance cut** (`--tolerance-sampling`, default 1.0). A hard cut that
   keeps candidates whose predicted target lies within ±tol of the boundary in
   transformed space. With t = log(Y / Y₀) and tol = 1 this is a multiplicative
   window of e around Y₀: candidates the surrogate predicts far from the
   boundary are dropped. 0 disables it.
3. **Proximity weighting** (`--proximity-sampling`, default 0.1). A soft
   Gaussian weight on the surviving variances,
   `exp(-(mean - threshold)² / sigma) * variance`, which favours the boundary
   without eliminating anything. 0 disables it.
4. **Focused pool** (`entropy_batch` only, `--entropy-pool-size`). The top
   candidates by weighted variance are kept for the covariance computation
   (5,000 in production).

The GP pipelines evaluate the full million candidates because one pass is
cheap; the dropout surrogates pay `--mc-samples` passes per candidate, which is
why their focused pool is small. `doc/IMPLEMENTATION_DETAILS.md` has the cost
comparison.

## Stage 3: the selection strategy (`--selection-strategy`)

| strategy | tolerance cut | proximity weighting | batch construction |
|---|---|---|---|
| `top_k_tol_only` | yes | no | the `--n-select` largest raw variances |
| `top_k` | yes | yes | the `--n-select` largest weighted variances |
| `entropy_batch` | yes | yes | greedy, diverse batch from the predictive covariance (below) |
| `tol_only_random` | yes | no | a uniform draw among the survivors of the cut; never consults the uncertainty |
| `cls_entropy` | yes | yes | classification head: the entropy of the mean verdict probability |
| `bald` | yes | yes | classification head: mutual information, the epistemic part of that entropy |

`entropy_batch` is the production choice for the transformer, the DNN and
both GPs; TabPFN defaults to `top_k` because the covariance of its in-context
ensemble over the focused pool is prohibitively expensive (pass
`--selection-strategy entropy_batch` explicitly to pay for it).
`tol_only_random` is the control arm: it isolates the value of the tolerance
cut from the value of the uncertainty ranking.

**`entropy_batch`.** Over the focused pool the surrogate's samples give a
covariance matrix C (dropout or ensemble samples for the neural surrogates,
the posterior covariance for the GPs). The first point is the candidate of
largest smoothed entropy; every further point is chosen by its conditional
batch entropy given the points already in the batch, `log|Σ + I| - log|Σ + 2I|
+ n log 2`, with a Gibbs draw at temperature `--entropy-beta` (default 50,
quasi-deterministic) and a blur `--entropy-blur` (default 0.15) for numerical
stability. Once a point is in the batch, candidates correlated with it lose
conditional entropy, which spreads the batch over the uncertainty landscape.

## Heads: what the output means (`--head`)

| head | trains on | loss | acquisition score |
|---|---|---|---|
| `regression` (default) | t = log(Y / Y₀) | MSE | predictive variance, with the cut and weighting above |
| `classification` | 1[t > 0] | binary cross-entropy | `cls_entropy` or `bald` from the MC samples of the verdict probability |
| `lsq_classification` | ±1 verdict labels | Gaussian likelihood (least squares) | probit read-out of the Gaussian predictive; the GP classifier of Rasmussen and Williams, section 6.5 |

The GP pipeline additionally offers `--model-type laplace_gpc`, a Laplace
approximation to the GP classifier. Every head keeps one scalar output and
the same decision point (output > 0), so the per-iteration verdict accuracy
is head-agnostic; metrics that read the output as a value of t (MSE, R²,
NLPD) are reported as missing for classification heads rather than as wrong
numbers. `pmssm.heads.head_for_run` recovers a run's head from its
`summary.json`, defaulting to `regression` for runs made before heads existed.
Adding a head means one class in `pmssm/heads.py`; `select_points` already
ranks by whatever score a head produces.

## Defaults worth knowing

| quantity | driver default | production (`slurm/al`) |
|---|---|---|
| initial labelled set | the pool slice selected by `--n-samples` | 2,000 |
| candidates per iteration | 20,000 | 1,000,000 |
| points selected per iteration | 10 | 500 |
| iterations | 1 | 40 (200 in the campaign) |
| focused pool (`entropy_batch`) | 1,500 | 5,000 |
| MC dropout passes | 30 | 30 |
| tolerance, proximity | 1.0, 0.1 | 1.0, 0.1 |
| static evaluation set | 100,000 models reserved from the pool | 100,000 |
