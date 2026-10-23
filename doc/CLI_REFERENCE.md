# CLI reference

The four pipelines share most of their options; the complete lists below are
the output of `--help` and are the authoritative reference. Conventions:

- **Config files and sweeps.** `--config-file sweep.yaml` overrides any option
  (keys with underscores: `n_iterations: 10`); a list value creates a
  Cartesian product, and `--sweep-index N` selects one combination, which is
  what `slurm/al/submit_sweep.sh` passes from the array task id.
- **Resuming.** `--resume-from <run dir> --n-additional-iterations <n>`
  continues a run in place from its `state.pt`; `--allow-config-drift`
  permits a changed configuration. Architecture flags must be repeated.
- **Targets and heads.** `--target DMRD|ExpR`, `--head
  regression|classification|lsq_classification`, `--selection-strategy`
  (`doc/ACQUISITION.md`, `doc/TARGETS.md`). `--no-mcmc-eval` is required for
  targets without a reference posterior.
- **Testing.** `--testing` loads a small slice of the pool; `--no-generate-data`
  selects without labelling.
- **Production values** are set by the per-model scripts in `slurm/al/`
  (`doc/SLURM.md`), not by the driver defaults.

## `pipelines/active_learning_transformer.py`

```
Usage: active_learning_transformer.py [OPTIONS]

  Active learning pipeline for pMSSM relic density prediction.

  Trains PMSSMTransformerTabular, computes uncertainty via MC Dropout, and
  selects most informative points for data generation.

Options:
  --testing / --no-testing        Run in testing mode (small data).
  --n-iterations INTEGER          Number of active learning iterations.
  --n-candidates INTEGER          Candidate pool size.
  --n-select INTEGER              Number of points to select per iteration.
  --mc-samples INTEGER            Number of MC dropout forward passes.
  --epochs INTEGER                Training epochs per iteration (default:
                                  2000).
  --dropout FLOAT                 Dropout rate for MC dropout.
  --n-datasets INTEGER            Number of ROOT datasets to load.
  --n-samples INTEGER             Number of samples to use from data.
  --val-fraction FLOAT            Fraction of data reserved for validation
                                  (default: 0.2). Applied to initial data and
                                  each batch of new points.
  --output-dir TEXT               Output directory.
  --generate-data / --no-generate-data
                                  Generate new models using Run3ModelGen.
  --min-gen-fraction FLOAT        Minimum fraction of n-select that must be
                                  generated successfully before stopping
                                  retries (default: 0.6).
  --max-gen-attempts INTEGER      Maximum number of generation attempts per
                                  iteration (default: 10).
  --gen-workers INTEGER           Number of parallel genModels.py workers per
                                  generation attempt (default: 1).
  --head [classification|lsq_classification|regression]
                                  Acquisition head (see pmssm.heads).
                                  'regression' (default) is the production
                                  path: MSE on the transformed target,
                                  acquisition by predictive variance.
                                  'classification' trains binary cross-entropy
                                  on the verdict 1[t>0] and must be paired
                                  with --selection-strategy bald or
                                  cls_entropy, since its output is a logit and
                                  the mean-based tolerance cut and proximity
                                  weighting have no meaning in logit units.
  --selection-strategy [top_k|top_k_tol_only|tol_only_random|entropy_batch|bald|cls_entropy]
                                  Selection strategy: top_k, top_k_tol_only
                                  (short-circuit, no proximity weighting), or
                                  entropy_batch (default).
  --entropy-blur FLOAT            Entropy smoothing parameter (entropy_batch
                                  only).
  --entropy-beta FLOAT            Gibbs sampling temperature (entropy_batch
                                  only).
  --entropy-pool-size INTEGER     Focused pool size for entropy_batch pre-
                                  filtering.
  --candidate-generation [uniform|lhs]
                                  Candidate pool generation method: uniform
                                  random or Latin Hypercube Sampling (default:
                                  lhs).
  --candidate-source [pool|mcmc]  Candidate source: 'pool' (LHS/uniform,
                                  default) or 'mcmc' (theoretical-limit /
                                  oracle mode — restrict candidates to the
                                  pre-loaded MCMC dataset; --generate-data is
                                  forced off).
  --proximity-sampling FLOAT      Gaussian proximity weighting width around
                                  target value (0 to disable, default: 0.1).
  --tolerance-sampling FLOAT      Hard cut: keep only candidates within
                                  ±tolerance of threshold in transformed space
                                  (0 to disable, default: 1.0).
  --target [CLs|CrossSection|DMRD|ExpR]
                                  Observable the surrogate learns and the AL
                                  loop targets. 'DMRD' is the dark matter
                                  relic density (boundary Omega h^2 = 0.12);
                                  'ExpR' is the SModelS best expected r-value,
                                  whose boundary r = 1 is the LHC exclusion
                                  contour. Selects the ROOT branch, the ingest
                                  validity cuts and the Run3ModelGen steps
                                  used to label generated points (default:
                                  DMRD).
  --target-value FLOAT            Physical target value for proximity
                                  weighting. Defaults to the selected target's
                                  registry value (0.12 for DMRD, 1.0 for
                                  ExpR), so it normally needs no explicit
                                  setting.
  --no-mcmc-eval                  Ignore --mcmc-data-dir and run without an
                                  MCMC reference set. Needed for targets that
                                  have no posterior reference: the per-model
                                  submit scripts always pass --mcmc-data-dir
                                  (two of them hardcode it), and an empty
                                  value cannot override a shell ':-' default,
                                  so this flag is the only way to disable it
                                  purely through AL_EXTRA_ARGS.
  --config-file TEXT              YAML config file (overrides CLI args).
                                  Supports parameter sweeps.
  --sweep-index INTEGER           Sweep combination index (requires --config-
                                  file).
  --early-stopping / --no-early-stopping
                                  Enable early stopping on validation loss
                                  (default: enabled).
  --patience INTEGER              Early stopping patience (epochs without
                                  improvement, default: 200).
  --warm-starting / --no-warm-starting
                                  Warm-start from previous iteration
                                  checkpoint (default: enabled). NOTE: because
                                  this defaults to enabled, a cold run must be
                                  resumed with an explicit --no-warm-starting
                                  or it will silently continue warm; --resume-
                                  from now refuses to do so.
  --n-ensemble INTEGER            Train K independent members per iteration
                                  and use their disagreement as the
                                  acquisition uncertainty instead of MC
                                  dropout (K=1, the default, keeps MC
                                  dropout). Members get distinct seeds and
                                  fresh initialisations, so their spread
                                  measures how much functions that all fit the
                                  labelled data still differ. Requires --no-
                                  train-baseline. NOTE the across-member
                                  covariance has rank <= K-1, so entropy_batch
                                  is degenerate for batches larger than that;
                                  use top_k.
  --acq-uncertainty [dropout|laplace]
                                  How the acquisition uncertainty is computed.
                                  'dropout' (default) takes the spread of
                                  --mc-samples stochastic passes. 'laplace'
                                  uses a last-layer linearised Laplace
                                  posterior: sigma^2(x)=phi^T (Phi^T Phi/sn2 +
                                  I/tau) phi over the penultimate features,
                                  which is EXACT here because the head is
                                  linear and the loss is MSE. Unlike dropout
                                  it grows with distance from the labelled
                                  features rather than with local weight
                                  sensitivity, needs ONE forward pass instead
                                  of --mc-samples, and yields a candidate
                                  covariance of rank up to the feature
                                  dimension instead of --mc-samples-1, which
                                  entropy_batch needs. Incompatible with
                                  --n-ensemble > 1.
  --laplace-prior-precision FLOAT
                                  Gaussian prior precision on the last-layer
                                  weights for --acq-uncertainty laplace.
                                  Ranking is insensitive to it in this regime
                                  (Phi^T Phi/sn2 is O(1e4)), so the default is
                                  only a numerical safeguard.
  --allow-config-drift            Permit --resume-from to change the loop
                                  configuration (warm start, strategy,
                                  n_select, target, transform). Off by
                                  default: such a resume yields a run whose
                                  halves are not comparable, which is silent
                                  in the outputs.
  --eval-data-path TEXT           Path to external eval dataset (ROOT/CSV) for
                                  validation.
  --compute-full-metrics / --no-compute-full-metrics
                                  Compute comprehensive evaluation metrics
                                  (accuracy, MSE, RMSE).
  --y-transform [zscore|log]      Y transformation: 'log' (default,
                                  recommended) or 'zscore' (legacy).
  --mcmc-data-dir TEXT            Directory containing MCMC ROOT files for
                                  static evaluation (e.g.,
                                  data/neutralino_v4).
  --mcmc-max-samples INTEGER      Seeded uniform subsample cap on the MCMC set
                                  (emcee chains are ~96% repeated rows; the
                                  subsample preserves multiplicity weighting).
                                  0 disables.
  --static-eval-size INTEGER      Number of models to reserve from the random
                                  pool as a static evaluation set (default:
                                  100000).
  --data-dir TEXT                 Directory containing training ROOT files
                                  (default: data/18387358).
  --use-mcmc-loader / --no-use-mcmc-loader
                                  Load initial training data via
                                  load_mcmc_data (MCMC-specific filters:
                                  straddle-0.12, exclude <0.04). Default:
                                  load_pmssm_data via --data-dir.
  --train-baseline / --no-train-baseline
                                  Also train a random-sampling baseline model
                                  each iteration (default: enabled). Disable
                                  for pure capacity tests.
  --resume-from TEXT              Path to previous output dir to resume from
                                  (loads state.pt).
  --n-additional-iterations INTEGER
                                  If --resume-from given, run this many more
                                  iterations.
  --gpu-ids TEXT                  Comma-separated GPU IDs for AL and baseline
                                  models (default: 2,3).
  --pack-debris / --keep-debris   At the end of each iteration, summarise the
                                  SModelS winners and the LSP composition into
                                  the log plus a small JSON, keep the ntuples,
                                  and pack iteration_NNN/{worker_*,retry_*}
                                  into one debris.tar. On by default: each
                                  iteration otherwise leaves ~6,200 files and
                                  /ptmp has a hard filesystem inode ceiling
                                  that does not grow. --keep-debris restores
                                  the old behaviour of leaving every workspace
                                  in place.
  --seed INTEGER                  Master random seed propagated to torch /
                                  numpy / candidate pool (default: 42).
  --help                          Show this message and exit.
```

## `pipelines/active_learning_dnn.py`

```
Usage: active_learning_dnn.py [OPTIONS]

  Active learning pipeline for pMSSM relic density prediction (DNN variant).

  Trains PMSSMFeedForward, computes uncertainty via MC Dropout, and selects
  most informative points for data generation.

Options:
  --testing / --no-testing        Run in testing mode (small data).
  --n-iterations INTEGER          Number of active learning iterations.
  --n-candidates INTEGER          Candidate pool size.
  --n-select INTEGER              Number of points to select per iteration.
  --mc-samples INTEGER            Number of MC dropout forward passes.
  --epochs INTEGER                Training epochs per iteration (default:
                                  2000).
  --dropout FLOAT                 Dropout rate for MC dropout.
  --n-datasets INTEGER            Number of ROOT datasets to load.
  --n-samples INTEGER             Number of samples to use from data.
  --val-fraction FLOAT            Fraction of data reserved for validation
                                  (default: 0.2). Applied to initial data and
                                  each batch of new points.
  --output-dir TEXT               Output directory.
  --generate-data / --no-generate-data
                                  Generate new models using Run3ModelGen.
  --min-gen-fraction FLOAT        Minimum fraction of n-select that must be
                                  generated successfully before stopping
                                  retries (default: 0.6).
  --max-gen-attempts INTEGER      Maximum number of generation attempts per
                                  iteration (default: 10).
  --gen-workers INTEGER           Number of parallel genModels.py workers per
                                  generation attempt (default: 1).
  --head [classification|lsq_classification|regression]
                                  Acquisition head (see pmssm.heads).
                                  'regression' (default) is the production
                                  path: MSE on the transformed target,
                                  acquisition by predictive variance.
                                  'classification' trains binary cross-entropy
                                  on the verdict 1[t>0] and must be paired
                                  with --selection-strategy bald or
                                  cls_entropy, since its output is a logit and
                                  the mean-based tolerance cut and proximity
                                  weighting have no meaning in logit units.
  --selection-strategy [top_k|top_k_tol_only|tol_only_random|entropy_batch|bald|cls_entropy]
                                  Selection strategy: top_k, top_k_tol_only
                                  (short-circuit, no proximity weighting), or
                                  entropy_batch (default).
  --entropy-blur FLOAT            Entropy smoothing parameter (entropy_batch
                                  only).
  --entropy-beta FLOAT            Gibbs sampling temperature (entropy_batch
                                  only).
  --entropy-pool-size INTEGER     Focused pool size for entropy_batch pre-
                                  filtering.
  --candidate-generation [uniform|lhs]
                                  Candidate pool generation method: uniform
                                  random or Latin Hypercube Sampling (default:
                                  lhs).
  --proximity-sampling FLOAT      Gaussian proximity weighting width around
                                  target value (0 to disable, default: 0.1).
  --tolerance-sampling FLOAT      Hard cut: keep only candidates within
                                  ±tolerance of threshold in transformed space
                                  (0 to disable, default: 1.0).
  --target [CLs|CrossSection|DMRD|ExpR]
                                  Observable the surrogate learns and the AL
                                  loop targets (default: DMRD). See
                                  active_learning_transformer.py for details.
  --no-mcmc-eval                  Ignore --mcmc-data-dir and run without an
                                  MCMC reference set, for targets that have no
                                  posterior reference.
  --target-value FLOAT            Target relic density value for proximity
                                  weighting (default: 0.12).
  --config-file TEXT              YAML config file (overrides CLI args).
                                  Supports parameter sweeps.
  --sweep-index INTEGER           Sweep combination index (requires --config-
                                  file).
  --early-stopping / --no-early-stopping
                                  Enable early stopping on validation loss
                                  (default: enabled).
  --patience INTEGER              Early stopping patience (epochs without
                                  improvement, default: 200).
  --warm-starting / --no-warm-starting
                                  Warm-start from previous iteration
                                  checkpoint (default: enabled).
  --eval-data-path TEXT           Path to external eval dataset (ROOT/CSV) for
                                  validation.
  --compute-full-metrics / --no-compute-full-metrics
                                  Compute comprehensive evaluation metrics
                                  (accuracy, MSE, RMSE).
  --y-transform [zscore|log]      Y transformation: 'log' (default,
                                  recommended) or 'zscore' (legacy).
  --mcmc-data-dir TEXT            Directory containing MCMC ROOT files for
                                  static evaluation (e.g.,
                                  data/neutralino_v4).
  --mcmc-max-samples INTEGER      Seeded uniform subsample cap on the MCMC set
                                  (emcee chains are ~96% repeated rows; the
                                  subsample preserves multiplicity weighting).
                                  0 disables.
  --static-eval-size INTEGER      Number of models to reserve from the random
                                  pool as a static evaluation set (default:
                                  100000).
  --data-dir TEXT                 Directory containing training ROOT files
                                  (default: data/18387358).
  --use-mcmc-loader / --no-use-mcmc-loader
                                  Load initial training data via
                                  load_mcmc_data (MCMC-specific filters:
                                  straddle-0.12, exclude <0.04). Default:
                                  load_pmssm_data via --data-dir.
  --train-baseline / --no-train-baseline
                                  Also train a random-sampling baseline model
                                  each iteration (default: enabled). Disable
                                  for pure capacity tests.
  --resume-from TEXT              Path to previous output dir to resume from
                                  (loads state.pt).
  --n-additional-iterations INTEGER
                                  If --resume-from given, run this many more
                                  iterations.
  --gpu-ids TEXT                  Comma-separated GPU IDs for AL and baseline
                                  models (default: 2,3).
  --n-ensemble INTEGER            Train K independent members per iteration
                                  and use their disagreement as the
                                  acquisition uncertainty instead of MC
                                  dropout (K=1, the default, keeps MC
                                  dropout). Requires --no-train-baseline. The
                                  across-member covariance has rank <= K-1, so
                                  entropy_batch is degenerate for larger
                                  batches; use top_k.
  --acq-uncertainty [dropout|laplace]
                                  How the acquisition uncertainty is computed.
                                  'dropout' (default) takes the spread of
                                  --mc-samples stochastic passes. 'laplace'
                                  uses a last-layer linearised Laplace
                                  posterior over the penultimate features,
                                  which is EXACT here because the head is
                                  linear and the loss is MSE. It grows with
                                  distance from the labelled features rather
                                  than with local weight sensitivity, needs
                                  ONE forward pass instead of --mc-samples,
                                  and yields a candidate covariance of rank up
                                  to the feature dimension instead of --mc-
                                  samples-1, which entropy_batch needs.
                                  Incompatible with --n-ensemble > 1.
  --laplace-prior-precision FLOAT
                                  Gaussian prior precision on the last-layer
                                  weights for --acq-uncertainty laplace.
                                  Ranking is insensitive to it in this regime,
                                  so the default is a numerical safeguard.
  --pack-debris / --keep-debris   At the end of each iteration, summarise the
                                  SModelS winners and the LSP composition into
                                  the log plus a small JSON, keep the ntuples,
                                  and pack iteration_NNN/{worker_*,retry_*}
                                  into one debris.tar. On by default: each
                                  iteration otherwise leaves ~6,200 files and
                                  /ptmp has a hard filesystem inode ceiling
                                  that does not grow. --keep-debris restores
                                  the old behaviour of leaving every workspace
                                  in place.
  --seed INTEGER                  Master random seed propagated to torch /
                                  numpy / candidate pool (default: 42).
  --d-model INTEGER               DNN per-parameter embedding dim (default:
                                  64).
  --num-layers INTEGER            DNN hidden layer count (default: 4).
  --dim-feedforward INTEGER       DNN hidden layer width (default: 256).
  --help                          Show this message and exit.
```

## `pipelines/active_learning_gp.py`

```
Usage: active_learning_gp.py [OPTIONS]

  Active learning pipeline for pMSSM relic density prediction using GP models.

  Trains ExactGP, DeepGP, SparseGP, or MLP; computes uncertainty via GP
  posterior variance (or entropy-based batch selection); and selects most
  informative points for data generation.

Options:
  --testing / --no-testing        Run in testing mode (small data).
  --n-iterations INTEGER          Number of active learning iterations.
  --n-candidates INTEGER          Candidate pool size.
  --n-select INTEGER              Number of points to select per iteration.
  --n-datasets INTEGER            Number of ROOT datasets to load.
  --n-samples INTEGER             Number of samples to use from data.
  --val-fraction FLOAT            Fraction of data reserved for validation
                                  (default: 0.2). Applied to initial data and
                                  each batch of new points.
  --output-dir TEXT               Output directory.
  --generate-data / --no-generate-data
                                  Generate new models using Run3ModelGen.
  --min-gen-fraction FLOAT        Minimum fraction of n-select that must be
                                  generated successfully before stopping
                                  retries (default: 0.6).
  --max-gen-attempts INTEGER      Maximum number of generation attempts per
                                  iteration (default: 10).
  --gen-workers INTEGER           Number of parallel genModels.py workers per
                                  generation attempt (default: 1).
  --no-mcmc-eval                  Ignore --mcmc-data-dir and run without an
                                  MCMC reference set, for targets that have no
                                  posterior reference.
  --target [CLs|CrossSection|DMRD|ExpR]
                                  Target function to predict.
  --model-type [exact_gp|deep_gp|sparse_gp|mlp|laplace_gpc]
                                  Model type. 'laplace_gpc' is the binary
                                  Laplace GP classifier of Rasmussen &
                                  Williams section 3.4 and is a classifier
                                  only: pair it with --head classification.
  --kernel TEXT                   Kernel type (RBF, Matern, RQK,
                                  SpectralMixture, RBF+Matern).
  --lengthscale FLOAT             Initial kernel lengthscale.
  --noise FLOAT                   Initial noise level.
  --jitter FLOAT                  Cholesky jitter.
  --learning-rate FLOAT           Optimizer learning rate.
  --epochs INTEGER                Max training epochs per AL iteration
                                  (default: 2000).
  --early-stopping / --no-early-stopping
                                  Enable early stopping on validation loss.
  --patience INTEGER              Early stopping patience (epochs without
                                  improvement, default: 200).
  --use-ard / --no-ard            Use ARD (Automatic Relevance Determination).
  --use-dkl / --no-dkl            Use Deep Kernel Learning (ExactGP only).
  --feature-dim INTEGER           DKL feature dimension.
  --num-hidden-dims INTEGER       DeepGP hidden layer dimensions.
  --num-middle-dims INTEGER       DeepGP middle layer dimensions (0 = no
                                  middle layer).
  --num-inducing-max INTEGER      Max inducing points (DeepGP/SparseGP).
  --inducing-strategy [kmeans|vanilla]
                                  Inducing point initialization (sparse_gp
                                  only).
  --gp-num-samples INTEGER        Number of likelihood samples (DeepGP).
  --batch-size INTEGER            Batch size (DeepGP/SparseGP).
  --warm-starting / --no-warm-starting
                                  Warm-start from previous iteration.
  --m-nu FLOAT                    Matern nu parameter.
  --num-mixtures INTEGER          Number of mixtures for SpectralMixture
                                  kernel.
  --head [classification|lsq_classification|regression]
                                  Acquisition head. 'regression' is production
                                  (MSE on t). 'classification' attaches a
                                  Bernoulli likelihood and so is available for
                                  the variational models (deep_gp, sparse_gp)
                                  only; 'lsq_classification' regresses the +-1
                                  verdict under the Gaussian likelihood, which
                                  keeps the exact GP exact. Both verdict heads
                                  pair with --selection-strategy bald or
                                  cls_entropy.
  --selection-strategy [top_k|top_k_tol_only|tol_only_random|entropy_batch|bald|cls_entropy]
                                  Selection strategy: top_k or entropy_batch
                                  (default).
  --entropy-blur FLOAT            Entropy smoothing parameter (entropy_batch
                                  only).
  --entropy-beta FLOAT            Gibbs sampling temperature (entropy_batch
                                  only).
  --tolerance-sampling FLOAT      Hard cut: keep only candidates within
                                  ±tolerance of threshold in transformed space
                                  (0 to disable, default: 1.0).
  --proximity-sampling FLOAT      Gaussian proximity weighting width around
                                  target value (0 to disable, default: 0.1).
  --entropy-pool-size INTEGER     Focused pool size for entropy_batch pre-
                                  filtering.
  --candidate-source [pool|mcmc]  Candidate source: 'pool' (random sampling,
                                  default) or 'mcmc' (theoretical-limit /
                                  oracle mode — restrict candidates to the
                                  pre-loaded MCMC dataset; --generate-data is
                                  forced off).
  --compute-full-metrics / --no-compute-full-metrics
                                  Compute comprehensive GoF metrics (accuracy,
                                  chi2, pulls, etc.).
  --eval-data-path TEXT           Path to true eval dataset (ROOT/CSV).
  --mcmc-data-dir TEXT            Directory containing MCMC ROOT files for
                                  static evaluation (e.g. data/neutralino_v4).
  --mcmc-max-samples INTEGER      Seeded uniform subsample cap on the MCMC set
                                  (emcee chains are ~96% repeated rows; the
                                  subsample preserves multiplicity weighting).
                                  0 disables.
  --static-eval-size INTEGER      Number of models to reserve from the random
                                  pool as a static evaluation set (default:
                                  100000).
  --track-lengthscales / --no-track-lengthscales
                                  Save learned ARD lengthscales per iteration.
  --advanced-plots / --no-advanced-plots
                                  Generate advanced diagnostic plots
                                  (heatmaps, residuals).
  --config-file TEXT              YAML config file (overrides CLI args).
                                  Supports parameter sweeps.
  --sweep-index INTEGER           Sweep combination index (requires --config-
                                  file).
  --data-dir TEXT                 Directory containing training ROOT files
                                  (default: data/18387358).
  --resume-from TEXT              Path to previous output dir to resume from
                                  (loads state.pt).
  --n-additional-iterations INTEGER
                                  If --resume-from given, run this many more
                                  iterations.
  --gpu-ids TEXT                  Comma-separated GPU IDs for AL and baseline
                                  models (default: 0,1).
  --pack-debris / --keep-debris   At the end of each iteration, summarise the
                                  SModelS winners and the LSP composition into
                                  the log plus a small JSON, keep the ntuples,
                                  and pack iteration_NNN/{worker_*,retry_*}
                                  into one debris.tar. On by default: each
                                  iteration otherwise leaves ~6,200 files and
                                  /ptmp has a hard filesystem inode ceiling
                                  that does not grow. --keep-debris restores
                                  the old behaviour of leaving every workspace
                                  in place.
  --seed INTEGER                  Master random seed propagated to torch /
                                  numpy / candidate pool (default: 42).
  --help                          Show this message and exit.
```

## `pipelines/active_learning_tabpfn.py`

```
Usage: active_learning_tabpfn.py [OPTIONS]

  Active learning pipeline for pMSSM relic density prediction using TabPFN.

  Uses TabPFN's native predictive variance for uncertainty and ensemble
  diversity for entropy-based batch selection.

Options:
  --testing / --no-testing        Run in testing mode (small data).
  --n-iterations INTEGER          Number of active learning iterations.
  --n-candidates INTEGER          Candidate pool size.
  --n-select INTEGER              Number of points to select per iteration.
  --n-ensemble-samples INTEGER    Number of TabPFN ensemble runs for
                                  uncertainty (default: 16).
  --n-datasets INTEGER            Number of ROOT datasets to load.
  --n-samples INTEGER             Number of samples to use from data.
  --val-fraction FLOAT            Fraction of data reserved for validation
                                  (default: 0.2). Applied to initial data and
                                  each batch of new points.
  --output-dir TEXT               Output directory.
  --generate-data / --no-generate-data
                                  Generate new models using Run3ModelGen.
  --min-gen-fraction FLOAT        Minimum fraction of n-select that must be
                                  generated successfully before stopping
                                  retries (default: 0.6).
  --max-gen-attempts INTEGER      Maximum number of generation attempts per
                                  iteration (default: 10).
  --gen-workers INTEGER           Number of parallel genModels.py workers per
                                  generation attempt (default: 1).
  --selection-strategy [top_k|top_k_tol_only|tol_only_random|entropy_batch]
                                  Selection strategy: top_k (default),
                                  top_k_tol_only (short-circuit, no
                                  proximity), or entropy_batch (prohibitively
                                  expensive for TabPFN).
  --entropy-blur FLOAT            Entropy smoothing parameter (entropy_batch
                                  only).
  --entropy-beta FLOAT            Gibbs sampling temperature (entropy_batch
                                  only).
  --entropy-pool-size INTEGER     Focused pool size for entropy_batch pre-
                                  filtering.
  --candidate-generation [uniform|lhs]
                                  Candidate pool generation method: uniform
                                  random or Latin Hypercube Sampling (default:
                                  lhs).
  --proximity-sampling FLOAT      Gaussian proximity weighting width around
                                  target value (0 to disable, default: 0.1).
  --tolerance-sampling FLOAT      Hard cut: keep only candidates within
                                  ±tolerance of threshold in transformed space
                                  (0 to disable, default: 1.0).
  --target [CLs|CrossSection|DMRD|ExpR]
                                  Observable the surrogate learns and the AL
                                  loop targets (default: DMRD). See
                                  active_learning_transformer.py for details.
  --no-mcmc-eval                  Ignore --mcmc-data-dir and run without an
                                  MCMC reference set, for targets that have no
                                  posterior reference.
  --target-value FLOAT            Target relic density value for proximity
                                  weighting (default: 0.12).
  --config-file TEXT              YAML config file (overrides CLI args).
                                  Supports parameter sweeps.
  --sweep-index INTEGER           Sweep combination index (requires --config-
                                  file).
  --mcmc-data-dir TEXT            Directory containing MCMC ROOT files for
                                  static evaluation (e.g.,
                                  data/neutralino_v4).
  --mcmc-max-samples INTEGER      Seeded uniform subsample cap on the MCMC set
                                  (emcee chains are ~96% repeated rows; the
                                  subsample preserves multiplicity weighting).
                                  0 disables.
  --static-eval-size INTEGER      Number of models to reserve from the random
                                  pool as a static evaluation set (default:
                                  100000).
  --data-dir TEXT                 Directory containing training ROOT files
                                  (default: data/18387358).
  --resume-from TEXT              Path to previous output dir to resume from
                                  (loads state.pt).
  --n-additional-iterations INTEGER
                                  If --resume-from given, run this many more
                                  iterations.
  --gpu-ids, --gpu-id TEXT        Comma-separated GPU IDs. One: sequential
                                  AL+baseline on that GPU. Two: AL on the
                                  first, baseline on the second, in parallel
                                  (default: 0). --gpu-id is kept as an alias
                                  for backward compatibility.
  --pack-debris / --keep-debris   At the end of each iteration, summarise the
                                  SModelS winners and the LSP composition into
                                  the log plus a small JSON, keep the ntuples,
                                  and pack iteration_NNN/{worker_*,retry_*}
                                  into one debris.tar. On by default: each
                                  iteration otherwise leaves ~6,200 files and
                                  /ptmp has a hard filesystem inode ceiling
                                  that does not grow. --keep-debris restores
                                  the old behaviour of leaving every workspace
                                  in place.
  --seed INTEGER                  Master random seed propagated to torch /
                                  numpy / candidate pool (default: 42).
  --help                          Show this message and exit.
```

## Other command lines

Every script under `scripts/` answers `--help` except the positional-argument
ones noted in `scripts/README.md` (`best_analysis_arms.py`,
`diversity_tests.py`, `extract_emcee_chain_h5.py`, `acquisition_efficacy.py`,
`plot_transformer_architecture.py`). The ones a user is likely to call:

| command | purpose |
|---|---|
| `python scripts/campaign/campaign_chase.py --cells <csv> --dry-run` | preview what the chase would resubmit |
| `python scripts/campaign/update_sweep_manifest.py [--manifest ...] [--all]` | refresh manifest statuses |
| `python scripts/campaign/prune_run_debris.py --runs <glob> [--apply]` | pack finished iterations |
| `bash scripts/campaign/archive_runs.sh [SRC] [DEST] [GLOB]` | archive runs |
| `python scripts/analysis/analyse_runs.py --run-dirs ... --output-dir ...` | cross-run metrics (`doc/ANALYSIS_METRICS.md`) |
| `python scripts/figures/plot_hit_rate_trajectories_multiseed.py --manifest ... [--compute-accuracy]` | the central figure script |
| `python -m pmssm.gifs <run dir>` | remake a run's animations |
| `python tests/check_repo_layout.py [--help-all] [--paper DIR]` | the layout guard |
