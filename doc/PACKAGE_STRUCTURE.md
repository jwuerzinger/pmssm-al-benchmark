# The `pmssm` package

The library shared by the four pipelines and by every analysis script. It is
imported from the repository root (every script's bootstrap puts the root on
`sys.path`; nothing is installed). `pmssm/__init__.py` re-exports the public
functions listed below.

| module | contents |
|---|---|
| `config.py` | `PARAM_ORDER` (the 19 stored parameters), `PARAM_RANGES` (fixed parameters have degenerate ranges), `TARGET_CONFIG` (the target registry, `doc/TARGETS.md`), `MODELGEN_STEP_DEFS` (the Run3ModelGen steps), `GP_RANGE_DICT`, `CSV_TO_MODELGEN` |
| `paths.py` | the data, output, analysis, archive, SModelS and manuscript roots as environment variables with the campaign's defaults; `POOLS`, `MCMC_REFERENCE`, `MANIFESTS`, `ANALYSIS_DIRS`, `JOINT_DIR`, `POOL_CACHE`, `CAMPAIGN_DIR` |
| `data.py` | `load_pmssm_data` (pool ingest with the target's validity cuts), `load_mcmc_data`, `count_mcmc_rows`, `load_mcmc_ordered`, `make_split`, `compute_stats`, `build_norm_tensors`, `normalize_x`, `unnormalize_x`, `transform_y`, `inverse_transform_y` (log(Y / Y₀)), `target_validity_mask`, `split_mcmc_for_oracle` |
| `datasets.py` | `PMSSMDataset`, `Dummy_PMSSMDataset` |
| `models/` | `PMSSMTransformer`, `PMSSMTransformerTabular` (the production transformer: one token per parameter, MC dropout), `PMSSMFeedForward` (the DNN), `is_transformer`, `get_model_name` |
| `heads.py` | the acquisition heads: `HEADS`, `get_head`, `head_for_run`, `head_names`, `LeastSquaresVerdictHead` (`doc/ACQUISITION.md`) |
| `selection.py` | `generate_candidate_pool` (uniform or Latin hypercube), the tolerance cut and proximity weighting, `select_points` dispatching to `top_k`, `top_k_tol_only`, `tol_only_random`, `entropy_batch`, `head_score`; `select_entropy_batch_mc` adapts the entropy strategy of `al_pmssmwithgp` to MC samples |
| `uncertainty.py` | `compute_uncertainty_mc_dropout`, `compute_uncertainty_ensemble`, `compute_uncertainty_laplace`, `compute_uncertainty_gp`, `compute_uncertainty_head`, `compute_uncertainty_gp_head` |
| `training.py` | `train_with_validation` and `train_model_worker` (neural surrogates, early stopping, warm start, one worker per GPU), `create_gp_model`, `train_gp_model`, `train_gp_worker` (exact, deep, sparse GPs and the Laplace classifier from the submodule), `collect_worker_result`, `model_has_likelihood`, `model_is_variational` |
| `evaluation.py` | `compute_gp_r2`, `compare_random_predictions`, `compute_comprehensive_metrics` (accuracy, χ², pulls), `extract_lengthscales` |
| `accuracy.py` | `binary_accuracy`, `update_accuracy_trajectory`, `write_iter_accuracies`: the in-training verdict accuracy on every evaluation set, written to `accuracy_trajectory.json` |
| `resume.py` | `state.pt` loading for `--resume-from`; the state is overwritten atomically after each iteration |
| `iteration_housekeeping.py` | the end-of-iteration summarise-and-pack step (`finalise_iteration`, `pack_iteration_debris`, `summarise_smodels`, `load_iteration_smodels`), `doc/STORAGE.md` |
| `model_generation.py` | the Run3ModelGen interface: `save_selected_points`, `generate_models_from_csv` (parallel `genModels.py` workers with the target's step list, retries, the SModelS database check), `load_generated_data`, `load_true_eval_dataset` |
| `visualization.py` | per-iteration diagnostics (`plot_losses`, `scatter_true_vs_pred`, `hist_true_vs_pred`, `plot_candidate_uncertainty`, `plot_parallel_coordinates`, `plot_iteration_metrics`, `plot_advanced_diagnostics`, ...), `classify_lsp_type`, `gp_predict` |
| `gifs.py` | `generate_gifs`, the end-of-run animations; also `python -m pmssm.gifs <run dir>` |
| `logging_utils.py` | `setup_logging`, `setup_worker_logging` (`doc/LOGGING_INFO.md`) |

## Where the pipelines fit

`pipelines/active_learning_{transformer,dnn,gp,tabpfn}.py` hold the loop
itself: argument parsing, data ingest, the per-iteration train, score, select,
label and record sequence, checkpointing and the end-of-run summary. They
differ in the surrogate and in what it costs to query, which is why the
candidate budgets and pre-filters differ (`doc/IMPLEMENTATION_DETAILS.md`); the
data, selection, evaluation and bookkeeping code is shared through the
package. The GP pipeline additionally imports the model classes of the
`al_pmssmwithgp` submodule (`al_pmssmwithgp/model`, put on `sys.path` by the
pipeline and the batch scripts).

## Import rule

`from pmssm import ...` everywhere; analysis scripts import each other as
`scripts.<folder>.<module>`; the only `sys.path` manipulation is the bootstrap
at the top of each entry file. `tests/check_repo_layout.py` enforces this and
rejects import cycles between scripts.
