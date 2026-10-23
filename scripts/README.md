# scripts/

Everything that reads finished runs. Run any file by path from any directory
(`python scripts/figures/plot_best_config.py --help`); the bootstrap at the top
of each file puts the repository root on `sys.path`, and sibling scripts are
imported as `scripts.<folder>.<module>`. The hub modules that many others import
are marked with *(hub)*.

`slurm/campaign/submit_campaign_regen.sh` runs the scripts marked *regen* in the
documented order and copies the renders into the paper; see
`doc/REPRODUCING_THE_PAPER.md` for the figure-by-figure map and the stale-cache
traps.

## figures/ — paper and supplementary figure renderers

| script | produces |
|---|---|
| `plot_hit_rate_trajectories_multiseed.py` *(hub, regen)* | mean ± SEM hit-rate, hits-per-desired and (with `--compute-accuracy`) verdict-accuracy trajectories per cell; writes `random_baseline_prevalence.json`, `data_efficiency_best_per_model.json` |
| `plot_hit_rate_seeds_per_model.py` *(regen)* | every seed's hit-rate trajectory for one model, no aggregation |
| `plot_hit_rate_single_run.py` | single-run hit-rate panels |
| `plot_r2_trajectories_multiseed.py` *(regen)* | validation and static R², n_train trajectories |
| `plot_r2_mcmc_trajectories_multiseed.py` | R² on the MCMC reference set |
| `plot_mmd2_trajectories_multiseed.py` | MMD² between the training set and the MCMC reference |
| `plot_mse_trajectories_multiseed.py` *(regen)* | best-per-model MSE trajectories, one panel per evaluation set, with the ratio strip |
| `plot_iteration_metrics_seeds.py` | multi-seed version of a run's `iteration_metrics.png` |
| `plot_pairwise_input_summary.py` *(hub, regen)* | per-cell pairwise input-density grids |
| `plot_pairwise_paper.py` | the paper-styled pairwise density of one cell |
| `plot_al_input_target_diagnostics.py` *(hub, regen)* | corner plots and input-versus-target panels of the pooled training sets, and of the MCMC reference |
| `plot_compute_vs_dataset.py` *(hub, regen)* | labelled-set size against cumulative surrogate compute |
| `plot_support_efficiency.py` *(hub, regen)* | in-band support covered against budget, normalised by each dataset's own cost |
| `plot_diversity_vs_budget.py` *(regen)* | space-filling quality against budget |
| `plot_prelim_paper_style.py` *(hub, regen)* | the acquisition-arm figures (hit rate, hits per desired, accuracy against labelled-set size), one panel per model |
| `plot_prelim_support.py` *(hub, regen)* | in-band support covered per arm |
| `plot_prelim_compute_support.py` *(regen)* | in-band support against cumulative compute per arm |
| `plot_best_config.py` *(hub)* | main-body figures: the best configuration per surrogate, one per (target, metric) |
| `plot_hits_per_desired_heads.py` *(hub)* | Appendix A: yield per simulator call, regression picks against classification arms |
| `plot_accuracy_heads.py` *(hub)* | Appendix A: verdict accuracy in the same layout |
| `plot_support_heads.py` | Appendix A: in-band support in the same layout, `--metric budget|compute` |
| `plot_strategy_pair_heads.py` | entropy batch against tol+random for the dropout surrogates, both targets |
| `plot_distribution_overlay.py` *(hub)* | in-band input marginals plus the target marginal, one row |
| `plot_distribution_overlay_rows.py` *(hub)* | the same in two rows, regression head above classification |
| `plot_corner_overlay.py` | corner version of the distribution overlay |
| `plot_omega_vs_m2_lsp.py` | Ω h² against a gaugino or higgsino mass, coloured by LSP type |
| `plot_hitrate_vs_labelled.py` | hit rate against labelled-set size and the gap between surrogates |
| `plot_weighted_accuracy.py` | verdict accuracy weighted by proximity to the boundary, from the static prediction cache |
| `plot_transformer_architecture.py` | `doc/transformer_architecture.png` |

## analysis/ — tables, numbers and diagnostics

| script | produces |
|---|---|
| `analyse_runs.py` *(hub)* | loads `state.pt` checkpoints; quality, physics and diversity metrics across runs (`load_run`, `compute_hit_rate_trajectory`, the free-parameter tables); also a CLI with eight summary plots and `summary.csv` |
| `mcmc_diagnostics.py` *(hub, regen)* | split-R̂ and bulk-ESS per parameter for chain outputs; holds the per-model picks (`DEFAULT_AL_PICKS`, `picks_with_tag`) |
| `coverage_saturation.py` *(hub, regen)* | the in-band support definition (`build_target`, `coverage_of`) and the union-coverage saturation figure |
| `compute_yield_comparison.py` *(regen)* | distinct in-band models per simulator call, AL against MCMC and random (paper table `tab:yield`) |
| `composition_fractions.py` *(hub)* | electroweakino composition fractions per scan (table `tab:composition`) |
| `evaluate_uq.py` *(hub, regen)* | post-hoc uncertainty-quantification evaluation from checkpoints (table `tab:uq`); needs a GPU |
| `rank_uniformity_al.py` | rank-statistic uniformity across seed replicas, through Run3ModelGen's `emcee_diagnostics` (run `--export-only` with the torch environment first, then the Run3ModelGen environment) |
| `best_analysis_arms.py`, `best_analysis_from_smodels.py` | the most sensitive SModelS analysis at the exclusion boundary, per arm and for the pool |
| `diversity_tests.py` | whether AL's narrower analysis mix costs anything |
| `find_islands.py` | islands of the target region in the baseline scan, and who finds them first |
| `funnel_analysis.py` | the low-|M₁| funnel region: who samples it, who profits |
| `exclusion_confusion.py` | confusion matrices of the exclusion verdict, AL model against random baseline |
| `mode_switch_diagnostic.py` | whether the neural surrogates are steered to the mode-switching ridges of r_exp |
| `prior_predictive_check.py` | prior-predictive / importance-sampling check of the MCMC reference |
| `extract_emcee_chain_h5.py` | interleaved (M₁, M₂, μ, Ω) from the raw emcee chains |
| `static_predictions_cache.py` *(hub)* | caches per-point predictions on the static-random evaluation set from checkpoints |

## campaign/ — sweep and campaign operations

| script | does |
|---|---|
| `campaign_chase.py` | keeps every campaign cell advancing to 200 iterations: reads the cells CSV and the sweep manifests, skips busy cells, submits bundled resumes |
| `update_sweep_manifest.py` *(regen)* | refreshes the `status` column of a sweep manifest from `sacct` and `state.pt` |
| `build_joint_manifest.py` *(regen)* | the manifest that selects the most advanced run directory per (cell, seed) |
| `build_probe_manifest.py` | a manifest for runs that were not launched by the sweep script |
| `prune_run_debris.py` | packs the simulator workspaces of finished iterations into one `debris.tar` each (see `doc/STORAGE.md`) |
| `headtest_progress.py` | reads the head/strategy test runs at whatever iteration they have reached |
| `archive_runs.sh` | per-run tarballs of the irreplaceable files on non-purged storage |
| `run_probe_quality.sh` | the quality metrics of the extended-budget and large-batch probes |

## tutorial/ — the walkthrough of doc/TUTORIAL.md

| script | does |
|---|---|
| `run_tutorial.sh` | the tutorial end to end: fetches the sample, runs two oracle-mode loops, builds their manifest, renders the figures (`pixi run tutorial`) |
| `fetch_data.sh` | downloads and verifies the 12 MB data sample into `data/tutorial` |
| `make_tutorial_sample.py` | writes that sample from the full pools (run once, on the cluster) |

## experiments/ — one-off studies behind supplementary sections

| script | question |
|---|---|
| `dispersion_experiment.py` | why the AL-trained deep GP's predictions disperse in the posterior region |
| `laplace_uq_probe.py`, `plot_laplace_input_overlays.py` | is the transformer's acquisition uncertainty fixable post hoc with a Laplace posterior |
| `verify_laplace_gpc.py` | the Laplace GP classifier against Rasmussen and Williams |
| `head_swap_stage_a.py`, `plot_head_swap_stage_a.py` | classification against regression heads: acquisition precision per selector |
| `entropy_batch_limit.py` | where entropy-batch selection stops working as a function of batch size |
| `transformer_capacity_sweep.py` | whether the transformer's uncertainty deficit comes from over-parametrisation |
| `ensemble_comparison.py` | ensemble disagreement against MC-dropout spread as the acquisition signal |
| `acquisition_efficacy.py` | whether the uncertainty ranking adds anything beyond the hard tolerance cut |
| `deepgp_validation_cost.py`, `deepgp_posthoc_accuracy.py` | where the deep GP's cost goes, and its verdict accuracy recomputed from checkpoints |
| `tabpfn_posthoc_accuracy.py` | TabPFN's verdict accuracy against the current references |
| `plot_probe_input_overlays.py` | probe-versus-benchmark input marginals: does a bigger budget move the search |
| `plot_headtest_accuracy.py`, `plot_hit_rate_preliminary.py` | the single-seed head/strategy test runs that preceded the campaign |
