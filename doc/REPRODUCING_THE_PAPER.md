# Reproducing the paper's figures and tables

## The entry point

`slurm/campaign/submit_campaign_regen.sh` renders every figure family in the
right order from the sweep manifests and copies the renders into the
manuscript's `figures/` and `figures_expr/` directories. To render once
without touching the manuscript and without chaining a successor:

```bash
CAMPAIGN_ID=<arm sweep id> REGEN_HORIZON=2000-01-01 COPY_TO_PAPER=0 \
    sbatch --partition=apu1 --gres=gpu:1 --mem=110000 --export=ALL \
    slurm/campaign/submit_campaign_regen.sh
```

It needs the two sweep manifests (`doc/CAMPAIGN.md`), the pools and the
reference (`doc/DATA.md`), a GPU for the accuracy and UQ passes, and about
twelve hours; `SKIP_SLOW=1` drops the yield table and the UQ evaluation.
Renders land in `$PMSSM_ANALYSIS_ROOT/all_runs` (relic density),
`expr_runs` (exclusion) and `joint/arms_{dmrd,expr}` (the arm figures).
`doc/CAMPAIGN.md` lists the steps.

## Order and dependencies

When regenerating by hand, keep this order; the first step writes a file the
later ones divide by.

1. `scripts/figures/plot_hit_rate_trajectories_multiseed.py` writes
   `random_baseline_prevalence.json`, read by `ensemble_comparison.py`, and
   `data_efficiency_best_per_model.json`. Add `--compute-accuracy` for the
   accuracy figures and `--accuracy-cache-refresh` whenever the reference
   changed: the per-run `accuracy_trajectory.json` carries no fingerprint of
   the reference and goes stale silently.
2. `scripts/analysis/compute_yield_comparison.py` recomputes the band
   prevalence itself, so it is order independent, but it must be given the
   same `--baseline-require-neutralino-lsp` setting as step 1 or the two
   disagree by about 2 % with nothing in either log to show it.
3. `scripts/experiments/ensemble_comparison.py` after step 1.
4. `scripts/analysis/composition_fractions.py` with no `--models` filter: a
   filtered run leaves a partial `composition_fractions.json`.
5. `scripts/analysis/mcmc_diagnostics.py`, `scripts/analysis/evaluate_uq.py`
   (GPU), `scripts/figures/plot_compute_vs_dataset.py`.
6. The probe appendices: `scripts/campaign/build_probe_manifest.py`, then
   `scripts/campaign/run_probe_quality.sh`.

Every number in the manuscript was transcribed by hand from script output
or from the JSON next to each render, so regenerating is half the job: diff
the `.tex` against the JSONs afterwards.

## Which manifest

- `sweep_manifest_benchmark.csv` (written by the regeneration job): the
  40-iteration benchmark, with the campaign's arm rows and the `*_laplace`
  rows removed. The main benchmark figures read it; for four cells the
  seed-1 rows point at frozen 40-iteration copies kept outside the live
  output root on purpose.
- `sweep_manifest_paper.csv`: the same cells with the live, 200-iteration
  directories, for every figure that follows the runs past 40 iterations
  (the best-configuration, heads and overlay figures). Reading the plain
  sweep manifest instead silently truncates those curves. The exclusion
  variant marks one stalled TabPFN seed, which the heads scripts take through
  `--seeds-axis-override tabpfn_expr:4`.
- `joint/manifest_{dmrd,expr}.csv`: the most advanced directory per (cell,
  seed), for the arm figures.

## Main-body figures

| figure files | script and options |
|---|---|
| `best_config_{yield,accuracy,support}.png` | `scripts/figures/plot_best_config.py --target DMRD\|ExpR --metric yield\|accuracy\|support` (`--ratio-panel` for accuracy), one panel per target holding the best configuration of every surrogate; the picks are the `TARGETS[...]["picks"]` table in the script |
| `distribution_overlay_rows.png` | `scripts/figures/plot_distribution_overlay_rows.py --best-picks ...`: in-band input marginals and the target marginal, regression head above classification |
| `corner_overlay_best.png`, `corner_overlay_allpoints.png` | `scripts/figures/plot_corner_overlay.py --manifest all_runs/sweep_manifest_paper.csv` (`--target ExpR --model-tag _expr` for the exclusion) |
| `hits_per_desired_heads.png`, `accuracy_heads.png`, `support_heads.png` | `scripts/figures/plot_hits_per_desired_heads.py`, `plot_accuracy_heads.py`, `plot_support_heads.py --metric budget --compute-marks 0.3,3,30`: two panels, the best regression cell per surrogate against the classification-entropy arms; `--reg-picks` and `--cls-picks model:strategy:warm,...` override the defaults in `REG_PICKS` and `CLS_PICKS`; line figures use `--smooth 15 --min-seeds-axis 5` |
| `hits_per_desired_strategy_pair.png` | `scripts/figures/plot_strategy_pair_heads.py`: entropy batch against tol+random for the dropout surrogates, both targets |

The picks follow one rule: the best configuration per surrogate, ranked by
in-band yield per attempt at iteration 40, the furthest point every arm
reaches. The exact `--reg-picks` strings used are recorded in the LaTeX
comment above each figure. The summaries written beside each render
(`best_config_<metric>.json`, `*_heads.json`) hold the plotted numbers.

## Supplementary figure families

| files | script |
|---|---|
| `hit_rate_joint`, `hits_per_desired_joint`, `accuracy_joint`, `hit_rate_model_<model>`, `hits_per_desired_model_<model>`, `hit_rate_strategy_*` | `scripts/figures/plot_hit_rate_trajectories_multiseed.py` (the `_joint` names are the regeneration job's renames of `*_best_per_model`) |
| `hit_rate_seeds_<cell>`, `hits_per_desired_seeds_<cell>` | `scripts/figures/plot_hit_rate_seeds_per_model.py` |
| `iteration_metrics_seeds_<cell>` | `scripts/figures/plot_iteration_metrics_seeds.py` |
| `val_r2_*`, `static_r2_*`, `n_train_*` | `scripts/figures/plot_r2_trajectories_multiseed.py` |
| `mse_joint` | `scripts/figures/plot_mse_trajectories_multiseed.py` (the ratio strip is the ratio of seed means, not the mean of per-seed ratios) |
| `compute_joint`, `compute_vs_dataset_{warm,cold}`, `compute_per_iteration` | `scripts/figures/plot_compute_vs_dataset.py` |
| `coverage_saturation_pool`, `coverage_diversity_joint` | `scripts/analysis/coverage_saturation.py`, `scripts/figures/plot_diversity_vs_budget.py` |
| `support_efficiency_*` | `scripts/figures/plot_support_efficiency.py` (`--run-set-label`, `--band-side`, `--picks`; the `atlas`, `ext160` and `probe20k` variants come from the probe manifests) |
| `corner_<model>`, `corner_random`, `corner_mcmc`, `<param>_vs_ExpR_<model>`, `input_overlay_inband`, `omega_overlay`, `expr_overlay` | `scripts/figures/plot_al_input_target_diagnostics.py` |
| `pairwise_density_<cell>` | `scripts/figures/plot_pairwise_input_summary.py`; the paper-styled single cell is `plot_pairwise_paper.py` |
| `prelim_arms_{hit_rate,hits_per_desired,accuracy}_size`, `prelim_arms_support`, `prelim_arms_compute_support` | `scripts/figures/plot_prelim_paper_style.py`, `plot_prelim_support.py`, `plot_prelim_compute_support.py` (renamed by the regeneration job) |
| `mcmc_diagnostics_comparison`, `prior_predictive_marginals` | `scripts/analysis/mcmc_diagnostics.py`, `scripts/analysis/prior_predictive_check.py` |
| `omega_vs_M_2_lsp_type` | `scripts/figures/plot_omega_vs_m2_lsp.py` |
| `scaling_hitrate_vs_labelled` | `scripts/figures/plot_hitrate_vs_labelled.py` |
| `probe_extended_inputs`, `probe_20k_{inputs,hitrate,yield,compute}` | `scripts/experiments/plot_probe_input_overlays.py` and the main scripts run on the probe manifests through `scripts/campaign/run_probe_quality.sh` |
| `doc/transformer_architecture.png` | `scripts/figures/plot_transformer_architecture.py` |

## Tables

| label | source |
|---|---|
| `tab:yield`, `tab:expr-yield` | `scripts/analysis/compute_yield_comparison.py` (`yield_comparison.json`) |
| `tab:composition`, `tab:expr-composition` | `scripts/analysis/composition_fractions.py` |
| `tab:mcmc` | `scripts/analysis/mcmc_diagnostics.py` |
| `tab:uq` | `scripts/analysis/evaluate_uq.py` (`uq_evaluation.json`) |
| `tab:rank` | `scripts/analysis/rank_uniformity_al.py` |
| `tab:accuracy`, `tab:expr-accuracy`, `tab:support` | the JSON summaries of the hit-rate and support scripts |
| `tab:cost`, `tab:expr-cost` | `scripts/figures/plot_compute_vs_dataset.py`, `scripts/experiments/deepgp_validation_cost.py` |
| `tab:ensemble` | `scripts/experiments/ensemble_comparison.py` |
| `tab:funnel`, `tab:islands`, `tab:exclusion-confusion` | `scripts/analysis/funnel_analysis.py`, `find_islands.py`, `exclusion_confusion.py` |
| `tab:laplace`, `tab:laplace-comp`, `tab:laplace-diag` | `scripts/experiments/laplace_uq_probe.py`, `verify_laplace_gpc.py` |
| `tab:capacity` | `scripts/experiments/transformer_capacity_sweep.py` |
| `tab:extended`, `tab:comp-extended` | `scripts/campaign/run_probe_quality.sh` |
| `tab:lengthscales` | the GP runs' `lengthscales.csv` |
| `tab:models`, `tab:params`, `tab:configs`, `tab:software`, `tab:expr-analysis` | written by hand from `pmssm/config.py`, the per-model scripts and `pixi.lock` |

## Traps that have each produced a silently wrong number

- `random_baseline_prevalence.json` must record the neutralino veto and the
  validity fraction 0.4427, not the pool's raw 0.4454; an older file makes
  everything that divides by the random baseline about 0.6 % off.
- Probe runs (extended budget, 20k batch) were submitted through the
  per-model scripts and have no manifest row until `build_probe_manifest.py`
  runs; the 20k cells use `top_k`, so the default picks match nothing and the
  cells vanish from a figure without an error. Pass `--picks`.
- Never feed a base run and its continuation as two rows to
  `mcmc_diagnostics.py`: their chains overlap verbatim, R̂ compares a run with
  itself and the ESS inflates. The compute parser follows base to
  continuation chains itself.
- The probe output directories reuse the main figures' file names;
  `probe_20k/`, `probe_extended/` and `all_runs/` each hold their own
  `compute_vs_dataset.png`. The paper takes the main-body figure from
  `all_runs/` and the probe versions under renamed names, so never sync
  figures by the newest file of a given name across those directories.
- Plot-script warnings go to `.err`; a `[warn] baseline skip` there once
  meant every baseline curve was dropped from figures that rendered fine.
- The relic-density MCMC evaluation set has almost no target variance
  (Ω within the band by construction), so an MSE on it is prediction
  scatter, not accuracy of the shell; do not build claims on that panel.
- The regeneration job backs up the manuscript's figures before every cycle
  (dated directory plus a tarball on the archive root); do the same before
  overwriting by hand.
