# Targets

The loop learns one observable at a time, chosen with `--target` on every
pipeline and defined once in `TARGET_CONFIG` (`pmssm/config.py`). Two targets
are in production use; `CrossSection` and `CLs` exist in the registry as
examples and have no pool.

| `--target` | observable | ROOT branch | boundary | pool (`pmssm/paths.py`) |
|---|---|---|---|---|
| `DMRD` (default) | dark-matter relic density Ω h² | `MO_Omega` | 0.12 | `POOLS["DMRD"]` |
| `ExpR` | SModelS best expected r-value (LHC exclusion) | `SModelS_bestExpR_r_expected` | 1.0 | `POOLS["ExpR"]` |

Both boundaries sit at 0 in training space because `transform_y` computes
log(Y / Y₀), so the acquisition machinery (`--tolerance-sampling 1.0`,
`--proximity-sampling 0.1`) and the hit-band metric |Y − Y₀| / Y₀ < tol carry
over unchanged. `--target-value` defaults to the registry value.
`--y-transform zscore` is rejected for non-`DMRD` targets: its threshold comes
from raw-Y statistics, meaningless for a quantity spanning nine decades.

What a registry entry controls, beyond the branch and the boundary:

- **`valid_max`**, the ingest upper cut. `DMRD` keeps Y < 1 (sub-dominant dark
  matter); `ExpR` has none, because r > 1 is the excluded region and the whole
  point. Run3ModelGen writes `-1.` when a quantity was not computed, so Y > 0
  is exactly "we have a value".
- **`gen_require_neutralino_lsp`**, whether newly generated points without a
  neutralino LSP are vetoed. True for `DMRD` (a relic density needs a
  dark-matter candidate), false for `ExpR` (a collider limit does not care).
  Pool ingest never applies the veto by default.
- **`gen_steps`**, the Run3ModelGen steps that label a selected candidate.
  `ExpR` adds `SModelS`, kept last among the SPheno consumers, and retains
  `micromegas` so generated points still carry `MO_Omega`.
- **`has_mcmc_reference`**, true only for `DMRD`: `load_mcmc_data` and
  `--use-mcmc-loader` refuse other targets. Use `--no-mcmc-eval` to run
  without a reference set; it is the only way to override the
  `--mcmc-data-dir` the per-model scripts always pass.

## The SModelS prerequisite

The `ExpR` step list needs the SModelS results database, and compute nodes
have no internet. `Run3ModelGen/build/setup.sh` points `SMODELS_CACHEDIR` at
the checkout when `/eos` is absent, which holds no database, so every
production batch script exports `SMODELS_CACHEDIR` (from `slurm/cluster.conf`,
through `slurm/cluster_env.sh`), which `setup.sh` honours. Both files must be
present there: the `official<version>.pcl` pickle and its
`https___..._database_<version>` descriptor; with only the pickle SModelS still
tries to reach the server and fails. `pmssm/model_generation.py` checks for
both before spawning any worker, so a missing database is one clear error
instead of twenty stack traces.

## Analysing a non-default target

Every analysis script that reads the pool or the reference takes the target
explicitly, because a hardcoded key silently substitutes relic-density data
and the figure still renders. `_load_y_full` and `_load_xy_full` refuse a pool
that does not carry the requested branch.

| flag | scripts | purpose |
|---|---|---|
| `--target` | `compute_yield_comparison`, `funnel_analysis`, `plot_compute_vs_dataset`, `plot_al_input_target_diagnostics`, `plot_omega_vs_m2_lsp`, `coverage_saturation`, `plot_support_efficiency` | registry key: branch, band centre, validity cuts |
| `--target-name` | `composition_fractions` | the same, where `--target` is already a numeric band centre |
| `--model-tag` | `plot_compute_vs_dataset`, `plot_mse_trajectories_multiseed`, `plot_al_input_target_diagnostics`, `coverage_saturation`, `evaluate_uq` | resolves `OUTPUT_TAG`'d manifest rows (`transformer_expr`) against the per-model picks |
| `--support-source mcmc\|pool` | `coverage_saturation` | the population defining the in-band support |
| `--mcmc-yield-json ""` | `plot_hit_rate_trajectories_multiseed` | suppresses the relic-density MCMC yield reference lines |

A representative `ExpR` invocation:

```bash
python scripts/figures/plot_hit_rate_trajectories_multiseed.py \
    --manifest "$PMSSM_ANALYSIS_ROOT/expr_runs/sweep_manifest.csv" \
    --output-dir "$PMSSM_ANALYSIS_ROOT/expr_runs" \
    --target ExpR --baseline-data-dir "$PMSSM_DATA_ROOT/260804" \
    --no-baseline-require-neutralino-lsp --mcmc-yield-json ""
```

`--baseline-require-neutralino-lsp` defaults on; turn it off for `ExpR`, whose
population deliberately keeps non-neutralino LSPs.

## The in-band support and its two sources

`coverage_saturation.build_target` defines the support by binning a reference
population's in-band half on a 12-bin equal-occupancy quantile grid in
(`M_1`, `M_2`, `mu`) and keeping cells with at least `--min-cell` points (20).
`coverage_of` scores a run by the fraction of those cells it reaches with
in-band points. `--support-source` selects the reference: `mcmc` (the emcee
posterior, the paper's original definition) or `pool` (the random scan). The
relic density can use both; a target with no posterior must use `pool`.

The supports differ greatly in resolution, so a coverage value compares
models within a panel and not across panels: a flat scan puts 0.9 % of its
points in the relic band, giving 27 support cells, against 942 from the
posterior and 639 from the r-value pool. Ranked by mutual information with
band membership, `M_2`, `M_1` and `mu` are the right axes for both targets,
with a 4.5x gap to the next parameter; no single parameter determines band
membership, which is why the support is joint and why per-dimension marginal
coverage saturates at 1.

## Smoke-testing a target end to end

The SModelS path cannot be verified offline:

```bash
source slurm/cluster.conf
AL_TARGET=ExpR AL_DRIVER=transformer \
  sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_1}" \
         --export=ALL slurm/smoke/test_al_target.sh
```

It runs a short loop with generation and prints the per-iteration generation
cost, which decides whether 40 iterations fit the 24 h limit.
