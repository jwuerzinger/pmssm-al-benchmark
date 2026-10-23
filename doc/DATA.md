# Data

Nothing below ships with the repository. The code finds everything through
`pmssm/paths.py` (`PMSSM_DATA_ROOT`, the pools, the reference) and the batch
scripts through `slurm/cluster.conf`. All of it was produced with the
`Run3ModelGen` submodule, whose README documents the scan and the sampler.

## The random pools

| target | directory under the data root | content |
|---|---|---|
| `DMRD` | `18387358` | 300 ROOT ntuples (`ntuple.18387358.<n>.root`), 11 GB: a flat random scan of the nine free parameters through SPheno and micrOMEGAs; `MO_Omega` is the target branch |
| `ExpR` | `260804` | 1,500 ntuples, 13 GB: the same scan range with the SModelS step, so every point also carries `SModelS_bestExpR_r_expected` and the per-analysis r-values; `MO_Omega` is present too |

A pool plays three roles in a run: the initial labelled set is a slice of it
(`--n-samples`, with `--n-datasets` limiting how many files are read), the
static evaluation set is reserved from it (below), and the random-sampling
baseline draws its additions from what remains. Ingest applies the target's
validity cuts (`doc/TARGETS.md`); the fraction of simulated points that
survive them, p_valid, is what converts hits per labelled point into hits per
simulator call, and it is strongly range dependent (about 0.44 after the
neutralino veto on the relic-density pool, 0.58 on the exclusion pool).

## The MCMC reference (`neutralino_v4`)

Four emcee ensembles of 256 walkers each (`ntuple.<id>.{0..3}.root`, 688 MB,
with the sampler's `diag/` directory), targeting a Gaussian likelihood in the
relic density, Ω h² = 0.120 ± 0.012, with a neutralino LSP enforced at
sampling. Rows are stored iteration-major, walker-minor (row = iteration x
256 + walker). About 96 % of the 17.5 million rows are walker repeats, and
those repeats are the posterior weights: always subsample with
`--mcmc-max-samples` (500,000 by default), never de-duplicate. The chains are
formally unconverged (split-R̂ between 1.6 and 2.3) while the Ω marginal is the
targeted Gaussian; `scripts/analysis/mcmc_diagnostics.py --mcmc-nwalkers 256`
computes the diagnostics, `scripts/analysis/prior_predictive_check.py`
cross-checks the reference by importance-reweighting the random pool, and
`scripts/analysis/extract_emcee_chain_h5.py` reads the raw chains with burn-in.

Only the relic density has a reference (`has_mcmc_reference` in the target
registry); exclusion runs pass `--no-mcmc-eval`. The reference feeds the MCMC
evaluation set, the oracle candidate source (`--candidate-source mcmc`), the
MMD and coverage metrics, and the `mcmc` support definition.

## The static evaluation set

At the start of a run `--static-eval-size` models (100,000 in production)
are reserved from the pool with a seeded permutation and never used for
training; the accuracy, MSE and R² trajectories labelled `static_random` are
measured on them, and the per-point prediction cache
(`scripts/analysis/static_predictions_cache.py`) is built on them. The split
is reproducible from the run's seed, which is how post-hoc evaluations
recover it from `state.pt`.

## Caches and auxiliary inputs

| what | where | notes |
|---|---|---|
| parsed pools | `$PMSSM_ANALYSIS_ROOT/pool_cache/*.npy` | built once by the figure scripts (`--pool-cache-dir`); re-parsing 1,500 ROOT files takes half an hour |
| SModelS database | `$SMODELS_CACHEDIR` | `official<version>.pcl` and its descriptor file, both required |
| TabPFN weights | the TabPFN cache directory of the environment | downloaded once with a licence token on a node with internet access |
| sweep manifests, renders | `$PMSSM_ANALYSIS_ROOT/{all_runs,expr_runs,joint,campaign}` | `doc/CAMPAIGN.md` |
| run directories | `$PMSSM_OUTPUT_ROOT/active_learning_*` | `doc/RUN_OUTPUT_LAYOUT.md` |
| run archives | `$PMSSM_ARCHIVE_ROOT/runs/*.tar` | `doc/STORAGE.md` |

## Regenerating the data

A pool is a Run3ModelGen scan with the repository's parameter ranges
(`pmssm/config.py`, identical to the ATLAS Run 2 electroweak pMSSM scan) and
the step list of the target; the reference is Run3ModelGen's emcee sampler
with the Gaussian relic-density likelihood. The directory names are the SLURM
job ids of the original scans and carry no other meaning.
