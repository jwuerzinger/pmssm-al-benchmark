# pmssm-al-benchmark

Active learning for valid-model search in the electroweak pMSSM. This is the
code behind *A bitter lesson for active learning? A systematic benchmark in the
electroweak pMSSM* (J. Würzinger, 2026): the active-learning loop, the five
surrogate families and six acquisition rules it compares, the SLURM machinery
that ran the five-seed, 200-iteration campaign on two physics targets, and the
scripts that make every figure and table of the paper and its supplementary
material.

## What the code does

An active-learning run starts from a random pool of simulated pMSSM models,
trains a surrogate of one observable, scores a large candidate pool with the
surrogate's uncertainty, selects a batch, labels it with the physics chain
(SPheno, micrOMEGAs and, for the exclusion target, SModelS, driven through the
`Run3ModelGen` submodule), adds the labelled points to the training set and
repeats. A random-sampling baseline is trained alongside on the same budget.
Every iteration leaves a checkpoint, so runs resume and the campaign can be
chased to a fixed number of iterations across many 24 h jobs.

- **Surrogates** (one pipeline each under `pipelines/`): an MC-dropout
  transformer, an MC-dropout feed-forward network (and a width-matched
  variant), exact and deep Gaussian processes, and TabPFN used in context.
- **Targets** (`--target`): the dark-matter relic density `DMRD`
  (Ω h² = 0.12) and the LHC exclusion ratio `ExpR` (SModelS best expected r
  = 1). Both boundaries sit at zero in training space. [doc/TARGETS.md](doc/TARGETS.md).
- **Acquisition**: a tolerance cut and proximity weighting around the
  boundary, then `top_k`, `entropy_batch` (a diverse batch from the predictive
  covariance), `tol_only_random`, or the classification-head scores
  `cls_entropy` and `bald`; the surrogate's output can be a regression of
  log(Y / Y₀) or a classifier of the verdict. [doc/ACQUISITION.md](doc/ACQUISITION.md).
- **Inputs**: the ntuples carry 19 pMSSM parameters, of which 10 are fixed at
  the values of the ATLAS Run 2 electroweak scan and 9 are free: the slepton
  masses `meL`, `meR`, the gaugino and higgsino masses `M_1`, `M_2`, `mu`, the
  trilinears `At`, `Ab`, `Atau`, and `tanb`. Ranges live in `pmssm/config.py`.

## Layout

```
pipelines/   the four active-learning entry points (transformer, DNN, GP, TabPFN)
pmssm/       the library: data, models, heads, selection, uncertainty, training,
             evaluation, resume, iteration housekeeping, paths
scripts/     figures/ analysis/ campaign/ experiments/  (scripts/README.md lists each)
slurm/       al/ campaign/ analysis/ smoke/ experiments/ batch scripts (slurm/README.md)
tests/       package tests and the layout guard (tests/check_repo_layout.py)
doc/         the guides linked under Documentation below; doc/history/ holds old design records
Run3ModelGen/, al_pmssmwithgp/   submodules: the physics chain, the GP model code
```

Every script runs by path from any directory; the first statement of each
file puts the repository root on `sys.path`, and scripts import each other as
`scripts.<folder>.<module>`. No installation step is needed beyond the
environment.

## Installation

```bash
git clone --recurse-submodules git@github.com:jwuerzinger/pmssm-al-benchmark.git
cd pmssm-al-benchmark
pixi install -e rocm        # AMD MI300A; `-e cuda` for NVIDIA
```

`pixi.lock` pins the environment that produced the paper. The ROCm feature
installs PyTorch from the ROCm wheel index and resolves `pytorch-triton-rocm`
against the stub wheels in `stubs/`.

Then:

1. **Submodules, both required.** `al_pmssmwithgp` holds the GP models and
   the `entropy_batch` selection code; `Run3ModelGen` is the physics chain
   that labels every selected point (SPheno, micrOMEGAs, SModelS) and the
   source of all data. `Run3ModelGen` is hosted on CERN's GitLab
   (`ssh://git@gitlab.cern.ch:7999/jwuerzin/Run3ModelGen.git`), so fetching it
   needs a CERN account; build it following its own README
   (`cd Run3ModelGen && pixi run build`). The loop launches its `genModels.py`
   workers from that environment.
2. **Cluster configuration.** `cp slurm/cluster.conf.template slurm/cluster.conf`
   and edit it: partitions, GPU gres strings, the pixi environment name, the
   data, output, analysis and archive roots, the SModelS database directory
   and, for TabPFN, the licence token. The batch scripts and `pmssm/paths.py`
   read the same values, so this is the one place paths are set.
3. **Data.** The random pools, the emcee reference posterior and the SModelS
   database are not part of the repository (the tutorial's small sample is);
   [doc/DATA.md](doc/DATA.md) says what they are, where the code expects them
   and how they were made.

## Quick start

A short loop without the physics chain (selection only, no labelling):

```bash
pixi run -e rocm al-transformer -- --testing --n-iterations 3 --no-generate-data \
    --data-dir /path/to/pool --no-mcmc-eval --output-dir /tmp/al_smoke
```

`--testing` loads a small slice of the pool. The pixi tasks `al-transformer`,
`al-dnn`, `al-gp` and `al-tabpfn` wrap the four pipelines; `--help` on any of
them lists every option ([doc/CLI_REFERENCE.md](doc/CLI_REFERENCE.md) reproduces the output).

One production run on the cluster, with labelling:

```bash
source slurm/cluster.conf
sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_2}" \
       --export=ALL slurm/al/submit_al_gp_exact.sh
```

The per-model scripts in `slurm/al/` carry the paper's production settings
(2,000 initial models, 500 selections per iteration from a million
candidates, 40 iterations) and read overrides from `AL_*` environment
variables. A run writes one directory ([doc/RUN_OUTPUT_LAYOUT.md](doc/RUN_OUTPUT_LAYOUT.md)) and resumes
from its `state.pt`:

```bash
AL_RESUME_FROM=/path/to/run AL_N_ADDITIONAL_ITERATIONS=20 \
    sbatch ... --export=ALL slurm/al/submit_al_gp_exact.sh
```

The smoke jobs in `slurm/smoke/` exercise each pipeline end to end on the
development partition, including the physics chain.

## Tutorial

[doc/TUTORIAL.md](doc/TUTORIAL.md) walks through one complete loop without the physics chain:
a 12 MB data sample shipped with the repository, two runs in the oracle configuration (candidates drawn
from the reference and labelled by lookup), the run directory, a manifest and
the comparison figure that the paper's figures are built from.
`pixi run -e rocm tutorial` executes the same steps end to end, about forty
minutes on a CPU.

## The benchmark campaign

```bash
BUNDLE_SEEDS=1 bash slurm/al/submit_strategy_sweep.sh      # model x strategy x warm-start grid, 5 seeds
bash slurm/campaign/submit_campaign_200.sh                 # the acquisition arms, both targets
CAMPAIGN_CELLS=<cells csv> CAMPAIGN_HORIZON=<date> \
    sbatch --partition=apu1 --gres=gpu:1 --mem=16G --export=ALL slurm/campaign/submit_campaign_chase.sh
CAMPAIGN_ID=<id> REGEN_HORIZON=<date> \
    sbatch --partition=apu1 --gres=gpu:1 --mem=110000 --export=ALL slurm/campaign/submit_campaign_regen.sh
bash scripts/campaign/archive_runs.sh                      # tarball of every run's irreplaceable files
```

The sweep launcher writes one manifest row per run; the chase job wakes every
six hours and resumes whatever is short of 200 iterations; the regeneration
job re-renders every figure every three days. [doc/CAMPAIGN.md](doc/CAMPAIGN.md) explains each
piece and the traps that shaped it; [doc/STORAGE.md](doc/STORAGE.md) explains why the loop
packs each iteration's simulator workspaces and how finished runs are archived.

## Reproducing the paper

`slurm/campaign/submit_campaign_regen.sh` is the single entry point: it
refreshes the manifests, renders every figure family in the documented order
and copies the renders into the manuscript's figure directories. Run it once
with `COPY_TO_PAPER=0` to render without touching the paper.
[doc/REPRODUCING_THE_PAPER.md](doc/REPRODUCING_THE_PAPER.md) maps each figure and table to the script that
makes it, lists the inputs, and records the stale-cache traps that have
produced silently wrong numbers before.

## Documentation

| guide | contents |
|---|---|
| [doc/TUTORIAL.md](doc/TUTORIAL.md) | a complete loop on a small sample, with the plots to expect; also runnable as `pixi run tutorial` |
| [doc/ACQUISITION.md](doc/ACQUISITION.md) | candidate generation, the tolerance cut and proximity weighting, the selection strategies, the acquisition heads |
| [doc/TARGETS.md](doc/TARGETS.md) | the target registry (`DMRD`, `ExpR`), the SModelS prerequisite, the two definitions of the in-band support |
| [doc/CLI_REFERENCE.md](doc/CLI_REFERENCE.md) | the complete option list of the four pipelines, YAML sweeps, resuming |
| [doc/RUN_OUTPUT_LAYOUT.md](doc/RUN_OUTPUT_LAYOUT.md) | what a run directory contains and which file means what |
| [doc/SLURM.md](doc/SLURM.md) | the cluster contract, the per-model scripts and their variables, bundled multi-seed jobs, resuming |
| [doc/CAMPAIGN.md](doc/CAMPAIGN.md) | sweep manifests, the 200-iteration campaign, the chase loop, the regeneration job |
| [doc/REPRODUCING_THE_PAPER.md](doc/REPRODUCING_THE_PAPER.md) | figure and table to script map, regeneration order, known traps |
| [doc/DATA.md](doc/DATA.md) | the pools, the MCMC reference, the static evaluation set, caches |
| [doc/STORAGE.md](doc/STORAGE.md) | the purge, the inode limit, in-loop packing, archiving |
| [doc/PACKAGE_STRUCTURE.md](doc/PACKAGE_STRUCTURE.md) | the `pmssm` package module by module |
| [doc/IMPLEMENTATION_DETAILS.md](doc/IMPLEMENTATION_DETAILS.md) | log-space uncertainty, warm starting, proximity weighting, pipeline differences, GP jitter |
| [doc/ANALYSIS_METRICS.md](doc/ANALYSIS_METRICS.md) | the cross-run metrics of `scripts/analysis/analyse_runs.py` |
| [doc/LOGGING_INFO.md](doc/LOGGING_INFO.md) | where the logs are and what they contain |
| [scripts/README.md](scripts/README.md), [slurm/README.md](slurm/README.md) | one line per script |

## Not in this repository, and other caveats

- The data (pools, reference posterior, static evaluation sets, SModelS
  database) and the campaign's run directories live on the cluster's `/ptmp`
  and are not deposited with the code; only the tutorial's 12 MB sample
  (`data/tutorial/`) ships with it.
- `Run3ModelGen` is required and hosted on CERN's GitLab; readers without a
  CERN account cannot fetch it.
- The scripts under `scripts/experiments/` and `slurm/experiments/` back
  one-off studies in the supplementary material and still carry the
  cluster's literal paths; the production code reads every path from
  `pmssm/paths.py` and `slurm/cluster.conf`.
- `tests/check_repo_layout.py` (`pixi run check`) guards the layout rules; it
  runs locally only, there is no continuous integration.

## Citation and licence

Please cite the paper when using this code. MIT License, see [LICENSE](LICENSE).
