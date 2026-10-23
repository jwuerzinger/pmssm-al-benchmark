# SLURM: the cluster contract and the batch scripts

## The contract

- Submit from the repository root. Scripts resolve `REPO_ROOT` from
  `SLURM_SUBMIT_DIR` (falling back to their own location) and log to
  `logs/<job name>_<job id>.{out,err}` relative to it.
- `slurm/cluster.conf` (gitignored; copy `cluster.conf.template`) names the
  partitions, the GPU gres strings, the pixi environment, the pixi binary, the
  data, output, analysis and archive roots, the SModelS database and the
  TabPFN token. `slurm/cluster_env.sh` sources it, fills in the campaign's
  defaults for anything unset and exports the `PMSSM_*` names that
  `pmssm/paths.py` reads, so shell and Python agree.
- Pass `--export=ALL` whenever a script reads environment variables.
- Every production script exports `PYTHONPATH="$REPO_ROOT:$REPO_ROOT/al_pmssmwithgp/model"`,
  selects `$REPO_ROOT/.pixi/envs/$PIXI_ENV/bin/python`, installs the
  environment with `$PIXI_BIN` if it is missing, and detects the visible GPUs
  (`--gpu-ids 0,1` runs the active-learning and the baseline model on
  separate devices).
- SLURM copies a batch script at submission, but the script calls the
  repository by path when it starts: do not move files under queued jobs.

The partitions the campaign used, for orientation: `apu` (exclusive nodes
with two MI300A GPUs, multi-node allocations, 24 h limit; multi-node
`--gres=gpu:1` is rejected), `apu1` (shared single-node jobs that backfill
within minutes; the chase and regeneration jobs live here) and `apudev`
(15 minutes, for smoke tests). The association allowed eight running jobs,
single-node ones included, and 300 submissions.

## One run: `slurm/al/submit_al_<model>.sh`

Five scripts, one per surrogate (`transformer`, `dnn`, `gp_exact`, `gp_deep`,
`tabpfn`), each carrying the paper's production flags and reading overrides
from the environment:

| variable | default | passed as |
|---|---|---|
| `AL_N_SAMPLES` | 2000 | `--n-samples`, the initial labelled set |
| `AL_N_ITERATIONS` | 40 | `--n-iterations` |
| `AL_N_SELECT` | 500 | `--n-select` |
| `AL_N_CANDIDATES` | 1000000 | `--n-candidates` |
| `AL_EPOCHS`, `AL_PATIENCE` | 10000, 200 | `--epochs`, `--patience` (neural surrogates) |
| `AL_ENTROPY_POOL_SIZE` | 5000 | `--entropy-pool-size` |
| `AL_TOLERANCE_SAMPLING`, `AL_PROXIMITY_SAMPLING` | 1.0, 0.1 | the pre-filter |
| `AL_GEN_WORKERS` | 20 | parallel `genModels.py` workers |
| `AL_GENERATE_DATA` | `--generate-data` | set empty for selection without labelling |
| `AL_DATA_DIR`, `AL_MCMC_DATA_DIR` | the `DMRD` pool, the emcee reference | `--data-dir`, `--mcmc-data-dir` |
| `AL_STATIC_EVAL_SIZE` | 100000 | `--static-eval-size` |
| `AL_OUTPUT_DIR` | `$CLUSTER_OUTPUT_DIR/active_learning_<model>_output_<timestamp>` | `--output-dir` |
| `AL_SELECTION_STRATEGY` | `top_k` | TabPFN only |
| `AL_N_ENSEMBLE_SAMPLES` | 16 | TabPFN only |
| `AL_EXTRA_ARGS` | empty | appended last; overrides anything above (`--target ExpR`, `--head classification`, `--seed 3`, ...) |
| `AL_RESUME_FROM`, `AL_N_ADDITIONAL_ITERATIONS` | empty | resume (below) |

The GP scripts hardcode the kernel, ARD, jitter and 40 iterations; override
through `AL_EXTRA_ARGS`, which Click parses last. `AL_MODEL=dnn_match_trafo`
in the bundled worker dispatches to the DNN script with the width-matched
architecture flags. The TabPFN script refuses to start without `TABPFN_TOKEN`
(compute nodes cannot download the weights; the cache directory must already
hold them).

## Many seeds of one cell: `slurm/al/submit_al_bundled.sh`

Invoked by the sweep launcher in bundled mode, or by hand after exporting
`AL_MODEL`, `AL_STRATEGY`, `AL_WARM`, `AL_SEEDS`, `AL_OUTPUT_BASE` and
`AL_SWEEP_ID`. It allocates one node per seed, forks one `srun` per seed into
the per-model script above, and in its epilogue archives the bundle's runs.
`AL_RESUME_TO=<n>` makes it a resume round: each seed continues its own
directory to n iterations, seeds already there are skipped, and with
`AL_START_MISSING=1` a seed without `state.pt` is started fresh.

## Hyper-parameter arrays: `slurm/al/submit_sweep.sh`

An array job over a YAML config; each task runs one combination of the
Cartesian product (`--config-file` plus `--sweep-index $SLURM_ARRAY_TASK_ID`),
for any pipeline (`AL_SWEEP_SCRIPT`). `doc/CLI_REFERENCE.md` describes the
config format.

## Resuming

Every pipeline writes `state.pt` atomically after each iteration. A single
run continues with `--resume-from <run dir> --n-additional-iterations <n>`;
the per-model scripts build those flags from `AL_RESUME_FROM` and
`AL_N_ADDITIONAL_ITERATIONS` (`slurm/al/resume_args.sh`). The same output
directory is reused, so plots, GIFs and `iteration_NNN/` accumulate in place.
`--allow-config-drift` permits a resume that changes the loop's configuration.
A resume that changed the architecture must repeat the architecture flags:
the resume helper does not forward them.

## Logs and smoke tests

`logs/<name>_<id>.out` is renamed to a timestamped copy when a per-model job
starts, and the epilogue writes to the un-timestamped name, so a run leaves
two `.out` files; warnings from the Python side go to `.err`.
`slurm/smoke/test_al_<model>.sh` runs each pipeline for a few iterations on
the development partition, `test_al_target.sh` does so with generation for a
chosen target and driver, and `check_env.sh` and `discover_gpu.sh` report what
a compute node can see.
