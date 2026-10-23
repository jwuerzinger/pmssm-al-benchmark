# slurm/

Batch scripts for a SLURM cluster. Submit every script **from the repository
root**: each one resolves the repository through `SLURM_SUBMIT_DIR`, and the
log paths (`logs/<job name>_<job id>.out`) are relative to it. Cluster-specific
settings (partition names, GPU gres strings, the pixi environment, the data
root, the TabPFN token) live in `slurm/cluster.conf`, which is gitignored:

```bash
cp slurm/cluster.conf.template slurm/cluster.conf   # then edit for your cluster
source slurm/cluster.conf
```

| folder | contents |
|---|---|
| `al/` | one submit script per surrogate (`submit_al_{transformer,dnn,gp_exact,gp_deep,tabpfn}.sh`), the multi-node bundled worker that runs several seeds of one cell (`submit_al_bundled.sh`), the sweep launchers (`submit_strategy_sweep.sh` for the model x strategy x warm-start grid, `submit_sweep.sh` for a YAML hyper-parameter array), and `resume_args.sh` |
| `campaign/` | the 200-iteration campaign: `submit_campaign_200.sh` launches the cells, `submit_campaign_chase.sh` is the self-resubmitting resume loop, `submit_campaign_regen.sh` re-renders every paper figure on a schedule, `submit_pack_debris.sh` packs finished runs' simulator workspaces |
| `analysis/` | GPU jobs behind individual figures and tables: the UQ evaluation, the accuracy plots, the arm support figures, the static prediction cache, the deep-GP post-hoc accuracy |
| `smoke/` | short end-to-end runs of each pipeline (`test_al_*.sh`, `test_al_target.sh` with data generation) and of single features; `check_env.sh` and `discover_gpu.sh` inspect a compute node |
| `experiments/` | the one-off studies reported in the supplementary material |

## Running the loop

**One run.** Every per-model script carries production defaults and reads
overrides from `AL_*` environment variables (`AL_N_ITERATIONS`, `AL_N_SELECT`,
`AL_OUTPUT_DIR`, `AL_DATA_DIR`, `AL_EXTRA_ARGS` for anything else, which is
spliced last so it wins):

```bash
sbatch --partition="${CLUSTER_PARTITION}" --gres="${CLUSTER_GPU_GRES_2}" \
       --export=ALL,AL_N_ITERATIONS=40 slurm/al/submit_al_gp_exact.sh
```

**A seeded grid.** `submit_strategy_sweep.sh` submits one job per
(model, strategy, warm start, seed) and appends a row per run to a manifest CSV
that every figure script reads. `BUNDLE_SEEDS=1` packs all seeds of a cell into
one multi-node job through `submit_al_bundled.sh`.

**The campaign.** `submit_campaign_200.sh` starts the campaign cells through the
sweep launcher; the chase job then wakes every six hours and resubmits whatever
has not reached its iteration target (`AL_RESUME_TO`), until a horizon date.
`doc/CAMPAIGN.md` walks through it.

## Resuming

A single run continues from its `state.pt` with `AL_RESUME_FROM=<run dir>` and
`AL_N_ADDITIONAL_ITERATIONS=<n>` on any per-model script (`resume_args.sh` turns
them into the `--resume-from` flags). The bundled worker takes
`AL_RESUME_TO=<n>` instead and derives each seed's remaining iterations itself,
so a resume round is idempotent.

## Three things that bite

- Pass `--export=ALL` whenever a script reads environment variables; a bare
  `--export=VAR=value` drops everything else, including `CAMPAIGN_CELLS`.
- `AL_SEEDS` must be exported in the shell, never given inline in `--export`:
  SLURM splits that list at its commas and the job silently runs one seed.
- SLURM copies a batch script when it is submitted, but the script calls the
  repository by path when it starts; do not move files under queued jobs.
