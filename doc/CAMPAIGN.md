# The campaign: sweeps, manifests, chase and regeneration

Four pieces turn single runs into the paper's five-seed benchmark, all under
`slurm/al/` and `slurm/campaign/` with their Python under `scripts/campaign/`.

## 1. The sweep launcher: `slurm/al/submit_strategy_sweep.sh`

Submits the grid of (model x strategy x warm start x seed) and appends one row
per run to a manifest CSV. Environment overrides:

| variable | default | meaning |
|---|---|---|
| `MODELS` | `transformer,dnn,deep_gp,exact_gp,tabpfn` | `dnn_match_trafo` adds the width-matched DNN (`--d-model 64 --num-layers 3 --dim-feedforward 400`) |
| `STRATEGIES` | `top_k,top_k_tol_only,entropy_batch` | TabPFN skips `entropy_batch` unless `TABPFN_ALLOW_ENTROPY=1` |
| `WARM_MODES` | `warm,cold` | TabPFN has no warm-start axis and uses the sentinel `tabpfn` |
| `SEEDS` | `1,2,3,4,5` | |
| `MANIFEST` | `$CLUSTER_ANALYSIS_DIR/all_runs/sweep_manifest.csv` | keep one manifest per target |
| `OUTPUT_TAG` | empty | suffixes the manifest model and the run directory (`transformer_expr`) without touching the dispatch key |
| `EXTRA_AL_ARGS` | empty | appended last to every run, so it wins Click's last-occurrence parsing (`--target ExpR --data-dir ... --no-mcmc-eval`) |
| `BUNDLE_SEEDS` | `0` | `1` submits one multi-node job per cell through `submit_al_bundled.sh`, one node per seed |
| `DRY_RUN` | `0` | preview and append `DRY` rows |

Run directories are named
`active_learning_{model}{_tag}_{strategy}_{warm}_seed{N}_{sweep id}` under
`CLUSTER_OUTPUT_DIR`; the driver appends the sweep id only when the given
`--output-dir` does not already end with a timestamp.

**Bundled mode.** One `sbatch` per cell allocates one two-GPU node per seed
and forks one `srun` per seed; the manifest still gets one row per seed with a
shared `job_id`. It exists because the association allows eight running jobs
and single-node jobs count against the same cap. TabPFN bundles also request
`gpu:2` because multi-node `gpu:1` is rejected on the `apu` partition.
`AL_SEEDS` must be exported in the shell: given inline in `--export`, SLURM
splits it at the commas and the job silently runs one seed (the worker aborts
when the seed count and the node count disagree).

**Manifest columns.** `sweep_id`, `submit_time`, `model`, `strategy`,
`warm_start`, `seed`, `job_id`, `expected_run_dir`, `status`, `slurm_log`.
`scripts/campaign/update_sweep_manifest.py` refreshes `status` from `sacct` and
the run directory: `completed` only when `summary.json` exists (the driver
writes it once, at the very end), `running` when only `state.pt` does, a
terminal SLURM state (`failed`, `timeout`, `out_of_memory`, ...) otherwise, and
`missing` when SLURM says `COMPLETED` but no `summary.json` appeared. Every
figure script filters on `status`, so a run that reached its iteration target
but whose job later exited non-zero is dropped from the figures until its row
is corrected. `--supersede-before <sweep id>` marks older generations
`superseded` so default filters pick one generation.

`scripts/campaign/build_joint_manifest.py` writes the manifest that selects,
per (cell, seed), the most advanced run directory, which is what the arm
figures read; `build_probe_manifest.py` writes a manifest for runs that were
started through the per-model scripts and never had a row.

## 2. The 200-iteration campaign: `slurm/campaign/submit_campaign_200.sh`

Starts every acquisition arm at five seeds on both targets through the sweep
launcher in bundled mode: `tol_only_random` on six models, `bald` and
`cls_entropy` on the five models with a classification head (TabPFN has none).
Warm mode is fixed per model, cold for the neural surrogates and warm for the
GPs. `entropy_batch` is not here; its continuations were queued separately.
The script writes the cells CSV (`CLUSTER_ANALYSIS_DIR/campaign/<id>_cells.csv`)
that the chase reads; `DRY_RUN=1`, `ARMS=...`, `TARGETS=...` restrict it.

## 3. The chase: `slurm/campaign/submit_campaign_chase.sh`

No fresh run reaches 200 iterations inside the 24 h wall clock, so every cell
is resumed several times. `scripts/campaign/campaign_chase.py` reads the cells
CSV and both sweep manifests, finds every cell whose longest seed is past 40
iterations but short of the target, skips cells with a queued or running job
(two jobs resuming one directory would corrupt `state.pt`; the `squeue` check
on the `c200_` job-name prefix is the only guard), and submits
`submit_al_bundled.sh` with `AL_RESUME_TO=200`, which derives each seed's
remaining iterations from its own `state.pt` and is therefore idempotent.
`AL_START_MISSING=1` starts a seed that has no `state.pt` yet.

The batch script chains its successor six hours ahead **before** doing any
work, so a crash cannot end the loop silently, and stops after
`CAMPAIGN_HORIZON`. Variables: `CAMPAIGN_CELLS` (required), `CAMPAIGN_HORIZON`,
`CHASE_RESUME_TO` (200), `CHASE_PER_WAKE` (12), `CHASE_QUEUE_CAP` (250),
`CHASE_EVERY` (6hours), `DRY_RUN=1`, and `CHASE_PRUNE=1` to pack finished
iterations' workspaces at every wake (`doc/STORAGE.md`). TabPFN resumes on the
exclusion target request 200 GB (SModelS inside the generation workers pushed
every 64 GB request to `OUT_OF_MEMORY`). Always submit with `--export=ALL`: a
bare `--export=VAR=value` drops `CAMPAIGN_CELLS` and the loop runs for a day
doing nothing.

Frozen 40-iteration copies of some benchmark runs live outside
`CLUSTER_OUTPUT_DIR` (`output_benchmark40/`); the chase never resumes a
manifest row that points outside the live output root.

## 4. The regeneration job: `slurm/campaign/submit_campaign_regen.sh`

Re-renders every figure every three days and copies them into the manuscript
(`PAPER_DIR`), self-resubmitting until `REGEN_HORIZON`. Steps, in order:

0. date-stamped backup of the paper's figure directories and a tarball on the
   archive root;
1. refresh both sweep manifests, build the joint manifests;
1b. write `sweep_manifest_benchmark.csv` per target with the campaign rows
   (`c200_` sweep id) and the `*_laplace` variant rows held back, because the
   campaign writes into the same manifests the benchmark figures read and
   the new arms would otherwise enter figures whose captions describe five
   strategies;
2. per target, hit rate, hits per desired and verdict accuracy
   (`plot_hit_rate_trajectories_multiseed.py --compute-accuracy`, the one
   GPU step), which also writes the random baseline the later steps divide by;
3. per target, the remaining families: per-seed curves, MCMC diagnostics, R²,
   MSE, compute, coverage, support efficiency, diversity, input diagnostics,
   pairwise densities, and unless `SKIP_SLOW=1` the yield table and the UQ
   evaluation;
4. the arm figures with seed bands from the joint manifests;
5. copy the renders into the paper, overwriting only basenames the paper
   already carries, choosing for each the newest render across the search
   roots (an older copy in a subdirectory once shadowed fresh renders);
6. rebuild both PDFs.

`COPY_TO_PAPER=0` renders without touching the manuscript. `CAMPAIGN_ID` names
the arm sweep. `doc/REPRODUCING_THE_PAPER.md` has the figure map and the
traps.

## 5. Archiving: `scripts/campaign/archive_runs.sh`

The bundled worker's epilogue, and a manual run after a sweep, tar each run's
irreplaceable files to the archive root (`doc/STORAGE.md`). Arguments: source
directory, destination, run-name glob.

## The former per-model script variants

Seven near-duplicate submit scripts were removed; they differed from the base
scripts only in the following, all reproducible through the environment:

| former script | equivalent |
|---|---|
| `submit_al_{transformer,dnn,gp_exact,gp_deep}_top_k.sh` | `AL_EXTRA_ARGS="--selection-strategy top_k"` on the base script |
| `submit_al_{transformer,dnn}_top_k_20k.sh` | the same plus `AL_N_SELECT=20000` (the 20k probes of the supplementary) |
| `submit_al_tabpfn_entropy.sh` | `AL_SELECTION_STRATEGY=entropy_batch slurm/al/submit_al_tabpfn.sh` |
