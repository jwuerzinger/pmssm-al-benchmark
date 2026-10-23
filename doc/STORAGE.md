# Storage: the purge, the inode limit, packing and archiving

Two facts about the cluster's scratch filesystem shape how runs are stored.

- **`/ptmp` purges files unaccessed for about twelve weeks** and leaves the
  empty directory skeletons behind. It has eaten per-iteration checkpoint
  trails of finished runs before the figures were final.
- **`/ptmp` has a soft per-user limit of ten million inodes** (slightly above
  is tolerated). Each active-learning iteration leaves about 6,200 files of
  Run3ModelGen workspaces (`iteration_NNN/worker_*`, `retry_*`: SLHA spectra,
  SPheno logs, Snakemake metadata), so a 40-iteration run was 227,000 files
  and a 200-iteration campaign would have claimed hundreds of millions. The
  home filesystem is not purged but has a tight inode quota of its own, so
  raw copies of run trees are not an option anywhere.

## In-loop packing (`pmssm/iteration_housekeeping.py`)

After every iteration's `save_state`, each pipeline summarises and then packs
the iteration's workspaces (`--pack-debris`, default on; `--keep-debris`
disables it):

- the ntuples `ntuple.*.root` are moved to a flat `iteration_NNN/ntuples/`
  with the worker index in the file name, so `rglob("ntuple.*.root")` still
  finds them (`composition_fractions.py` reads them);
- for the exclusion target, the most sensitive SModelS analysis per point is
  extracted from the `*.slha.py` outputs into
  `iteration_NNN/smodels_best_analysis.json`, which `best_analysis_arms.py`,
  `best_analysis_from_smodels.py` and `mode_switch_diagnostic.py` read through
  `load_iteration_smodels()`;
- everything else goes into `iteration_NNN/debris.tar` and the trees are
  removed, about 6,200 inodes to one. Removal is gated on counting the files,
  writing the archive, reading it back and comparing member counts; any
  failure keeps the workspaces, which is the harmless direction.

A packed 200-iteration run is about 20,000 inodes. Nothing in the figure
pipeline needs the archive expanded; `tar -xf` restores an iteration in under
a second if ever needed.

## Packing older runs (`scripts/campaign/prune_run_debris.py`)

Runs that predate the in-loop packing are packed with the same code:

```bash
python scripts/campaign/prune_run_debris.py --runs "$PMSSM_OUTPUT_ROOT/active_learning_*"          # dry run, reports
python scripts/campaign/prune_run_debris.py --runs ... --keep-last 3 --mode tar --apply           # pack
python scripts/campaign/prune_run_debris.py --runs ... --keep-last 0 --mode tar --apply           # finished runs
```

`--keep-last N` leaves the N newest iteration directories untouched (the one
in progress is always the highest-numbered, and a killed iteration may be
retried against its workspace); use 0 only for runs that are finished. A
`debris.tar` left by an interrupted pass is recovered on the next pass: a
complete archive finishes the removal, a truncated one is set aside and the
iteration re-packed. `--max-dirs` and `--deadline-min` bound a pass. `--mode
delete` exists, is irreversible, and requires
`--i-know-this-deletes-analysis-inputs`. `slurm/campaign/submit_pack_debris.sh`
runs one chunk of run directories per job; on a saturated cluster the packer
also runs fine on a login node (it is I/O bound). `CHASE_PRUNE=1` on the chase
loop packs the finished iterations of the campaign cells at every wake.

## Archiving finished runs (`scripts/campaign/archive_runs.sh`)

One tarball per run on the non-purged archive root
(`CLUSTER_ARCHIVE_DIR/runs`), holding the curated, unregenerable subset:
`state.pt` (the cumulative training data), `active_learning.log` (provenance,
including the validity fraction line), `accuracy_trajectory.json`, and the
per-iteration `al_model_checkpoint.pt` and `baseline_model_checkpoint.pt`.
Simulation workspaces, GIFs and plots are excluded as regenerable. Idempotent:
a tarball is rewritten only when the run's `state.pt` is newer. The bundled
worker archives its own runs in its epilogue; run it by hand after any other
sweep:

```bash
bash scripts/campaign/archive_runs.sh [SRC=$PMSSM_OUTPUT_ROOT] [DEST=$PMSSM_ARCHIVE_ROOT/runs] [GLOB="*"]
```

The regeneration job backs up the manuscript's figure directories the same
way (dated copies plus a tarball) before overwriting anything.

## Caches

The figure scripts parse the pools once into `CLUSTER_ANALYSIS_DIR/pool_cache`
(`x_full_<pool>_<target>.npy`, `lspneut_full_*.npy`); without the cache each
step re-reads about 1,500 ROOT files from GPFS, half an hour per pass. Per-run
accuracy caches (`accuracy_trajectory.json`, `uq_eval_cache.json`) carry no
fingerprint of the reference set and go stale silently when the reference
changes: pass `--accuracy-cache-refresh` after a reference change.
