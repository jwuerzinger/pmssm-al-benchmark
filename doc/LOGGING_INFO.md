# Logging

All pipelines log through `structlog` (`pmssm/logging_utils.py`), to the
console and to a file at once, one line per event with a timestamp and a
level.

| log | where | contents |
|---|---|---|
| run log | `<run dir>/active_learning.log` (or `logs/active_learning_<timestamp>.log` when a run has no output directory yet) | the full configuration, data ingest with the validity cuts and the resulting validity fraction, per-iteration selection and generation statistics, every metric the figures later read, the housekeeping summary |
| training workers | `<run dir>/iteration_NNN/al_training.log`, `baseline_training.log` | epoch-level training progress of the two models trained in parallel (`setup_worker_logging`) |
| SLURM | `logs/<job name>_<job id>.out`, `.err` | the batch script's own output; the per-model scripts rename the `.out` to a timestamped copy when they start and the epilogue writes to the original name, so a job leaves two `.out` files; Python warnings (`[warn] ...`) land in `.err` |
| manifests | `sweep_manifest.csv` `slurm_log` column | the SLURM log path of each run |

The validity-fraction line in the run log is provenance the figures depend
on: it is what converts hits per labelled point into hits per simulator call.
`scripts/campaign/archive_runs.sh` keeps the run log for that reason.
