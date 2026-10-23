#!/bin/bash
# Pack the simulator debris of one batch of completed runs into debris.tar.
#
# One job per chunk file; the chunk is a list of run directories, one per
# line, built from the runs that are unpacked and not live. The packer itself
# still refuses the newest --keep-last iterations and anything touched inside
# --min-age-hours, so a run that has become live since the list was made is
# protected twice.
#
#   PACK_CHUNK=/path/chunk_0.txt sbatch --partition=apu1 --gres=gpu:1 --mem=16G \
#       slurm/campaign/submit_pack_debris.sh
#
#SBATCH --job-name=pack_debris
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=23:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -u
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"
: "${PACK_CHUNK:?set PACK_CHUNK to a chunk file}"
mapfile -t RUNS < "${PACK_CHUNK}"
echo "[pack] $(date): ${#RUNS[@]} run(s) from ${PACK_CHUNK}"
ARGS=(); for r in "${RUNS[@]}"; do ARGS+=(--runs "$r"); done
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.pixi/envs/rocm/bin/python scripts/campaign/prune_run_debris.py "${ARGS[@]}" \
    --apply --mode tar --keep-last 3 --min-age-hours 6 --deadline-min 1380
echo "[pack] $(date): exit $?"
