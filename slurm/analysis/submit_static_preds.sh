#!/bin/bash
# Per-point predictions on the static-random eval set for the paper picks
# (scripts/analysis/static_predictions_cache.py). One GPU, apu1 so it backfills.
#   TARGET=DMRD MODELS=transformer sbatch slurm/analysis/submit_static_preds.sh
#SBATCH --job-name=static_preds
#SBATCH --partition=apu1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "${REPO_ROOT}"; mkdir -p logs
source "${REPO_ROOT}/slurm/cluster_env.sh"   # TABPFN_TOKEN, PIXI_ENV
export TABPFN_TOKEN="${TABPFN_TOKEN:-}"
export PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
PYTHON="${REPO_ROOT}/.pixi/envs/${PIXI_ENV}/bin/python"
"${PYTHON}" scripts/analysis/static_predictions_cache.py --target "${TARGET:-DMRD}" \
    --model "${MODELS:-transformer}" --seeds "${SEEDS:-1,2,3,4,5}" --every "${EVERY:-10}" ${EXTRA:-}
