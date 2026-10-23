#!/bin/bash
# The tutorial of doc/TUTORIAL.md, executable end to end:
#   two oracle-mode loops on the data sample, a manifest for them, and the
#   single-run and comparison figures. The two loops run concurrently: about
#   forty minutes on a CPU with four threads each, a few minutes on a GPU.
#
#   pixi run -e rocm tutorial            # or: bash scripts/tutorial/run_tutorial.sh
#
# Env: TUTORIAL_OUT         output directory      (default: <repo>/tutorial_runs)
#      TUTORIAL_DATA        the data sample       (default: <repo>/data/tutorial, fetched if missing)
#      TUTORIAL_ITERATIONS  iterations per run    (default: 5)
#      TUTORIAL_EPOCHS      training epochs       (default: 80)
#      PYTHON               interpreter           (default: python, i.e. the pixi environment)
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
mkdir -p "${TUTORIAL_OUT:-${REPO_ROOT}/tutorial_runs}"
PY="${PYTHON:-python}"
DATA="${TUTORIAL_DATA:-${REPO_ROOT}/data/tutorial}"
OUT="${TUTORIAL_OUT:-${REPO_ROOT}/tutorial_runs}"
N_ITER="${TUTORIAL_ITERATIONS:-5}"
EPOCHS="${TUTORIAL_EPOCHS:-80}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/al_pmssmwithgp/model:${PYTHONPATH:-}"
STAMP="$(date +%Y%m%d_%H%M%S)"
step () { echo; echo "########## $* ##########"; }

step "0/4 data"
TUTORIAL_DATA="${DATA}" bash scripts/tutorial/fetch_data.sh

# The loop's settings, scaled down from production (doc/ACQUISITION.md lists
# both). --candidate-source mcmc is the oracle configuration: candidates are
# drawn from the reference sample and their labels looked up, so no simulator
# runs. A run directory that ends in a timestamp is used as given.
COMMON=(--target DMRD --data-dir "${DATA}/pool" --mcmc-data-dir "${DATA}/mcmc"
        --candidate-source mcmc --mcmc-max-samples 100000
        --n-samples 1500 --static-eval-size 5000
        --n-iterations "${N_ITER}" --n-select 100 --n-candidates 5000 --entropy-pool-size 500
        --mc-samples 10 --epochs "${EPOCHS}" --patience 25 --no-warm-starting --seed 1 --gpu-ids 0)
RUN_A="${OUT}/active_learning_transformer_entropy_batch_cold_seed1_${STAMP}"
RUN_B="${OUT}/active_learning_transformer_top_k_cold_seed1_${STAMP}"

step "1/4 and 2/4: transformer with entropy_batch and with top_k, concurrently (${N_ITER} iterations each)"
"${PY}" pipelines/active_learning_transformer.py "${COMMON[@]}" \
    --selection-strategy entropy_batch --output-dir "${RUN_A}" > "${OUT}/run_a.log" 2>&1 &
PID_A=$!
"${PY}" pipelines/active_learning_transformer.py "${COMMON[@]}" \
    --selection-strategy top_k --output-dir "${RUN_B}" > "${OUT}/run_b.log" 2>&1 &
PID_B=$!
wait "${PID_A}" || { echo "[tutorial] the entropy_batch run failed; see ${OUT}/run_a.log" >&2; tail -20 "${OUT}/run_a.log"; exit 1; }
wait "${PID_B}" || { echo "[tutorial] the top_k run failed; see ${OUT}/run_b.log" >&2; tail -20 "${OUT}/run_b.log"; exit 1; }
for L in "${OUT}/run_a.log" "${OUT}/run_b.log"; do
    echo "--- $(basename "${L}")"; grep -E "Oracle mode|Iteration [0-9]+/|Summary:" "${L}" | tail -8
done

step "3/4 a manifest for the two runs"
"${PY}" scripts/campaign/build_probe_manifest.py --output-root "${OUT}" \
    --pattern "active_learning_*_${STAMP}" --out "${OUT}/manifest.csv" 2>&1 | tail -4
cat "${OUT}/manifest.csv"

step "4/4 figures"
mkdir -p "${OUT}/figures"
"${PY}" scripts/figures/plot_hit_rate_single_run.py --run-dir "${RUN_A}" --seed 1 \
    --model-label transformer --strategy-label entropy_batch --warm-label cold \
    --baseline-data-dir "${DATA}/pool" --cache-dir "${OUT}/cache" --output-dir "${OUT}/figures" 2>&1 | tail -3
"${PY}" scripts/figures/plot_hit_rate_trajectories_multiseed.py --manifest "${OUT}/manifest.csv" \
    --output-dir "${OUT}/figures" --baseline-data-dir "${DATA}/pool" --mcmc-data-dir "${DATA}/mcmc" \
    --pool-cache-dir "${OUT}/cache" --mcmc-yield-json "" --min-seeds 1 --min-seeds-axis 1 \
    --tolerances 0.10,0.20,0.50 2>&1 | tail -6

echo
echo "run directories:  ${RUN_A}"
echo "                  ${RUN_B}"
echo "figures:          ${OUT}/figures"
ls "${OUT}/figures"
