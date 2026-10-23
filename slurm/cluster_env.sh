# Sourced by the production batch scripts once REPO_ROOT is set:
#
#     source "${REPO_ROOT}/slurm/cluster_env.sh"
#
# Reads slurm/cluster.conf (gitignored; copy cluster.conf.template), fills in
# the defaults of the paper's campaign on Viper for anything it leaves unset,
# and exports the PMSSM_* names that pmssm/paths.py reads, so the shell and
# the Python side always agree on where data, runs and results live.
if [[ -f "${REPO_ROOT}/slurm/cluster.conf" ]]; then
    source "${REPO_ROOT}/slurm/cluster.conf"
else
    echo "[warn] slurm/cluster.conf not found; using the defaults in slurm/cluster_env.sh" >&2
    echo "       cp slurm/cluster.conf.template slurm/cluster.conf" >&2
fi
: "${CLUSTER_PARTITION:=apu}"
: "${CLUSTER_PARTITION_DEV:=apudev}"
: "${CLUSTER_GPU_GRES_2:=gpu:2}"
: "${CLUSTER_GPU_GRES_1:=gpu:1}"
: "${PIXI_ENV:=rocm}"
: "${PIXI_BIN:=${HOME}/.pixi/bin/pixi}"
: "${CLUSTER_DATA_DIR:=/ptmp/jwuerzin/data}"
: "${CLUSTER_OUTPUT_DIR:=/ptmp/jwuerzin/output}"
: "${CLUSTER_ANALYSIS_DIR:=/ptmp/jwuerzin/analysis}"
: "${CLUSTER_ARCHIVE_DIR:=/viper/u2/jwuerzin/pmssm-archive}"
: "${SMODELS_CACHEDIR:=/ptmp/jwuerzin/cache/smodels}"
: "${PAPER_DIR:=/viper/u2/jwuerzin/ALPaper}"
export CLUSTER_PARTITION CLUSTER_PARTITION_DEV CLUSTER_GPU_GRES_2 CLUSTER_GPU_GRES_1 PIXI_ENV PIXI_BIN
export CLUSTER_DATA_DIR CLUSTER_OUTPUT_DIR CLUSTER_ANALYSIS_DIR CLUSTER_ARCHIVE_DIR SMODELS_CACHEDIR PAPER_DIR
export PMSSM_DATA_ROOT="${CLUSTER_DATA_DIR}" PMSSM_OUTPUT_ROOT="${CLUSTER_OUTPUT_DIR}" \
       PMSSM_ANALYSIS_ROOT="${CLUSTER_ANALYSIS_DIR}" PMSSM_ARCHIVE_ROOT="${CLUSTER_ARCHIVE_DIR}" \
       PMSSM_PAPER_DIR="${PAPER_DIR}"
