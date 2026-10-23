"""Where the data, runs and analysis products live.

Every location is an environment variable with the defaults of the paper's
campaign on the Viper cluster, so another installation overrides them once,
in ``slurm/cluster.conf`` (the batch scripts export the ``PMSSM_*`` names from
it through ``slurm/cluster_env.sh``), and nothing in the code is edited.

================  ===================================  =================================
variable          holds                                default
================  ===================================  =================================
PMSSM_DATA_ROOT   the pools and the MCMC reference     /ptmp/jwuerzin/data
PMSSM_OUTPUT_ROOT one directory per run                /ptmp/jwuerzin/output
PMSSM_ANALYSIS_ROOT manifests, caches and renders      /ptmp/jwuerzin/analysis
PMSSM_ARCHIVE_ROOT per-run tarballs on non-purged disk /viper/u2/jwuerzin/pmssm-archive
SMODELS_CACHEDIR  the SModelS database                 /ptmp/jwuerzin/cache/smodels
PMSSM_PAPER_DIR   the manuscript (figure directories)  /viper/u2/jwuerzin/ALPaper
================  ===================================  =================================

The pools are Run3ModelGen scan outputs (ROOT ntuples), one per target; the
MCMC reference is the emcee posterior of the relic density. ``doc/DATA.md``
describes them.
"""
import os
from pathlib import Path


def _env(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default))


DATA_ROOT = _env("PMSSM_DATA_ROOT", "/ptmp/jwuerzin/data")
OUTPUT_ROOT = _env("PMSSM_OUTPUT_ROOT", "/ptmp/jwuerzin/output")
ANALYSIS_ROOT = _env("PMSSM_ANALYSIS_ROOT", "/ptmp/jwuerzin/analysis")
ARCHIVE_ROOT = _env("PMSSM_ARCHIVE_ROOT", "/viper/u2/jwuerzin/pmssm-archive")
SMODELS_CACHE = _env("SMODELS_CACHEDIR", "/ptmp/jwuerzin/cache/smodels")
PAPER_DIR = _env("PMSSM_PAPER_DIR", "/viper/u2/jwuerzin/ALPaper")

#: the random-scan pool each target is trained and evaluated against
POOLS = {"DMRD": DATA_ROOT / "18387358", "ExpR": DATA_ROOT / "260804"}
#: the emcee posterior of the relic density (the only target with a reference)
MCMC_REFERENCE = DATA_ROOT / "neutralino_v4"
#: the analysis directory and sweep manifest of each target
ANALYSIS_DIRS = {"DMRD": ANALYSIS_ROOT / "all_runs", "ExpR": ANALYSIS_ROOT / "expr_runs"}
MANIFESTS = {t: d / "sweep_manifest.csv" for t, d in ANALYSIS_DIRS.items()}
#: the joint (most advanced run per cell and seed) manifests and composites
JOINT_DIR = ANALYSIS_ROOT / "joint"
#: the parsed-pool cache shared by every figure script
POOL_CACHE = ANALYSIS_ROOT / "pool_cache"
#: the campaign's cells CSVs
CAMPAIGN_DIR = ANALYSIS_ROOT / "campaign"
