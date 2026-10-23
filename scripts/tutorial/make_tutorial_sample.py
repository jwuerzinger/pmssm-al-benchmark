"""Write the tutorial's small data sample from the full pools.

Run once, on the cluster, where the pools and the reference live
(``pmssm/paths.py``). It writes two small ROOT files with the ``susy`` tree and
the branches the loaders read, so every pipeline and figure script works on
them unchanged:

    data/tutorial/pool/ntuple.tutorial.0.root   a uniform sample of the relic-density pool
    data/tutorial/mcmc/ntuple.tutorial.0.root   a uniform sample of the emcee reference
    data/tutorial/README.txt                    provenance: sources, seed, counts

and optionally a tarball for distribution (``--tar``). The reference sample is
drawn uniformly over the whole chain, repeats included, because the repeats are
the posterior weights (``doc/DATA.md``).

    python scripts/tutorial/make_tutorial_sample.py --tar /path/to/pmssm-al-tutorial-data-v1.tar.gz
"""
# --- repository bootstrap: lets this file run from any directory --------------
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ------------------------------------------------------------------------------

import datetime
import glob
import hashlib
import subprocess
import tarfile

import click
import numpy as np

from pmssm import paths
from pmssm.config import PARAM_ORDER

EXTRA = ["MO_Omega", "SP_m_h", "SP_LSP_type", "SP_LSP_Bino_frac", "SP_LSP_Wino_frac",
         "SP_LSP_Higgsino_frac", "MO_cdm_is_neutralino"]


def _sample_files(files, n_rows, rng, chunks_per_file=4, log=print):
    """Uniform row sample across the given files, in a bounded number of reads."""
    import uproot
    branches = None
    parts = []
    total = 0
    for f in files:
        t = uproot.open(f)["susy"]
        if branches is None:
            branches = list(PARAM_ORDER) + [b for b in EXTRA if b in t.keys()]
        total += t.num_entries
    frac = min(1.0, 1.25 * n_rows / total)           # oversample a little, trim below
    for f in files:
        t = uproot.open(f)["susy"]
        n = t.num_entries
        edges = np.linspace(0, n, chunks_per_file + 1, dtype=int)
        for a, b in zip(edges[:-1], edges[1:]):
            arr = t.arrays(branches, entry_start=int(a), entry_stop=int(b), library="np")
            keep = rng.random(int(b - a)) < frac
            parts.append({k: v[keep] for k, v in arr.items()})
        log(f"  read {f} ({n} rows)")
    cat = {k: np.concatenate([p[k] for p in parts]) for k in branches}
    m = len(cat[branches[0]])
    take = rng.permutation(m)[:n_rows]
    take.sort()
    return {k: v[take] for k, v in cat.items()}, total


def _write(path: Path, arrays: dict):
    import uproot
    path.parent.mkdir(parents=True, exist_ok=True)
    with uproot.recreate(path) as f:
        f["susy"] = arrays


@click.command()
@click.option("--pool", default=str(paths.POOLS["DMRD"]), show_default=True, help="The relic-density pool directory.")
@click.option("--mcmc", default=str(paths.MCMC_REFERENCE), show_default=True, help="The emcee reference directory.")
@click.option("--out", default=str(_REPO_ROOT / "data" / "tutorial"), show_default=True)
@click.option("--n-pool", default=50_000, show_default=True)
@click.option("--n-mcmc", default=100_000, show_default=True)
@click.option("--max-pool-files", default=40, show_default=True, help="Pool files to draw from (the pool is uniform, so a prefix is a fair sample).")
@click.option("--seed", default=20261023, show_default=True)
@click.option("--tar", default=None, help="Also write this tarball (and its .sha256) for distribution.")
def main(pool, mcmc, out, n_pool, n_mcmc, max_pool_files, seed, tar):
    rng = np.random.default_rng(seed)
    out = Path(out)
    pool_files = sorted(glob.glob(f"{pool}/*.root"))[:max_pool_files]
    mcmc_files = sorted(glob.glob(f"{mcmc}/*.root"))
    if not pool_files or not mcmc_files:
        raise click.ClickException(f"no ROOT files under {pool} or {mcmc}")
    click.echo(f"[pool] sampling {n_pool} rows from {len(pool_files)} files")
    pool_arr, pool_total = _sample_files(pool_files, n_pool, rng, log=click.echo)
    _write(out / "pool" / "ntuple.tutorial.0.root", pool_arr)
    click.echo(f"[mcmc] sampling {n_mcmc} rows from {len(mcmc_files)} files")
    mcmc_arr, mcmc_total = _sample_files(mcmc_files, n_mcmc, rng, chunks_per_file=8, log=click.echo)
    _write(out / "mcmc" / "ntuple.tutorial.0.root", mcmc_arr)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_REPO_ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = "unknown"
    readme = f"""pmssm-al-benchmark tutorial data sample
=======================================
Written {datetime.date.today().isoformat()} by scripts/tutorial/make_tutorial_sample.py
(repository commit {commit}, seed {seed}).

pool/ntuple.tutorial.0.root   {len(pool_arr['MO_Omega'])} rows drawn uniformly from the first
                              {len(pool_files)} files ({pool_total} rows) of the relic-density
                              pool {pool}
mcmc/ntuple.tutorial.0.root   {len(mcmc_arr['MO_Omega'])} rows drawn uniformly over the whole
                              emcee reference {mcmc} ({mcmc_total} rows; walker repeats kept,
                              they are the posterior weights)

Both hold the `susy` tree with the 19 input parameters (IN_*), MO_Omega, SP_m_h,
SP_LSP_type, the LSP composition fractions and MO_cdm_is_neutralino where the
source carries it. They exist for doc/TUTORIAL.md and are far too small for
any physics conclusion.
"""
    (out / "README.txt").write_text(readme)
    click.echo(readme)
    if tar:
        tar = Path(tar)
        tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar, "w:gz") as tf:
            tf.add(out, arcname="tutorial")
        digest = hashlib.sha256(tar.read_bytes()).hexdigest()
        (tar.with_suffix(tar.suffix + ".sha256")).write_text(f"{digest}  {tar.name}\n")
        click.echo(f"[tar] {tar} ({tar.stat().st_size / 1e6:.1f} MB) sha256 {digest}")


if __name__ == "__main__":
    main()
