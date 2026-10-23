"""Paper version of the pairwise-density figure for one (model, strategy, warm) cell.

Same data treatment as plot_pairwise_input_summary.py: for each parameter
pair, a density-normalised 2-D histogram on the min-max normalised scan range
(30 x 30 bins) per seed, then the mean and the standard deviation over seeds;
the emcee MCMC reference (seeded 500k subsample) on the same grid.

Presentation changes for the paper:
  * axes in physical units (TeV for masses, tan(beta) dimensionless), labelled
    in the notation of Table 1, shared across the column;
  * rows labelled "emcee MCMC", "<model> (<head>)" and "Seed-to-seed spread";
  * empty bins drawn in the lowest colour of the map instead of white, so the
    MCMC's unvisited regions do not read as missing data;
  * one shared logarithmic colour bar for the two density rows, one linear bar
    for the spread row.

Usage
-----
    python scripts/plot_pairwise_paper.py --model transformer --strategy cls_entropy \
        --warm cold --output .../pairwise_density_transformer_cls_entropy_cold.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import LogNorm, Normalize
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyse_runs import (FREE_PARAM_NAMES, FREE_PARAM_LO, FREE_PARAM_HI,  # noqa: E402
                          load_mcmc_numpy, load_run, normalise_free_params)
from plot_pairwise_input_summary import _pair_indices, _stack_hists, _mcmc_density  # noqa: E402
import plot_hit_rate_trajectories_multiseed as H                         # noqa: E402

# The six parameters of the distribution figure (no trilinears), same units (GeV).
PAIRS = [("M_1", "M_2"), ("M_1", "mu"), ("M_2", "mu"), ("meL", "meR"), ("M_1", "tanb"),
         ("mu", "tanb")]
LABEL = {"M_1": r"$M_1$", "M_2": r"$M_2$", "mu": r"$\mu$", "meL": r"$m_{\tilde e_L}$",
         "meR": r"$m_{\tilde e_R}$", "tanb": r"$\tan\beta$", "At": r"$A_t$"}
HEAD = {"cls_entropy": "classification", "bald": "classification"}


def _axis(name):
    """(lo, hi, label): GeV for masses, as in the distribution figure."""
    j = FREE_PARAM_NAMES.index(name)
    lo, hi = float(FREE_PARAM_LO[j]), float(FREE_PARAM_HI[j])
    if name == "tanb":
        return lo, hi, LABEL[name]
    return lo, hi, LABEL[name] + " [GeV]"


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest_paper.csv",
              show_default=True)
@click.option("--model", required=True)
@click.option("--strategy", required=True)
@click.option("--warm", required=True)
@click.option("--include-status", default="completed,running,timeout", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--n-bins", default=30, show_default=True)
@click.option("--reference", type=click.Choice(["mcmc", "pool-inband"]), default="mcmc",
              show_default=True, help="Top row: emcee chains, or the in-band random pool "
                                      "(targets without an MCMC reference).")
@click.option("--target", default="DMRD", show_default=True)
@click.option("--pool-dir", default="/ptmp/jwuerzin/data/260804", show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/expr_runs", show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--output", required=True)
def main(manifest, model, strategy, warm, include_status, mcmc_data_dir, mcmc_max_samples,
         n_bins, reference, target, pool_dir, cache_dir, tolerance, output):
    df = pd.read_csv(manifest)
    df = df[df["status"].isin({s.strip() for s in include_status.split(",") if s.strip()})]
    sub = df[(df["model"] == model) & (df["strategy"] == strategy) & (df["warm_start"] == warm)]
    runs_X = []
    for d in sub["expected_run_dir"].dropna():
        try:
            X = load_run(d).X_free_norm
        except Exception as exc:  # noqa: BLE001
            click.echo(f"[warn] skip {d}: {exc}", err=True)
            continue
        if X is not None and len(X):
            runs_X.append(np.asarray(X))
    if len(runs_X) < 2:
        raise click.ClickException(f"{model}/{strategy}/{warm}: {len(runs_X)} seeds")
    click.echo(f"[pairwise] {model}/{strategy}/{warm}: {len(runs_X)} seeds, "
               f"{sum(len(x) for x in runs_X):,} points")
    if reference == "mcmc":
        mX, _ = load_mcmc_numpy(mcmc_data_dir, max_samples=mcmc_max_samples or None)
        click.echo(f"[pairwise] MCMC: {len(mX):,} samples")
    else:
        from pmssm.config import TARGET_CONFIG  # noqa: PLC0415
        tv = float(TARGET_CONFIG[target]["true_value"])
        Xp, Yp = H._load_xy_full(pool_dir, target, Path(cache_dir))
        Yp = np.asarray(Yp).ravel()
        sel = np.isfinite(Yp) & (np.abs(Yp - tv) / tv < tolerance)
        mX = np.asarray(Xp)[sel]
        click.echo(f"[pairwise] in-band random pool: {len(mX):,} of {len(Yp):,} points")
    mX = normalise_free_params(np.asarray(mX, dtype=np.float64))

    mcmc_d, mean_d, std_d = [], [], []
    for a, b in PAIRS:
        idx = _pair_indices(a, b)
        Hs = _stack_hists(runs_X, idx, n_bins)
        mean_d.append(Hs.mean(axis=0))
        std_d.append(Hs.std(axis=0, ddof=1))
        mcmc_d.append(_mcmc_density(mX, idx, n_bins))
    pos = np.concatenate([np.concatenate([m.ravel(), d.ravel()]) for m, d in zip(mean_d, mcmc_d)])
    pos = pos[pos > 0]
    vmax = float(np.percentile(pos, 99))
    norm = LogNorm(vmin=vmax / 1e3, vmax=vmax)
    allstd = np.concatenate([s.ravel() for s in std_d])
    snorm = Normalize(0, float(np.percentile(allstd[allstd > 0], 99)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(cmap(0.0))
    cmap.set_under(cmap(0.0))
    smap = plt.get_cmap("magma").copy()

    n = len(PAIRS)
    # Explicit layout in inches: square panels, room for each column's y label
    # between panels, a left strip for the row titles, colour bars on the right.
    side, gap_w, gap_h = 1.9, 0.95, 0.12  # room for four-digit GeV tick labels
    left, right, bottom, top = 0.55, 1.15, 0.62, 0.08  # no row titles: the caption names the rows
    W = left + n * side + (n - 1) * gap_w + right
    Hh = bottom + 3 * side + 2 * gap_h + top
    fig = plt.figure(figsize=(W, Hh))
    axes = np.empty((3, n), dtype=object)
    for r in range(3):
        for c in range(n):
            x0 = (left + c * (side + gap_w)) / W
            y0 = (bottom + (2 - r) * (side + gap_h)) / Hh
            axes[r, c] = fig.add_axes([x0, y0, side / W, side / Hh])
    rows = [("emcee MCMC", mcmc_d, cmap, norm),
            (f"{H.MODEL_DISPLAY.get(model, model)}\n({HEAD.get(strategy, 'regression')})",
             mean_d, cmap, norm),
            ("Seed-to-seed\nspread", std_d, smap, snorm)]
    ims = []
    for r, (row_label, data, cm, nm) in enumerate(rows):
        for c, (a, b) in enumerate(PAIRS):
            ax = axes[r, c]
            xa, xb, xl = _axis(a)
            ya, yb, yl = _axis(b)
            z = np.ma.masked_less_equal(data[c].T, 0) if cm is cmap else data[c].T
            im = ax.imshow(z, origin="lower", extent=[xa, xb, ya, yb], aspect="auto",
                           cmap=cm, norm=nm, interpolation="nearest")
            if c == n - 1:
                ims.append(im)
            ax.tick_params(axis="both", labelsize=9.5, direction="out")
            # Tick steps of the distribution figure: 1000 GeV for M_1, M_2, mu,
            # 500 GeV for the slepton masses, default for tan(beta).
            step = {"M_1": 1000, "M_2": 1000, "mu": 1000, "meL": 500, "meR": 500}
            if a in step:
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step[a]))
            if b in step:
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(step[b]))
            if r == 2:
                ax.set_xlabel(xl, fontsize=12)
            else:
                ax.tick_params(axis="x", labelbottom=False)
            ax.set_ylabel(yl, fontsize=12, labelpad=1)
    top_p, mid_p, bot_p = (axes[i, -1].get_position() for i in range(3))
    cx = top_p.x1 + 0.15 / W
    cax1 = fig.add_axes([cx, mid_p.y0, 0.14 / W, top_p.y1 - mid_p.y0])
    cb1 = fig.colorbar(ims[0], cax=cax1)
    cb1.set_label("Density", fontsize=12)
    cax2 = fig.add_axes([cx, bot_p.y0, 0.14 / W, bot_p.height])
    cb2 = fig.colorbar(ims[2], cax=cax2)
    cb2.set_label("Std. dev. over seeds", fontsize=11)
    for cb in (cb1, cb2):
        cb.ax.tick_params(labelsize=9.5)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out}")


if __name__ == "__main__":
    main()
