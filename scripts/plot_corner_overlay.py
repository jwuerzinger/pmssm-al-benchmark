"""Corner version of the holistic comparison (distribution_overlay_best + pairwise projections).

  * diagonal: the panels of distribution_overlay_best.png, unchanged in content:
    in-band, density-normalised marginals of the six parameters for every
    surrogate's best cell, the random pool and the emcee MCMC; the target
    marginal (every point, log10 axis) in the last diagonal panel;
  * lower triangle: 2-D projections of the reference and one surrogate
    (``--pair-model``) as 39% and 86% highest-density contours of a lightly
    smoothed 2-D histogram, in the colours of the diagonal. Parameter pairs use
    the in-band points, as the diagonal does; the target row uses every point.
    The reference is the emcee MCMC, or the random pool for targets without an
    MCMC (exclusion boundary, ``--mcmc-data-dir ""``), drawn in black;
  * upper triangle: the seed-to-seed standard deviation of the ``--pair-model``
    2-D density (the bottom row of the pairwise-projection figure), with the
    axes of the mirrored lower panel (same x and y variables),
    on the same points as the lower triangle; densities are taken on axes
    scaled to the panel range (unit square), so one colour bar serves all
    panels whatever their units;
  * legend above the grid.

Sample loading is that of plot_distribution_overlay_rows.py (--best-picks).

Usage
-----
    python scripts/plot_corner_overlay.py --manifest .../sweep_manifest_paper.csv \
        --best-picks transformer:cls_entropy:cold,... --output .../corner_overlay_best.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_hit_rate_trajectories_multiseed as H                         # noqa: E402
from plot_hits_per_desired_heads import parse_picks                       # noqa: E402
from plot_al_input_target_diagnostics import _pooled_cell_data, _arm_color  # noqa: E402
from plot_distribution_overlay import PRETTY                               # noqa: E402
from plot_distribution_overlay_rows import SHOWN, MODEL_ORDER, _band       # noqa: E402
from analyse_runs import FREE_PARAM_NAMES, FREE_PARAM_INDICES              # noqa: E402
from pmssm.config import TARGET_CONFIG                                     # noqa: E402

LEVELS = (0.39, 0.86)  # 1 and 2 sigma of a 2-D Gaussian
STEP = {"M_1": 1000, "M_2": 1000, "mu": 1000, "meL": 500, "meR": 500}


def _hpd_levels(h, fracs):
    """Density thresholds enclosing the given probability fractions of h."""
    flat = np.sort(h.ravel())[::-1]
    cum = np.cumsum(flat)
    cum /= cum[-1]
    return [flat[min(np.searchsorted(cum, f), len(flat) - 1)] for f in fracs]


def _contour(ax, x, y, xr, yr, color, n_bins, smooth):
    ok = np.isfinite(x) & np.isfinite(y)
    h, xe, ye = np.histogram2d(x[ok], y[ok], bins=n_bins, range=[xr, yr])
    h = gaussian_filter(h, smooth)
    lv = sorted(set(_hpd_levels(h, LEVELS)))
    xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
    # Levels ascending: the 95% contour first (thin), then the 68% contour.
    ax.contour(xc, yc, h.T, levels=lv, colors=color,
               linewidths=[1.0, 1.6][-len(lv):], linestyles=["--", "-"][-len(lv):])


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest_paper.csv",
              show_default=True)
@click.option("--best-picks", required=True, help="model:strategy:warm,... (one per model).")
@click.option("--pair-model", default="transformer", show_default=True,
              help="Surrogate drawn against the MCMC in the lower triangle.")
@click.option("--model-tag", default="", show_default=True)
@click.option("--include-status", default="completed,timeout,running,missing", show_default=True)
@click.option("--pool-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--n-bins", default=40, show_default=True)
@click.option("--smooth", default=1.5, show_default=True, help="Gaussian smoothing, in bins.")
@click.option("--std-bins", default=30, show_default=True,
              help="Bins per axis of the seed-spread panels (30 as in the pairwise figure).")
@click.option("--levels", default="0.39,0.86", show_default=True,
              help="Two enclosed probability fractions, inner first.")
@click.option("--all-points", is_flag=True, default=False,
              help="Parameter marginals and pairs from every valid point, not only the in-band ones "
                   "(the target panels already use every point).")
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(manifest, best_picks, pair_model, model_tag, include_status, pool_dir, cache_dir,
         mcmc_data_dir, mcmc_max_samples, target, tolerance, n_bins, smooth, std_bins, levels, all_points,
         output, summary_json):
    cfg = TARGET_CONFIG[target]
    true_val = float(cfg["true_value"])
    picks = parse_picks(best_picks, {})
    global LEVELS
    LEVELS = tuple(float(v) for v in levels.split(","))

    df = pd.read_csv(manifest)
    df = df[df["status"].isin({s.strip() for s in include_status.split(",") if s.strip()})]
    df = df[df["expected_run_dir"].map(lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists())]

    samples, per_seed, summary = {}, {}, {"target": target, "tolerance": tolerance, "pair_model": pair_model,
                            "contour_levels": list(LEVELS), "samples": {}}
    for model in MODEL_ORDER:
        if model not in picks:
            continue
        strat, warm = picks[model]
        sub = df[(df["model"] == model + model_tag) & (df["strategy"] == strat)
                 & (df["warm_start"] == warm)]
        seeds = []
        for d in sub["expected_run_dir"]:
            Xs, ys, _ = _pooled_cell_data([d], False)
            if Xs is not None:
                seeds.append((np.asarray(Xs), np.asarray(ys).ravel()))
        n_used = len(seeds)
        if not seeds:
            click.echo(f"[warn] {model}/{strat}/{warm}: no runs", err=True)
            continue
        per_seed[model] = seeds
        X = np.concatenate([a for a, _ in seeds])
        y = np.concatenate([b for _, b in seeds])
        samples[model] = (X, y)
        summary["samples"][model] = {"strategy": strat, "warm": warm, "n_seeds": n_used}
        click.echo(f"[corner] {model:12} {strat}/{warm}: {n_used} seeds, {len(y)} points")
    Xb, Yb = H._load_xy_full(pool_dir, target, Path(cache_dir))
    samples["random pool"] = (np.asarray(Xb)[:, FREE_PARAM_INDICES], np.asarray(Yb).ravel())
    ref_names = ["random pool"]
    if mcmc_data_dir and cfg.get("has_mcmc_reference", False):
        from pmssm.data import load_mcmc_data  # noqa: PLC0415
        Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target,
                                max_samples=mcmc_max_samples or None)
        Xm = Xm.numpy() if hasattr(Xm, "numpy") else np.asarray(Xm)
        Ym = (Ym.numpy() if hasattr(Ym, "numpy") else np.asarray(Ym)).ravel()
        samples["emcee MCMC"] = (Xm[:, FREE_PARAM_INDICES], Ym)
        ref_names.append("emcee MCMC")
    for k, (X, y) in samples.items():
        summary["samples"].setdefault(k, {}).update(
            n_pooled=int(len(y)), n_inband=int(_band(y, true_val, tolerance).sum()))
    if pair_model not in samples:
        raise click.ClickException(f"--pair-model {pair_model} not among the picks")

    col_idx = [FREE_PARAM_NAMES.index(p) for p in SHOWN]
    def _sel(y):
        """Points entering the parameter panels: in band, or every valid one."""
        return np.isfinite(y) if all_points else _band(y, true_val, tolerance)

    summary["all_points"] = all_points
    inb = {k: X[_sel(y)] for k, (X, y) in samples.items()}
    # Parameter ranges as in distribution_overlay_best: min/max over the in-band samples.
    rng = []
    for ci in col_idx:
        rng.append((min(float(np.nanmin(X[:, ci])) for X in inb.values() if len(X)),
                    max(float(np.nanmax(X[:, ci])) for X in inb.values() if len(X))))
    edges = np.linspace(-4, 0, 41) if target == "DMRD" else np.linspace(-6, 3, 41)
    trange = (edges[0], edges[-1])
    n = len(SHOWN) + 1  # six parameters + the target

    def _std_map(r, c):
        """Seed-to-seed std of the pair model's density in panel (r, c), c > r.
        Axes as in the mirrored lower panel (c, r): x = variable r, y = variable c;
        the target column uses every point."""
        target_col = c == len(SHOWN)
        xr = rng[r]
        yr = trange if target_col else rng[c]
        hs = []
        for Xs, ys in per_seed[pair_model]:
            if target_col:
                ok = np.isfinite(ys) & (ys > 0)
                xv = Xs[ok][:, col_idx[r]]
                yv = np.log10(np.clip(ys[ok], 10 ** trange[0], 10 ** trange[1]))
            else:
                Xi = Xs[_sel(ys)]
                xv, yv = Xi[:, col_idx[r]], Xi[:, col_idx[c]]
            h, _, _ = np.histogram2d(xv, yv, bins=std_bins, range=[xr, yr])
            hs.append(h / max(h.sum(), 1) * std_bins ** 2)  # density on the unit square
        return np.std(np.stack(hs), axis=0, ddof=1), xr, yr

    std_maps = {(r, c): _std_map(r, c) for r in range(n) for c in range(r + 1, n)}
    allstd = np.concatenate([m[0].ravel() for m in std_maps.values()])
    snorm = Normalize(0, float(np.percentile(allstd[allstd > 0], 99)))
    smap = plt.get_cmap("magma")
    summary["std_panels"] = {"bins": std_bins, "colour_max": snorm.vmax,
                             "n_seeds": len(per_seed[pair_model])}

    LW = 1.5
    fig, axes = plt.subplots(n, n, figsize=(14, 14))
    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.07, top=0.93, wspace=0.08, hspace=0.08)
    pair_keys = [ref_names[-1], pair_model]

    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if c > r:
                z, xr, yr = std_maps[(r, c)]
                im = ax.imshow(z.T, origin="lower", extent=[*xr, *yr], aspect="auto",
                               cmap=smap, norm=snorm, interpolation="nearest")
                ax.tick_params(axis="both", labelleft=False, labelbottom=False, labelsize=10)
                if SHOWN[r] in STEP:
                    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(STEP[SHOWN[r]]))
                if c < len(SHOWN) and SHOWN[c] in STEP:
                    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(STEP[SHOWN[c]]))
                ax.grid(alpha=0.3)
                continue
            if r == c and r < len(SHOWN):
                bins = np.linspace(*rng[r], 41)
                for key in samples:
                    x = inb[key][:, col_idx[r]]
                    ax.hist(x[np.isfinite(x)], bins=bins, density=True, histtype="step", lw=LW,
                            color=_arm_color(key, ref_names))
                ax.set_yticks([])
                ax.set_xlim(*rng[r])
                _, ymax = ax.get_ylim()
                ax.set_ylim(0, ymax * 1.05)
            elif r == c:
                # Target marginal, as in distribution_overlay_best: density in
                # log10(target), log y, every point.
                width = edges[1] - edges[0]
                ax.axvspan(np.log10(true_val * (1 - tolerance)), np.log10(true_val * (1 + tolerance)),
                           color="gold", alpha=0.3, lw=0)
                lo_d, hi_d = np.inf, 0.0
                for key in samples:
                    y = samples[key][1]
                    y = y[np.isfinite(y) & (y > 0)]
                    t = np.log10(np.clip(y, 10 ** trange[0], 10 ** trange[1]))
                    w = np.full(len(t), 1.0 / (len(t) * width))
                    nh, _, _ = ax.hist(t, bins=edges, weights=w, histtype="step", lw=LW,
                                       color=_arm_color(key, ref_names))
                    pos = nh[nh > 0]
                    if len(pos):
                        lo_d, hi_d = min(lo_d, pos.min()), max(hi_d, pos.max())
                ax.set_yscale("log")
                ax.set_ylim(lo_d * 0.5, hi_d * 1.3)
                ax.set_yticks([], minor=False)
                ax.set_yticks([], minor=True)
                ax.set_xlim(*trange)
            else:
                target_row = r == len(SHOWN)
                xr = rng[c]
                yr = trange if target_row else rng[r]
                if target_row:
                    ax.axhspan(np.log10(true_val * (1 - tolerance)), np.log10(true_val * (1 + tolerance)),
                               color="gold", alpha=0.3, lw=0)
                for key in pair_keys:
                    X, y = samples[key]
                    if target_row:
                        ok = np.isfinite(y) & (y > 0)
                        xv = X[ok][:, col_idx[c]]
                        yv = np.log10(np.clip(y[ok], 10 ** trange[0], 10 ** trange[1]))
                    else:
                        xv, yv = inb[key][:, col_idx[c]], inb[key][:, col_idx[r]]
                    _contour(ax, xv, yv, xr, yr, _arm_color(key, ref_names), n_bins, smooth)
                ax.set_xlim(*xr)
                ax.set_ylim(*yr)
                if not target_row and SHOWN[r] in STEP:
                    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(STEP[SHOWN[r]]))
                if c == 0:
                    ax.set_ylabel(PRETTY[SHOWN[r]] if not target_row
                                  else r"$\log_{10}$ " + (cfg.get("label") or target), fontsize=13)
                else:
                    ax.tick_params(axis="y", labelleft=False)
            ax.grid(alpha=0.3)
            ax.tick_params(axis="both", labelsize=10)
            if c < len(SHOWN) and SHOWN[c] in STEP:
                ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(STEP[SHOWN[c]]))
            if r == n - 1:
                ax.set_xlabel(PRETTY[SHOWN[c]] if c < len(SHOWN)
                              else r"$\log_{10}$ " + (cfg.get("label") or target), fontsize=13)
                for lab in ax.get_xticklabels():
                    lab.set_rotation(45)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    order = [m for m in MODEL_ORDER if m in samples] + ref_names
    handles = [Line2D([0], [0], color=_arm_color(k, ref_names), lw=2.2,
                      label=H.MODEL_DISPLAY.get(k, k)) for k in order]
    handles.append(Patch(facecolor="gold", alpha=0.3, label=f"target $\\pm{int(tolerance * 100)}\\%$"))
    if len(handles) % 2:  # keep the two contour entries in one legend column
        handles.append(Line2D([], [], lw=0, label=" "))
    handles += [Line2D([0], [0], color="0.3", lw=1.6, ls="-", label=f"{LEVELS[0]:.0%} contour"),
                Line2D([0], [0], color="0.3", lw=1.0, ls="--", label=f"{LEVELS[1]:.0%} contour")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.495, 0.935), ncol=5,
               fontsize=13, frameon=False, handlelength=2.2, columnspacing=1.5, labelspacing=0.5)
    top_p, bot_p = axes[0, -1].get_position(), axes[-2, -1].get_position()
    cax = fig.add_axes([top_p.x1 + 0.012, bot_p.y0, 0.012, top_p.y1 - bot_p.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"{H.MODEL_DISPLAY.get(pair_model, pair_model)}: std. dev. of the density over seeds",
                 fontsize=13)
    cb.ax.tick_params(labelsize=10)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out}")
    if summary_json:
        json.dump(summary, open(summary_json, "w"), indent=2)
        click.echo(f"wrote {summary_json}")


if __name__ == "__main__":
    main()
