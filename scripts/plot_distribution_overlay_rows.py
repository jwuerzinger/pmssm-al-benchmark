"""Holistic comparison in two rows: regression head on top, classification below.

Merges the paper's two distribution overlays (plot_distribution_overlay.py,
default picks and --arms heads variant) into one figure with shared columns:

  * columns: the six parameters the relic density couples to (the trilinears
    A_t, A_b, A_tau are dropped: their in-band marginals are flat for every
    sample) plus the relic-density marginal on log-log axes;
  * top row: each surrogate's best regression cell (``--reg-picks``);
  * bottom row: each surrogate's classification-entropy cell (``--cls-picks``);
  * both rows: the valid random pool and the emcee MCMC reference.

Parameter panels are restricted to the +-10% band and density-normalised;
the target panel shows every point. Seeds are pooled from state.pt, the same
reading as plot_distribution_overlay.py. Colours follow the hit-rate figures.

Usage
-----
    python scripts/plot_distribution_overlay_rows.py \
        --reg-picks tabpfn:tol_only_random:tabpfn \
        --output .../distribution_overlay_rows.png --summary-json .../distribution_overlay_rows.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_hit_rate_trajectories_multiseed as H                         # noqa: E402
from plot_hits_per_desired_heads import REG_PICKS, CLS_PICKS, parse_picks  # noqa: E402
from plot_al_input_target_diagnostics import _pooled_cell_data, _arm_color  # noqa: E402
from plot_distribution_overlay import PRETTY                               # noqa: E402
from analyse_runs import FREE_PARAM_NAMES, FREE_PARAM_INDICES              # noqa: E402
from pmssm.config import TARGET_CONFIG                                     # noqa: E402

SHOWN = ["M_1", "M_2", "mu", "meL", "meR", "tanb"]
MODEL_ORDER = ["transformer", "dnn", "deep_gp", "exact_gp", "tabpfn"]


def _band(y, true_val, tol):
    return np.isfinite(y) & (np.abs(y - true_val) / true_val < tol)



def _draw_single(samples, ref_names, picks, col_idx, true_val, tol, cfg, target, output):
    """Main-body layout (2026-10-13): a 2 x 4 grid of equal panels. Top row
    M_1, M_2, mu and the target; bottom row meL, meR, tan(beta) and the legend.
    Parameter panels: in-band points, density, no y axis. Target panel: every
    point, density in log10 of the target (a proper density on the plotted
    axis), log y so the narrow MCMC peak does not flatten the rest, no y axis.
    One line width throughout."""
    LW = 1.5
    fig = plt.figure(figsize=(14, 6.4))
    gs = fig.add_gridspec(2, 4, wspace=0.10, hspace=0.36, left=0.02, right=0.98,
                          bottom=0.09, top=0.98)
    slots = {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}
    inb = {k: X[_band(y, true_val, tol)] for k, (X, y) in samples.items()}
    for j, ci in enumerate(col_idx):
        ax = fig.add_subplot(gs[slots[j]])
        lo = min(float(np.nanmin(X[:, ci])) for X in inb.values() if len(X))
        hi = max(float(np.nanmax(X[:, ci])) for X in inb.values() if len(X))
        bins = np.linspace(lo, hi, 41)
        for key, X in inb.items():
            x = X[:, ci]
            ax.hist(x[np.isfinite(x)], bins=bins, density=True, histtype="step", lw=LW,
                    color=_arm_color(key, ref_names))
        ax.set_yticks([])
        ax.grid(alpha=0.3, axis="x")
        ax.margins(x=0)
        ax.tick_params(axis="x", labelsize=10.5)
        ax.set_xlabel(PRETTY[SHOWN[j]], fontsize=13)
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax * 1.05)
    # Target panel.
    ax = fig.add_subplot(gs[0, 3])
    # Same drawing as the parameter panels (2026-10-13): step histograms with
    # 40 bins, closed down to the axis at both ends. Bins are uniform in
    # log10(target); the weights make it a density in log10, and the y floor
    # is set below the lowest bin so no curve leaves the panel.
    # log10 range of the target: relic density 1e-4..1, exclusion ratio 1e-6..1e3.
    edges = np.linspace(-4, 0, 41) if target == "DMRD" else np.linspace(-6, 3, 41)
    width = edges[1] - edges[0]
    ax.axvspan(true_val * (1 - tol), true_val * (1 + tol), color="gold", alpha=0.3, lw=0)
    lo_d, hi_d = np.inf, 0.0
    for key, (X, y) in samples.items():
        y = y[np.isfinite(y) & (y > 0)]
        yc = np.clip(y, 10 ** edges[0], 10 ** edges[-1])
        w = np.full(len(yc), 1.0 / (len(yc) * width))
        n, _, _ = ax.hist(yc, bins=10 ** edges, weights=w, histtype="step", lw=LW,
                          color=_arm_color(key, ref_names))
        pos = n[n > 0]
        if len(pos):
            lo_d, hi_d = min(lo_d, pos.min()), max(hi_d, pos.max())
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(lo_d * 0.5, hi_d * 1.3)
    ax.set_yticks([], minor=False)
    ax.set_yticks([], minor=True)
    ax.grid(alpha=0.3, axis="x", which="major")
    ax.tick_params(axis="x", labelsize=10.5)
    ax.set_xlabel(cfg.get("label") or target, fontsize=13)
    ax.set_xlim(10 ** edges[0], 10 ** edges[-1])
    # Legend panel.
    axl = fig.add_subplot(gs[1, 3])
    axl.axis("off")
    order = [m for m in MODEL_ORDER if m in samples] + list(ref_names)
    handles = [Line2D([0], [0], color=_arm_color(k, ref_names), lw=2.2,
                      label=H.MODEL_DISPLAY.get(k, k)) for k in order]
    handles.append(Patch(facecolor="gold", alpha=0.3, label=f"target $\\pm{int(tol * 100)}\\%$"))
    axl.legend(handles=handles, loc="center", ncol=2, fontsize=12, frameon=False,
               handlelength=2.0, columnspacing=1.5, labelspacing=0.6)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out}")

@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv", show_default=True)
@click.option("--reg-picks", default="", show_default=True)
@click.option("--cls-picks", default="", show_default=True)
@click.option("--model-tag", default="", show_default=True,
              help="Manifest suffix of the model names, e.g. _expr.")
@click.option("--best-picks", default="", show_default=True,
              help="Single row instead of regression/classification: model:strategy:warm,... "
                   "(the per-model best configuration; legend names the head).")
@click.option("--include-status", default="completed,timeout,running,missing", show_default=True)
@click.option("--pool-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(manifest, reg_picks, cls_picks, model_tag, best_picks, include_status, pool_dir, cache_dir, mcmc_data_dir,
         mcmc_max_samples, target, tolerance, output, summary_json):
    cfg = TARGET_CONFIG[target]
    true_val = float(cfg["true_value"])
    if best_picks:
        rows = [("best", parse_picks(best_picks, {}))]
    else:
        rows = [("regression", parse_picks(reg_picks, REG_PICKS)),
                ("classification", parse_picks(cls_picks, CLS_PICKS))]
    nr = len(rows)

    df = pd.read_csv(manifest)
    df = df[df["status"].isin({s.strip() for s in include_status.split(",") if s.strip()})]
    df = df[df["expected_run_dir"].map(lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists())]

    col_idx = [FREE_PARAM_NAMES.index(p) for p in SHOWN]
    summary = {"target": target, "tolerance": tolerance, "parameters": SHOWN, "rows": {}}

    # References, shared by both rows.
    refs = {}
    Xb, Yb = H._load_xy_full(pool_dir, target, Path(cache_dir))
    refs["random pool"] = (np.asarray(Xb)[:, FREE_PARAM_INDICES], np.asarray(Yb).ravel())
    from pmssm.data import load_mcmc_data  # noqa: PLC0415
    if mcmc_data_dir and cfg.get("has_mcmc_reference", False):
      Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target, max_samples=mcmc_max_samples or None)
      Xm = Xm.numpy() if hasattr(Xm, "numpy") else np.asarray(Xm)
      Ym = (Ym.numpy() if hasattr(Ym, "numpy") else np.asarray(Ym)).ravel()
      refs["emcee MCMC"] = (Xm[:, FREE_PARAM_INDICES], Ym)
    ref_names = list(refs)

    row_samples = []
    for row_name, picks in rows:
        samples = {}
        summary["rows"][row_name] = {}
        for model in MODEL_ORDER:
            if model not in picks:
                continue
            strat, warm = picks[model]
            sub = df[(df["model"] == model + model_tag) & (df["strategy"] == strat)
                     & (df["warm_start"] == warm)]
            X, y, n_used = _pooled_cell_data(list(sub["expected_run_dir"]), False)
            if X is None:
                click.echo(f"[warn] {row_name} {model}/{strat}/{warm}: no runs", err=True)
                continue
            samples[model] = (np.asarray(X), np.asarray(y).ravel())
            summary["rows"][row_name][model] = {
                "strategy": strat, "warm": warm, "n_seeds": n_used, "n_pooled": int(len(y)),
                "n_inband": int(_band(np.asarray(y).ravel(), true_val, tolerance).sum())}
            click.echo(f"[rows] {row_name:14} {model:12} {strat}/{warm}: {n_used} seeds, {len(y)} points")
        samples.update(refs)
        row_samples.append(samples)
    for name, (X, y) in refs.items():
        summary[name] = {"n_pooled": int(len(y)), "n_inband": int(_band(y, true_val, tolerance).sum())}

    if nr == 1:
        _draw_single(row_samples[0], ref_names, rows[0][1], col_idx, true_val, tolerance, cfg,
                     target, output)
        if summary_json:
            json.dump(summary, open(summary_json, "w"), indent=2)
        return
    # Layout (2026-10-13): a 2x3 grid of parameters, (M_1, M_2, mu) on top and
    # (meL, meR, tanb) below. Each parameter cell is a pair of panels sharing the
    # y scale, regression head above classification head. The two target
    # panels (regression above classification) fill the right-hand column.
    fig = plt.figure(figsize=(15, 8.2 if nr == 2 else 6.2))
    outer = fig.add_gridspec(1, 2, width_ratios=[3, 1.25], wspace=0.10,
                             left=0.05, right=0.94, bottom=0.07, top=0.91 if nr == 2 else 0.85)
    pgrid = outer[0].subgridspec(2, 3, hspace=0.30, wspace=0.07)
    inb_all = [{k: X[_band(y, true_val, tolerance)] for k, (X, y) in s.items()} for s in row_samples]
    tbins = np.logspace(-4, 0, 81) if target == "DMRD" else np.logspace(-6, 3, 81)

    for j, ci in enumerate(col_idx):
        lo = min(float(np.nanmin(X[:, ci])) for d in inb_all for X in d.values() if len(X))
        hi = max(float(np.nanmax(X[:, ci])) for d in inb_all for X in d.values() if len(X))
        bins = np.linspace(lo, hi, 41)
        pair = pgrid[j // 3, j % 3].subgridspec(nr, 1, hspace=0.0)
        ax_top = fig.add_subplot(pair[0])
        ax_bot = fig.add_subplot(pair[1], sharex=ax_top, sharey=ax_top) if nr == 2 else ax_top
        for r, (ax, inb) in enumerate(zip((ax_top, ax_bot)[:nr], inb_all)):
            for key, X in inb.items():
                x = X[:, ci]
                ax.hist(x[np.isfinite(x)], bins=bins, density=True, histtype="step",
                        lw=1.4 if key in ref_names else 1.5, color=_arm_color(key, ref_names))
            ax.set_yticks([])
            ax.grid(alpha=0.3)
            ax.margins(x=0)
            if nr == 2:
                ax.text(0.015, 0.94, "reg." if r == 0 else "cls.", transform=ax.transAxes,
                        ha="left", va="top", fontsize=9.5, color="0.25")
        if nr == 2:
            ax_top.tick_params(axis="x", labelbottom=False)
        ax_bot.tick_params(axis="x", labelsize=10)
        ax_bot.set_xlabel(PRETTY[SHOWN[j]], fontsize=13)
        _, ymax = ax_top.get_ylim()
        ax_top.set_ylim(0, ymax * (1.22 if nr == 2 else 1.05))  # headroom for the reg./cls. tags

    tgrid = outer[1].subgridspec(nr, 1, hspace=0.0)
    t_top = fig.add_subplot(tgrid[0])
    t_bot = fig.add_subplot(tgrid[1], sharex=t_top, sharey=t_top) if nr == 2 else t_top
    for r, (ax_t, samples) in enumerate(zip((t_top, t_bot)[:nr], row_samples)):
        ax_t.axvspan(true_val * (1 - tolerance), true_val * (1 + tolerance), color="gold", alpha=0.25, lw=0)
        for key, (X, y) in samples.items():
            y = y[np.isfinite(y) & (y > 0)]
            dens, _ = np.histogram(np.clip(y, tbins[0], tbins[-1]), bins=tbins, density=True)
            ax_t.stairs(dens, tbins, lw=1.5, color=_arm_color(key, ref_names))
        ax_t.axvline(true_val, color="black", ls=":", lw=1.0)
        ax_t.set_xscale("log")
        ax_t.set_yscale("log")
        ax_t.yaxis.tick_right()
        ax_t.yaxis.set_label_position("right")
        ax_t.set_ylabel("density", fontsize=12)
        ax_t.tick_params(axis="both", labelsize=10)
        ax_t.grid(alpha=0.3, which="both")
        if nr == 2:
            ax_t.text(0.03, 0.97, "regression" if r == 0 else "classification",
                      transform=ax_t.transAxes, ha="left", va="top", fontsize=11,
                      bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.85))
    if nr == 2:
        t_top.tick_params(axis="x", labelbottom=False)
    t_bot.set_xlabel(cfg.get("label") or target, fontsize=13)
    t_top.set_ylim(1e-3, t_top.get_ylim()[1] * 3)
    shown_models = [m for m in MODEL_ORDER if any(m in s for s in row_samples)]
    head = {"cls_entropy": "classification"}
    def _lab(m):
        if nr == 2:
            return H.MODEL_DISPLAY.get(m, m)
        return H.MODEL_DISPLAY.get(m, m)  # the caption names each model's configuration
    handles = [Line2D([0], [0], color=_arm_color(m, ref_names), lw=2.2, label=_lab(m))
               for m in shown_models]
    handles += [Line2D([0], [0], color=_arm_color(n, ref_names), lw=2.2, label=n) for n in ref_names]
    handles.append(Patch(facecolor="gold", alpha=0.25, label=f"target ±{int(tolerance * 100)}%"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=len(handles) if nr == 2 else 4,
               fontsize=11, frameon=True, framealpha=0.9, columnspacing=1.3, handlelength=1.8)

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
