"""Two-panel in-band-support figures: regression picks vs classification-entropy arms.

Companion of plot_hits_per_desired_heads.py / plot_accuracy_heads.py with the
same layout (left: best regression cell per surrogate; right: the
classification head with entropy acquisition). Two metrics, one figure each:

  --metric budget    in-band support covered against the budget spent as a
                     fraction of the reference (random-scan) dataset, i.e.
                     (train + validation points simulated) / N_pool, with the
                     random scan's own prefix curve as the dotted reference
                     (it reaches 1 at x = 1 by construction);
  --metric compute   the same coverage against cumulative surrogate compute in
                     GPU hours (training plus selection per iteration, resume
                     chains followed). No random reference exists on this axis.

Nothing is re-derived: the support cells come from plot_prelim_support's
``_support_from_cache`` (the paper's pool support: 12^3 quantile grid over
(M_1, M_2, mu), cells with >= 20 in-band random-scan points), the point
sequences from plot_support_efficiency ``_al_sequence`` and the GPU hours from
plot_prelim_compute_support ``_compute_hours``. Seeds are interpolated onto a
shared logarithmic grid and a curve ends where fewer than ``--min-seeds-axis``
seeds remain.

Usage
-----
    python scripts/plot_support_heads.py --metric budget  --output .../support_heads.png
    python scripts/plot_support_heads.py --metric compute --output .../support_compute_heads.png
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
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_hit_rate_trajectories_multiseed as H            # noqa: E402
from plot_hits_per_desired_heads import REG_PICKS, CLS_PICKS, parse_picks, SMOOTH, smooth_rows, parse_axis_override, axis_min  # noqa: E402
from coverage_saturation import _cells                      # noqa: E402
from plot_support_efficiency import AXES, _al_sequence, _band_mask, _ref_curve  # noqa: E402
from plot_prelim_support import _support_from_cache         # noqa: E402
from plot_prelim_compute_support import _compute_hours      # noqa: E402
from pmssm.config import PARAM_ORDER                        # noqa: E402

GRID_POINTS = 300


def _seed_curve(d: Path, metric: str, ax_idx, edges, tmap, n_target, n_total,
                true_value, tolerance, band_side):
    """(x, coverage, iteration) after each iteration for one seed."""
    got = _al_sequence(d, ax_idx, False)
    if got is None:
        return None
    X, Y = got
    cid = np.where(_band_mask(Y, true_value, tolerance, band_side), _cells(X, edges), -1)
    s = torch.load(d / "state.pt", weights_only=False, map_location="cpu")
    nt = [int(v) for v in (s.get("al_n_train") or [])]
    nv = [int(v) for v in (s.get("al_n_val") or [])]
    n = min(len(nt), len(nv))
    if metric == "compute":
        hours = _compute_hours(d)
        n = min(n, len(hours))
    if n < 2:
        return None
    seen, cov, xs, its = set(), [], [], []
    for i in range(n):
        stop = min(nt[i] + nv[i], len(cid))
        chunk = cid[:stop]
        good = chunk[chunk >= 0]
        good = good[tmap[good] >= 0]
        seen |= set(good.tolist())
        cov.append(len(seen) / n_target)
        xs.append(hours[i] if metric == "compute" else (nt[i] + nv[i]) / n_total)
        its.append(i + 1)
    return np.asarray(xs, float), np.asarray(cov, float), np.asarray(its, int)


def _on_log_grid(curves, min_seeds):
    lo = max(c[0][0] for c in curves)
    hi = max(c[0][-1] for c in curves)
    grid = np.geomspace(lo, hi, GRID_POINTS)
    Y = np.full((len(curves), GRID_POINTS), np.nan)
    for k, (x, y, _) in enumerate(curves):
        inside = (grid >= x[0]) & (grid <= x[-1])
        Y[k, inside] = np.interp(grid[inside], x, y)
    keep = np.sum(~np.isnan(Y), axis=0) >= min_seeds
    return grid[keep], Y[:, keep]


MARK_SHAPES = ["o", "s", "^", "D", "v"]


def _compute_marks(ax, budget_curves, compute_curves, marks, color, min_seeds):
    """Points on the budget curve where the cumulative GPU hours reach each mark.

    Per seed the budget fraction and coverage at which the compute curve crosses
    the mark are interpolated (both curves are indexed by iteration), then
    averaged over the seeds that reach it. Drawn as scatter collections so the
    figure's line-marker policy leaves them alone.
    """
    out = {}
    for k, h in enumerate(marks):
        xs, ys = [], []
        for (xb, cb, ib), (hc, _, ic) in zip(budget_curves, compute_curves):
            n = min(len(ib), len(ic))
            hc, xb, cb = hc[:n], xb[:n], cb[:n]
            if hc[-1] < h or hc[0] > h:
                continue
            xs.append(np.interp(h, hc, xb))
            ys.append(np.interp(h, hc, cb))
        if len(xs) < min_seeds:
            continue
        x, y = float(np.exp(np.mean(np.log(xs)))), float(np.mean(ys))
        ax.scatter([x], [y], marker=MARK_SHAPES[k % len(MARK_SHAPES)], s=46, color=color,
                   edgecolor="black", linewidth=0.8, zorder=5)
        out[f"{h:g} GPU h"] = {"budget_fraction": x, "coverage": y, "n_seeds": len(xs)}
    return out


def _panel(ax, df, picks, metric, S, ax_idx, true_value, tolerance, band_side,
           min_seeds, min_seeds_axis, uncertainty, compute_marks=()):
    summary = {}
    for model, (strat, warm) in picks.items():
        sub = df[(df["model"] == model) & (df["strategy"] == strat) & (df["warm_start"] == warm)]
        curves, ccurves = [], []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                args = (ax_idx, S["edges"], S["tmap"], S["n_target"], S["n_total"],
                        true_value, tolerance, band_side)
                c = _seed_curve(Path(run_dir), metric, *args)
                cc = (_seed_curve(Path(run_dir), "compute", *args)
                      if compute_marks and metric == "budget" else None)
            except Exception as exc:  # noqa: BLE001
                click.echo(f"[warn] skip {run_dir}: {type(exc).__name__}: {exc}", err=True)
                continue
            if c is not None:
                curves.append(c)
                if cc is not None:
                    ccurves.append((c, cc))
        if len(curves) < min_seeds:
            click.echo(f"[warn] {model}/{strat}/{warm}: {len(curves)} seeds, skipped", err=True)
            continue
        grid, Y = _on_log_grid(curves, axis_min(model, min_seeds_axis))
        if len(grid) == 0:
            click.echo(f"[warn] no point with {min_seeds_axis} seeds; curve skipped", err=True)
            continue
        H._draw_curve(ax, grid, smooth_rows(Y), color=H.MODEL_COLORS.get(model, "gray"),
                      linestyle="-", marker=None, label=None, uncertainty=uncertainty)
        marks = (_compute_marks(ax, [a for a, _ in ccurves], [b for _, b in ccurves],
                                compute_marks, H.MODEL_COLORS.get(model, "gray"),
                                axis_min(model, min_seeds_axis))  # same rule as the curve end
                 if ccurves else {})
        it_max = int(max(c[2][-1] for c in curves))
        reached = [c for c in curves if c[2][-1] == it_max]
        finals = np.array([c[1][-1] for c in reached])
        summary[model] = {
            "compute_marks": marks,
            "strategy": strat, "warm": warm, "n_seeds": len(curves),
            "final_iteration": it_max, "n_seeds_at_final": len(reached),
            "final_coverage_mean": float(finals.mean()),
            "final_coverage_sem": float(finals.std(ddof=1) / np.sqrt(len(finals))) if len(finals) > 1 else None,
            "final_x_mean": float(np.mean([c[0][-1] for c in reached])),
            "coverage_at_iter_40": float(np.mean([c[1][list(c[2]).index(40)] for c in curves if 40 in list(c[2])]))
            if any(40 in list(c[2]) for c in curves) else None,
        }
    return summary


@click.command()
@click.option("--metric", type=click.Choice(["budget", "compute"]), required=True)
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv", show_default=True)
@click.option("--pool-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--pool-cache-dir", default="/ptmp/jwuerzin/analysis/pool_cache", show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--true-value", default=0.12, show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--n-bins", default=12, show_default=True)
@click.option("--min-cell", default=20, show_default=True)
@click.option("--band-side", default="in", type=click.Choice(["in", "out"]), show_default=True)
@click.option("--expect-cells", default=213, show_default=True,
              help="Guard: the published pool support has this many cells.")
@click.option("--min-budget", default=2000, show_default=True,
              help="Smallest prefix (points) of the random-scan reference curve.")
@click.option("--reg-picks", default="", show_default=True,
              help="Override regression picks: model:strategy:warm[,...].")
@click.option("--cls-picks", default="", show_default=True,
              help="Override classification picks: model:strategy:warm[,...].")
@click.option("--model-tag", default="", show_default=True)
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--min-seeds-axis", default=2, show_default=True)
@click.option("--include-status", default="completed,timeout,running,missing", show_default=True)
@click.option("--uncertainty", default="sem", show_default=True)
@click.option("--mcmc-reference-json", default="", show_default=True,
              help="With --metric budget: support_efficiency_*.json whose pool panel holds "
                   "the raw-emcee-chain curve (same cells, same x), drawn dash-dot.")
@click.option("--compute-marks", default="", show_default=True,
              help="With --metric budget: GPU-hour milestones (e.g. 0.3,3,30) marked "
                   "on every curve, so the budget figure carries the compute axis too.")
@click.option("--seeds-axis-override", default="", show_default=True,
              help="Per-model --min-seeds-axis, e.g. tabpfn_expr:4 (full model name).")
@click.option("--smooth", default=0, show_default=True,
              help="Rolling-average window in grid points (of 300) applied to each seed.")
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(metric, manifest, pool_dir, pool_cache_dir, target, true_value, tolerance,
         n_bins, min_cell, band_side, expect_cells, min_budget, reg_picks, cls_picks, model_tag,
         min_seeds, min_seeds_axis, include_status, uncertainty, mcmc_reference_json, compute_marks, seeds_axis_override, smooth, output,
         summary_json):
    SMOOTH["window"] = smooth
    parse_axis_override(seeds_axis_override)
    ax_idx = [PARAM_ORDER.index(a) for a in AXES]
    edges, tmap, n_target, oX, oY, n_rows, _ = _support_from_cache(
        pool_dir, target, pool_cache_dir, tolerance, n_bins, min_cell, true_value, band_side)
    if expect_cells and n_target != expect_cells:
        raise click.ClickException(f"support has {n_target} cells, expected {expect_cells}")
    S = dict(edges=edges, tmap=tmap, n_target=n_target, n_total=n_rows)
    marks = [float(v) for v in compute_marks.split(",") if v.strip()]
    mcmc_ref = None
    if mcmc_reference_json and metric == "budget":
        pan = json.load(open(mcmc_reference_json))["panels"]["pool"]
        if pan["n_target_cells"] != n_target or pan["dataset_calls"] != n_rows:
            raise click.ClickException(
                f"MCMC reference is on {pan['n_target_cells']} cells / {pan['dataset_calls']} "
                f"calls, this figure on {n_target} / {n_rows}")
        r = pan["reference"]["mcmc"]
        f, c = np.asarray(r["fraction"]), np.asarray(r["coverage"])
        # Same window as the random-scan reference, which starts at --min-budget.
        keep = (f <= 1.0) & (f >= min_budget / n_rows)
        mcmc_ref = (f[keep], c[keep])

    df = pd.read_csv(manifest)
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    df = df[df["expected_run_dir"].map(lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists())]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    reg_picks = parse_picks(reg_picks, REG_PICKS, model_tag)
    cls_picks = parse_picks(cls_picks, CLS_PICKS, model_tag)
    panels = [("regression", reg_picks), ("classification", cls_picks)]
    summary = {"metric": metric, "n_target_cells": n_target, "n_reference_rows": n_rows}
    for ax, (name, picks) in zip(axes, panels):
        summary[name] = _panel(ax, df, picks, metric, S, ax_idx, true_value, tolerance,
                               band_side, min_seeds, min_seeds_axis, uncertainty, marks)
        if metric == "budget":
            rcells = np.where(_band_mask(oY, true_value, tolerance, band_side), _cells(oX, edges), -1)
            rf, rc = _ref_curve(rcells, tmap, n_target, n_rows, 40, min_budget)
            ax.plot(rf, rc, color="black", linestyle=":", linewidth=1.4)
            if mcmc_ref is not None:
                ax.plot(*mcmc_ref, color="dimgray", linestyle="-.", linewidth=1.6)
            ax.set_xlabel("Budget / reference dataset size", fontsize=14)
        else:
            ax.set_xlabel("Cumulative surrogate compute (GPU h)", fontsize=14)
        ax.set_xscale("log")
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.98, name, transform=ax.transAxes, ha="left", va="top", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.8))
    axes[0].set_ylabel(f"In-band support covered ({n_target} cells)", fontsize=14)
    for ax in axes:
        ax.set_ylim(0, 1.05)
    fig.tight_layout()

    models = [m for m in reg_picks if m in summary[panels[0][0]] or m in summary[panels[1][0]]]
    handles = [Line2D([0], [0], color=H.MODEL_COLORS.get(m, "gray"), lw=2.4,
                      label=H.MODEL_DISPLAY.get(m, m)) for m in models]
    if metric == "budget":
        handles.append(Line2D([0], [0], color="black", linestyle=":", lw=1.4, label="random scan"))
        if mcmc_ref is not None:
            handles.append(Line2D([0], [0], color="dimgray", linestyle="-.", lw=1.6,
                                  label="emcee MCMC"))
        handles += [Line2D([0], [0], linestyle="none", marker=MARK_SHAPES[k % len(MARK_SHAPES)],
                           markersize=7, markerfacecolor="white", markeredgecolor="black",
                           label=f"{h:g} GPU h") for k, h in enumerate(marks)]
    # Before the legend: the policy strips markers from legend proxies too,
    # which blanked the GPU-hour symbols.
    H._enforce_marker_policy(fig)
    axes[1].legend(handles=handles, loc="lower right", bbox_to_anchor=(0.985, 0.03), ncol=2,
                   fontsize=9.5, frameon=True, framealpha=0.9, columnspacing=1.0, handlelength=2.0)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out}")
    if summary_json:
        json.dump(summary, open(summary_json, "w"), indent=2)
        click.echo(f"wrote {summary_json}")
    for name, _ in panels:
        for m, v in summary[name].items():
            click.echo(f"  {name:26} {m:12} cov={v['final_coverage_mean']:.3f}"
                       f"±{(v['final_coverage_sem'] or 0):.3f} at it.{v['final_iteration']}"
                       f" ({v['n_seeds_at_final']}/{v['n_seeds']}), x_final={v['final_x_mean']:.4g};"
                       f" it.40 cov={v['coverage_at_iter_40']}")


if __name__ == "__main__":
    main()
