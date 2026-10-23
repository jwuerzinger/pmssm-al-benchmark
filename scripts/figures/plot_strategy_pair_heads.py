"""Two-target figure: entropy batch against tol+rnd for the dropout surrogates.

Left panel the relic density, right panel the exclusion ratio. Each panel
shows the in-band yield per simulator call (hits/desired at the 10% band)
against the labelled-set size for the Transformer and the DNN with the
regression head, once acquiring with ``entropy_batch`` (solid) and once with
``tol_only_random`` (dashed), both with cold restarts. The random-scan line
is the per-attempt prevalence of each target's pool. The metric, the |L|
interpolation and the styling are those of plot_hits_per_desired_heads.py.

Usage
-----
    python scripts/plot_strategy_pair_heads.py --output .../strategy_pair_heads.png
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

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_hit_rate_trajectories_multiseed as H              # noqa: E402
from plot_hits_per_desired_heads import _run_curve, _cell_on_grid, SMOOTH, smooth_rows  # noqa: E402

TARGETS = {
    "DMRD": dict(manifest="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
                 baseline="/ptmp/jwuerzin/analysis/all_runs/random_baseline_prevalence.json",
                 tag="", label="Ω", mcmc=904840 / 48305152),
    "ExpR": dict(manifest="/ptmp/jwuerzin/analysis/expr_runs/sweep_manifest.csv",
                 baseline="/ptmp/jwuerzin/analysis/expr_runs/random_baseline_prevalence.json",
                 tag="_expr", label="r", mcmc=0.0),
}
# Warm mode per model, as in the paper: cold restarts for the neural nets,
# warm starts for the GPs (2026-10-13: GPs added).
MODELS = {"transformer": "cold", "dnn": "cold", "deep_gp": "warm", "exact_gp": "warm"}
STRATEGIES = {"entropy_batch": "-", "tol_only_random": "--"}
STRATEGY_LABEL = {"entropy_batch": "entropy batch", "tol_only_random": "tol + rnd"}


@click.command()
@click.option("--dmrd-manifest", default="", show_default=True,
              help="Override the relic-density manifest (e.g. sweep_manifest_paper.csv).")
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--min-seeds-axis", default=2, show_default=True)
@click.option("--include-status", default="completed,timeout,running,missing", show_default=True)
@click.option("--uncertainty", default="sem", show_default=True)
@click.option("--smooth", default=0, show_default=True,
              help="Rolling-average window in grid points (of 300) applied to each seed.")
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(dmrd_manifest, tolerance, min_seeds, min_seeds_axis, include_status, uncertainty, smooth, output, summary_json):
    SMOOTH["window"] = smooth
    if dmrd_manifest:
        TARGETS["DMRD"]["manifest"] = dmrd_manifest
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    summary = {}
    for ax, (target, cfg) in zip(axes, TARGETS.items()):
        base = json.load(open(cfg["baseline"]))
        true_val = float(base["true_value"])
        H._DESIRED_P_VALID = float(base["p_valid"]["p_valid"])
        rnd = float(base["prevalence_per_attempt"][f"{tolerance:.4f}"])
        df = pd.read_csv(cfg["manifest"])
        df = df[df["status"].isin(allowed)]
        df = df[df["expected_run_dir"].map(lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists())]
        summary[target] = {"random_per_attempt": rnd}
        for model in MODELS:
            for strat, ls in STRATEGIES.items():
                sub = df[(df["model"] == model + cfg["tag"]) & (df["strategy"] == strat)
                         & (df["warm_start"] == MODELS[model])]
                curves = []
                for run_dir in sub["expected_run_dir"].dropna():
                    try:
                        c = _run_curve(run_dir, true_val, tolerance)
                    except Exception as exc:  # noqa: BLE001
                        click.echo(f"[warn] skip {run_dir}: {exc}", err=True)
                        continue
                    if c is not None:
                        curves.append(c)
                if len(curves) < min_seeds:
                    click.echo(f"[warn] {target} {model}/{strat}: {len(curves)} seeds", err=True)
                    continue
                grid, Y = _cell_on_grid(curves, min_seeds_axis)
                if len(grid) == 0:
                    click.echo(f"[warn] no point with {min_seeds_axis} seeds; curve skipped", err=True)
                    continue
                H._draw_curve(ax, grid, smooth_rows(Y), color=H.MODEL_COLORS.get(model, "gray"),
                              linestyle=ls, marker=None, label=None, uncertainty=uncertainty)
                it_max = int(max(c[2][-1] for c in curves))
                reached = [c for c in curves if c[2][-1] == it_max]
                summary[target][f"{model}/{strat}"] = {
                    "n_seeds": len(curves), "final_iteration": it_max,
                    "final_mean": float(np.mean([c[1][-1] for c in reached])),
                    "yield_at_iter_40": float(np.mean([c[1][list(c[2]).index(40)] for c in curves if 40 in list(c[2])])),
                }
        ax.axhline(rnd, color="black", linestyle=":", linewidth=1.4)
        if cfg["mcmc"] > 0:
            ax.axhline(cfg["mcmc"], color="black", linestyle="-.", linewidth=1.4)
        # Panel tag names the target; the band definition is in the caption.
        ax.text(0.02, 0.98, {"DMRD": "relic density", "ExpR": "LHC exclusion"}[target],
                transform=ax.transAxes, ha="left", va="top", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.8))
        ax.set_xlabel(r"Dataset size $|L|$", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("In-band yield per simulator call", fontsize=14)
    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, max(ymax, 0.05) * 1.05)
    fig.tight_layout()
    handles = [Line2D([0], [0], color=H.MODEL_COLORS[m], lw=2.4, label=H.MODEL_DISPLAY[m]) for m in MODELS]
    handles += [Line2D([0], [0], color="black", linestyle=ls, lw=1.6, label=STRATEGY_LABEL[s])
                for s, ls in STRATEGIES.items()]
    handles += [Line2D([0], [0], color="black", linestyle=":", lw=1.4, label="random scan"),
                Line2D([0], [0], color="black", linestyle="-.", lw=1.4, label="emcee MCMC")]
    # Upper right of the exclusion panel: with the GPs added, nothing there
    # rises above ~0.08, whereas the relic-density Exact GP reaches the top.
    axes[1].legend(handles=handles, loc="upper right", bbox_to_anchor=(0.985, 0.985), ncol=2,
                   fontsize=9.5, frameon=True, framealpha=0.9, columnspacing=1.0, handlelength=2.2)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    H._enforce_marker_policy(fig)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out}")
    if summary_json:
        json.dump(summary, open(summary_json, "w"), indent=2)
    for t, d in summary.items():
        for k, v in d.items():
            if isinstance(v, dict):
                click.echo(f"  {t:5} {k:28} final={v['final_mean']:.4f} at it.{v['final_iteration']}  it.40={v['yield_at_iter_40']:.4f}")


if __name__ == "__main__":
    main()
