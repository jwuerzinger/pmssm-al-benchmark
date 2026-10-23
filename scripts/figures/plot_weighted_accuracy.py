"""Class accuracy weighted by proximity to the threshold, from the prediction cache.

Reads the per-point predictions written by ``static_predictions_cache.py`` and
scores each (run, role, iteration) on the static-random eval set with weights

    w_i = exp(-t_i^2 / s_p),   t_i = log(Y_i / Y_0) (true value),   s_p = 0.1,

the proximity weight of Eq. (2) of the paper evaluated on the true target, so
that a point on the threshold counts fully and one a factor e away counts
exp(-10). The weighted accuracy is sum_i w_i [verdict correct] / sum_i w_i.

Layout follows Fig. 1: one panel per target, active learning solid, random
baseline dashed, mean over seeds with the standard error as band, plotted
against the dataset size |L|, each curve ending where its first seed ends.

Usage
-----
    python scripts/figures/plot_weighted_accuracy.py --output .../accuracy_weighted.png
"""
from __future__ import annotations

# --- repository bootstrap: lets this file run from any directory --------------
import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ------------------------------------------------------------------------------

import json
import sys
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


import scripts.figures.plot_hit_rate_trajectories_multiseed as H                  # noqa: E402
from scripts.figures.plot_best_config import TARGETS, MODEL_ORDER                 # noqa: E402
from scripts.analysis.static_predictions_cache import OUT                          # noqa: E402
from pmssm.data import TARGET_CONFIG                              # noqa: E402

LABEL = {"DMRD": "(a) Relic density", "ExpR": "(b) LHC exclusion"}


def _curves(target, s_p, include_status):
    import pandas as pd  # noqa: PLC0415
    cfg = TARGETS[target]
    d = Path(OUT[target])
    y = np.load(d / "_y_true_t.npy").astype(np.float64)
    thr = float(TARGET_CONFIG[target]["threshold"])
    w = np.exp(-(y - thr) ** 2 / s_p)
    truth = y >= thr
    df = pd.read_csv(cfg["manifest"])
    df = df[df["status"].isin(include_status)]
    out = {"n_eff": float(w.sum() ** 2 / (w ** 2).sum())}
    for model in MODEL_ORDER:
        strat, warm = cfg["picks"][model]
        sub = df[(df["model"] == model + cfg["tag"]) & (df["strategy"] == strat)
                 & (df["warm_start"] == warm)]
        per_role = {"al": [], "baseline": []}
        for run_dir in sub["expected_run_dir"]:
            f = d / f"{Path(run_dir).name}.npz"
            if not f.exists():
                continue
            z = np.load(f)
            for role in per_role:
                its = sorted(int(k.split("_")[-1]) for k in z.files
                             if k.startswith(f"{role}_"))
                if len(its) < 2:
                    continue
                nL = np.array([int(z[f"n_L_{i}"]) for i in its], float)
                acc = np.array([np.sum(w * ((z[f"{role}_{i}"] >= thr) == truth)) / w.sum()
                                for i in its])
                per_role[role].append((nL, acc))
        out[model] = per_role
    return out


def _mean_on_grid(seeds, n=200):
    lo = max(s[0][0] for s in seeds)
    hi = min(s[0][-1] for s in seeds)  # ends where the first seed stops
    g = np.linspace(lo, hi, n)
    Y = np.array([np.interp(g, nL, a) for nL, a in seeds])
    return g, Y.mean(0), Y.std(0, ddof=1) / np.sqrt(len(Y))


@click.command()
@click.option("--s-p", default=0.1, show_default=True, help="Proximity scale of Eq. (2).")
@click.option("--min-seeds", default=3, show_default=True)
@click.option("--include-status", default="completed,timeout,running,missing,stalled",
              show_default=True)
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(s_p, min_seeds, include_status, output, summary_json):
    allowed = {s.strip() for s in include_status.split(",")}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    summary = {}
    for ax, target in zip(axes, TARGETS):
        if not (Path(OUT[target]) / "_y_true_t.npy").exists():
            click.echo(f"[warn] {target}: no prediction cache yet, panel left empty", err=True)
            ax.set_title(LABEL[target], fontsize=12, loc="left")
            continue
        cur = _curves(target, s_p, allowed)
        summary[target] = {"n_eff": cur["n_eff"]}
        for model in MODEL_ORDER:
            c = H.MODEL_COLORS.get(model, "gray")
            for role, ls, z in (("baseline", "--", 1.5), ("al", "-", 2.5)):
                seeds = cur[model][role]
                if len(seeds) < min_seeds:
                    continue
                g, m, e = _mean_on_grid(seeds)
                ax.plot(g, m, color=c, linestyle=ls, linewidth=1.8, zorder=z)
                ax.fill_between(g, m - e, m + e, color=c, alpha=0.2, linewidth=0, zorder=z - 0.5)
                summary[target].setdefault(model, {})[role] = {
                    "n_seeds": len(seeds), "first": float(m[0]), "final": float(m[-1]),
                    "final_L": float(g[-1])}
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
        ax.set_xlabel(r"Dataset size $|L|$ [$10^3$]", fontsize=12)
        ax.set_title(LABEL[target], fontsize=12, loc="left")
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=10)
    axes[0].set_ylabel("Proximity-weighted class accuracy", fontsize=12)
    handles = [Line2D([0], [0], color=H.MODEL_COLORS[m], lw=2.4, label=H.MODEL_DISPLAY[m])
               for m in MODEL_ORDER]
    handles += [Line2D([0], [0], color="black", lw=1.6, label="active learning"),
                Line2D([0], [0], color="black", lw=1.6, ls="--", label="random baseline")]
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    # Above the panels: inside (b) it covers the Deep GP, Exact GP and TabPFN curves.
    fig.legend(handles=handles, fontsize=9, ncol=len(handles), loc="upper center",
               bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=1.2, handlelength=2.0)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    click.echo(f"wrote {out}")
    if summary_json:
        Path(summary_json).write_text(json.dumps(summary, indent=1))
    for t, d in summary.items():
        click.echo(f"{t}: effective eval size {d['n_eff']:.0f}")
        for m, r in d.items():
            if isinstance(r, dict):
                click.echo("  " + m.ljust(12) + "  ".join(
                    f"{k}: {v['first']:.3f}->{v['final']:.3f} at |L|={v['final_L']:.0f} "
                    f"({v['n_seeds']} seeds)" for k, v in r.items()))


if __name__ == "__main__":
    main()
