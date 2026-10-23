"""Two-panel verdict-accuracy figure: regression picks vs classification-entropy arms.

The companion of plot_hits_per_desired_heads.py with the same layout:
left, the best regression cell per surrogate; right, every surrogate run
with the classification head and the mean-probability-entropy acquisition.
The y axis is the accuracy of the verdict (target above or below the
threshold) on the run's held-out static-random evaluation set, read from
the ``accuracy_trajectory.json`` cache every AL driver writes, so no model
is re-evaluated here. Solid curves are the active learner, dashed curves
the random-selection baseline of the same architecture trained alongside
it on an equal-sized random sample. The x axis is the labelled-set size
|L| = train + validation; seeds are interpolated onto a common |L| grid
and a curve ends where fewer than ``--min-seeds-axis`` seeds remain.

Usage
-----
    python scripts/plot_accuracy_heads.py \
        --output /path/to/accuracy_heads.png \
        --summary-json /path/to/accuracy_heads.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
for p in (_REPO_ROOT, _REPO_ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_hit_rate_trajectories_multiseed as H  # noqa: E402
from plot_hits_per_desired_heads import REG_PICKS, CLS_PICKS, parse_picks, GRID_POINTS, SMOOTH, smooth_rows, parse_axis_override, axis_min  # noqa: E402


def _series(run_dir: Path, role: str, dataset: str) -> dict[int, float]:
    """{iteration: accuracy} from the run's cache (same reading as plot_headtest_accuracy)."""
    p = run_dir / "accuracy_trajectory.json"
    if not p.exists():
        return {}
    d = json.load(open(p))
    out = {}
    for key, entry in d.items():
        if not str(key).isdigit() or not isinstance(entry, dict):
            continue
        v = (entry.get(role) or {}).get(dataset)
        if v is None:
            continue
        v = float(v)
        if v == v:
            out[int(key)] = v
    return out


def _labelled_sizes(run_dir: Path) -> list[int]:
    state = torch.load(run_dir / "state.pt", weights_only=False, map_location="cpu")
    n_tr = [int(x) for x in (state.get("al_n_train") or [])]
    n_va = [int(x) for x in (state.get("al_n_val") or [])]
    return [n_tr[i] + (n_va[i] if i < len(n_va) else 0) for i in range(len(n_tr))]


def _run_curves(run_dir: str, dataset: str):
    """((L, acc, iters) for AL, same for baseline) or None."""
    rd = Path(run_dir)
    L = _labelled_sizes(rd)
    out = []
    for role in ("al", "baseline"):
        ser = _series(rd, role, dataset)
        pts = [(L[i - 1], a, i) for i, a in sorted(ser.items()) if 1 <= i <= len(L)]
        if not pts:
            return None
        Ls, As, Is = map(np.asarray, zip(*pts))
        out.append((Ls.astype(float), As.astype(float), Is.astype(int)))
    return tuple(out)


def _on_grid(curves, min_seeds):
    lo = max(c[0][0] for c in curves)
    hi = max(c[0][-1] for c in curves)
    grid = np.linspace(lo, hi, GRID_POINTS)
    Y = np.full((len(curves), GRID_POINTS), np.nan)
    for k, (L, A, _) in enumerate(curves):
        inside = (grid >= L[0]) & (grid <= L[-1])
        Y[k, inside] = np.interp(grid[inside], L, A)
    keep = np.sum(~np.isnan(Y), axis=0) >= min_seeds
    return grid[keep], Y[:, keep]


def _final(curves):
    it_max = int(max(c[2][-1] for c in curves))
    reached = [c for c in curves if c[2][-1] == it_max]
    vals = np.array([c[1][-1] for c in reached])
    at40 = [c[1][list(c[2]).index(40)] for c in curves if 40 in list(c[2])]
    return {
        "final_iteration": it_max, "n_seeds_at_final": len(reached),
        "final_mean": float(vals.mean()),
        "final_sem": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else None,
        "L_final_mean": float(np.mean([c[0][-1] for c in reached])),
        "at_iter_40": float(np.mean(at40)) if at40 else None,
    }


def _panel(ax, df, picks, dataset, min_seeds, min_seeds_axis, uncertainty,
           with_baseline=True):
    summary = {}
    for model, (s, w) in picks.items():
        sub = df[(df["model"] == model) & (df["strategy"] == s) & (df["warm_start"] == w)]
        al, base = [], []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                c = _run_curves(run_dir, dataset)
            except Exception as exc:  # noqa: BLE001
                click.echo(f"[warn] skip {run_dir}: {exc}", err=True)
                continue
            if c is not None:
                al.append(c[0]); base.append(c[1])
        if len(al) < min_seeds:
            click.echo(f"[warn] {model}/{s}/{w}: {len(al)} seeds, skipped", err=True)
            continue
        color = H.MODEL_COLORS.get(model, "gray")
        for curves, ls in (((al, "-"), (base, "--")) if with_baseline else ((al, "-"),)):
            grid, Y = _on_grid(curves, axis_min(model, min_seeds_axis))
            if len(grid) == 0:
                click.echo(f"[warn] no point with {min_seeds_axis} seeds; curve skipped", err=True)
                continue
            H._draw_curve(ax, grid, smooth_rows(Y), color=color, linestyle=ls, marker=None,
                          label=None, uncertainty=uncertainty,
                          fill_alpha=0.15 if ls == "-" else 0.08)
        summary[model] = {"strategy": s, "warm": w, "n_seeds": len(al),
                          "al": _final(al), "baseline": _final(base)}
    return summary


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--dataset", default="static_random", show_default=True,
              help="Evaluation set in the cache: static_random, mcmc, train or val.")
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--min-seeds-axis", default=2, show_default=True)
@click.option("--include-status", default="completed,timeout,running,missing",
              show_default=True)
@click.option("--uncertainty", default="sem", show_default=True)
@click.option("--reg-picks", default="", show_default=True,
              help="Override regression picks: model:strategy:warm[,...].")
@click.option("--cls-picks", default="", show_default=True,
              help="Override classification picks: model:strategy:warm[,...].")
@click.option("--model-tag", default="", show_default=True)
@click.option("--ylabel", default=r"Verdict accuracy ($\Omega h^{2}\gtrless0.12$)", show_default=True)
@click.option("--ymin", default=0.90, show_default=True)
@click.option("--legend-bottom", default=0.04, show_default=True)
@click.option("--seeds-axis-override", default="", show_default=True,
              help="Per-model --min-seeds-axis, e.g. tabpfn_expr:4 (full model name).")
@click.option("--smooth", default=0, show_default=True,
              help="Rolling-average window in grid points (of 300) applied to each seed.")
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(manifest, dataset, min_seeds, min_seeds_axis, include_status,
         uncertainty, reg_picks, cls_picks, model_tag, ylabel, ymin, legend_bottom, seeds_axis_override, smooth, output,
         summary_json):
    SMOOTH["window"] = smooth
    parse_axis_override(seeds_axis_override)
    reg_picks = parse_picks(reg_picks, REG_PICKS, model_tag)
    cls_picks = parse_picks(cls_picks, CLS_PICKS, model_tag)
    df = pd.read_csv(manifest)
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    df = df[df["expected_run_dir"].map(
        lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists()
        and (Path(d) / "accuracy_trajectory.json").exists())]
    if df.empty:
        raise click.ClickException("no manifest rows with state.pt and an accuracy cache")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panels = [("regression", reg_picks), ("classification", cls_picks)]
    summary = {"dataset": dataset, "x_axis": "labelled-set size (train + validation)"}
    for ax, (name, picks) in zip(axes, panels):
        summary[name] = _panel(ax, df, picks, dataset, min_seeds, min_seeds_axis, uncertainty)
        ax.set_xlabel(r"Dataset size $|L|$", fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.grid(alpha=0.3)
        ax.text(0.98, 0.02, name, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.8))
    axes[0].set_ylabel(ylabel, fontsize=14)
    for ax in axes:
        ax.set_ylim(ymin, 1.0)
    fig.tight_layout()

    models = [m for m in reg_picks if m in summary[panels[0][0]] or m in summary[panels[1][0]]]
    handles = [Line2D([0], [0], color=H.MODEL_COLORS.get(m, "gray"), lw=2.4,
                      label=H.MODEL_DISPLAY.get(m, m)) for m in models]
    handles += [Line2D([0], [0], color="black", linestyle="-", lw=1.6, label="active learning"),
                Line2D([0], [0], color="black", linestyle="--", lw=1.6, label="random baseline")]
    # Lower right of the left panel: the panel tag sits there too, so anchor
    # the legend just above it; the curves are high in that corner.
    axes[0].legend(handles=handles, loc="lower right",
                   bbox_to_anchor=(0.985, legend_bottom + 0.07), ncol=2,
                   fontsize=9.5, frameon=True, framealpha=0.9,
                   columnspacing=1.0, handlelength=2.0)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    H._enforce_marker_policy(fig)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out}")
    if summary_json:
        json.dump(summary, open(summary_json, "w"), indent=2)
        click.echo(f"wrote {summary_json}")
    for name, _ in panels:
        for m, v in summary[name].items():
            a, b = v["al"], v["baseline"]
            click.echo(f"  {name:26} {m:12} AL {a['final_mean']:.4f}±{(a['final_sem'] or 0):.4f}"
                       f" | base {b['final_mean']:.4f} at it.{a['final_iteration']}"
                       f" ({a['n_seeds_at_final']}/{v['n_seeds']}), |L|={a['L_final_mean']:.0f};"
                       f" it.40 AL {a['at_iter_40']}")


if __name__ == "__main__":
    main()
