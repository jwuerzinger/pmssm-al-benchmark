"""Two-panel in-band-yield figure: regression picks vs classification-entropy arms.

Left panel: the best regression cell per surrogate (the paper's picks).
Right panel: every surrogate run with the classification head and the
mean-probability-entropy acquisition (``cls_entropy``).

The y axis is the in-band yield per simulator call (``hits_per_desired`` in
the main plotter's terms: in-band models in the labelled set divided by the
number of models requested from the generator, the initial chunk charged in
attempt units). The x axis is the labelled-set size |L| = train + validation,
the same definition as plot_compute_vs_dataset.py. Seeds reach different |L|
at the same iteration, so each seed's trajectory is interpolated onto a
common |L| grid before the mean and band are taken; a curve runs to the
furthest seed's |L|.

Nothing here re-derives the metric: ``_hits_per_desired_trajectory`` and the
colour/style helpers are imported from the paper's own plotter. The random
and MCMC reference lines are horizontal, per attempt: the random line is the
pool prevalence deflated by p_valid, read from the JSON that the main plotter
writes, and the MCMC line is the number of distinct in-band points per
proposal of the emcee reference.

Usage
-----
    python scripts/plot_hits_per_desired_heads.py \
        --output /path/to/hits_per_desired_heads.png \
        --summary-json /path/to/hits_per_desired_heads.json
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

# The paper's regression picks (DEFAULT_AL_PICKS without the matched DNN).
REG_PICKS = {
    "transformer": ("entropy_batch", "cold"),
    "dnn":         ("entropy_batch", "cold"),
    "deep_gp":     ("entropy_batch", "warm"),
    "exact_gp":    ("entropy_batch", "warm"),
    "tabpfn":      ("top_k",         "tabpfn"),
}
# Every surrogate that ran the classification head with entropy acquisition.
CLS_PICKS = {
    "transformer": ("cls_entropy", "cold"),
    "dnn":         ("cls_entropy", "cold"),
    "deep_gp":     ("cls_entropy", "warm"),
    "exact_gp":    ("cls_entropy", "warm"),
}
# emcee reference: distinct in-band (M_1, M_2, mu) points over every proposal
# of the four ensembles (904 840 / 48 305 152), see yield_comparison.json and
# the chain lengths in diag_resumed/diagnostics.txt.
MCMC_YIELD_PER_PROPOSAL = 904840 / 48305152
GRID_POINTS = 300

# Rolling-average window in grid points, shared by every heads script (set from
# --smooth in each main). 0 or 1 leaves the curves as they are.
SMOOTH = {"window": 0}

# Per-model override of --min-seeds-axis, e.g. {"tabpfn_expr": 4} while one
# seed is stalled and excluded from the manifest. Set from --seeds-axis-override.
AXIS_OVERRIDE: dict[str, int] = {}


def parse_axis_override(spec: str) -> None:
    AXIS_OVERRIDE.clear()
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        m, n = item.split(":")
        AXIS_OVERRIDE[m] = int(n)


def axis_min(model: str, default: int) -> int:
    return AXIS_OVERRIDE.get(model, default)


def smooth_rows(Y: np.ndarray, window: int | None = None) -> np.ndarray:
    """Centred, NaN-aware rolling mean along the grid of each seed row.

    Applied per seed before the mean and its band are formed, so the band is
    the spread of the smoothed seeds. The window is symmetric and shrinks at
    the ends of each row's finite range (2026-10-15): the first and last
    points are left unsmoothed and the window grows from there to its full
    width. A one-sided window at the start used to average only later points
    and pulled the active learner's first value above the baseline's,
    although both are trained on the same initial set. NaNs stay NaN.
    """
    w = SMOOTH["window"] if window is None else window
    if not w or w <= 1:
        return Y
    Y = np.atleast_2d(np.asarray(Y, dtype=float))
    h = int(w) // 2
    out = np.full_like(Y, np.nan)
    for r, row in enumerate(Y):
        ok = np.isfinite(row)
        if not ok.any():
            continue
        idx = np.nonzero(ok)[0]
        first, last = idx[0], idx[-1]
        for i in idx:
            k = min(h, i - first, last - i)
            seg = row[i - k:i + k + 1]
            out[r, i] = np.nanmean(seg)
    return out


def parse_picks(spec: str, default: dict, tag: str = "") -> dict:
    """Apply "model:strategy:warm,..." overrides to a picks dict, then append the
    model tag. Models absent from the spec keep their default cell."""
    picks = dict(default)
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        model, strat, warm = item.split(":")
        picks[model] = (strat, warm)
    return {m + tag: v for m, v in picks.items()}


def _run_curve(run_dir: str, true_val: float, tol: float):
    """(L, rate, iteration) arrays for one run; L = train + validation size."""
    run = H._load_run(run_dir)
    iters, rates = H._hits_per_desired_trajectory(run, true_val, tol)
    if not rates:
        return None
    state = torch.load(Path(run_dir) / "state.pt", weights_only=False,
                       map_location="cpu")
    n_tr = [int(x) for x in (state.get("al_n_train") or [])]
    n_va = [int(x) for x in (state.get("al_n_val") or [])]
    L, R, I = [], [], []
    for it, r in zip(iters, rates):
        i = int(it) - 1
        if i >= len(n_tr):
            break
        L.append(n_tr[i] + (n_va[i] if i < len(n_va) else 0))
        R.append(r)
        I.append(int(it))
    return np.asarray(L, float), np.asarray(R, float), np.asarray(I, int)


def _cell_on_grid(curves, min_seeds):
    """Interpolate each seed onto a common |L| grid; NaN outside a seed's range."""
    lo = max(c[0][0] for c in curves)
    hi = max(c[0][-1] for c in curves)
    grid = np.linspace(lo, hi, GRID_POINTS)
    Y = np.full((len(curves), GRID_POINTS), np.nan)
    for k, (L, R, _) in enumerate(curves):
        inside = (grid >= L[0]) & (grid <= L[-1])
        Y[k, inside] = np.interp(grid[inside], L, R)
    keep = np.sum(~np.isnan(Y), axis=0) >= min_seeds
    return grid[keep], Y[:, keep]


def _panel_curves(ax, df, picks, true_val, tol, min_seeds, min_seeds_axis,
                  uncertainty):
    summary = {}
    for model, (s, w) in picks.items():
        sub = df[(df["model"] == model) & (df["strategy"] == s)
                 & (df["warm_start"] == w)]
        curves = []
        for run_dir in sub["expected_run_dir"].dropna():
            try:
                c = _run_curve(run_dir, true_val, tol)
            except Exception as exc:  # noqa: BLE001
                click.echo(f"[warn] skip {run_dir}: {exc}", err=True)
                continue
            if c is not None:
                curves.append(c)
        if len(curves) < min_seeds:
            click.echo(f"[warn] {model}/{s}/{w}: {len(curves)} seeds, skipped",
                       err=True)
            continue
        grid, Y = _cell_on_grid(curves, axis_min(model, min_seeds_axis))
        if len(grid) == 0:
            click.echo(f"[warn] no point with {min_seeds_axis} seeds; curve skipped", err=True)
            continue
        H._draw_curve(ax, grid, smooth_rows(Y), color=H.MODEL_COLORS.get(model, "gray"),
                      linestyle="-", marker=None, label=None,
                      uncertainty=uncertainty)
        # "Final" = the cell's furthest iteration, averaged over the seeds that
        # reached it (a seed that stopped earlier must not drag the mean down).
        it_finals = np.array([I[-1] for _, _, I in curves])
        it_max = int(it_finals.max())
        reached = [c for c in curves if c[2][-1] == it_max]
        finals = np.array([R[-1] for _, R, _ in reached])
        L_finals = np.array([L[-1] for L, _, _ in reached])
        at40 = [R[list(I).index(40)] for _, R, I in curves if 40 in list(I)]
        summary[model] = {
            "strategy": s, "warm": w, "n_seeds": len(curves),
            "final_iteration": it_max, "n_seeds_at_final": len(reached),
            "final_mean": float(np.mean(finals)),
            "final_sem": float(np.std(finals, ddof=1) / np.sqrt(len(finals))) if len(finals) > 1 else None,
            "final_iterations_per_seed": it_finals.tolist(),
            "L_final_per_seed": L_finals.tolist(),
            "L_final_mean": float(L_finals.mean()),
            "yield_at_iter_40": float(np.mean(at40)) if at40 else None,
        }
    return summary


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv",
              show_default=True)
@click.option("--baseline-json",
              default="/ptmp/jwuerzin/analysis/all_runs/random_baseline_prevalence.json",
              show_default=True,
              help="Written by plot_hit_rate_trajectories_multiseed.py; supplies "
                   "p_valid and the per-attempt random prevalence.")
@click.option("--mcmc-yield", default=MCMC_YIELD_PER_PROPOSAL, type=float,
              show_default=True, help="MCMC distinct in-band points per proposal.")
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--min-seeds", default=2, show_default=True)
@click.option("--min-seeds-axis", default=1, show_default=True)
@click.option("--include-status", default="completed,timeout,running,missing",
              show_default=True,
              help="Manifest statuses to keep; a row also needs a state.pt.")
@click.option("--uncertainty", default="sem", show_default=True)
@click.option("--reg-picks", default="", show_default=True,
              help="Override regression picks: model:strategy:warm[,...].")
@click.option("--cls-picks", default="", show_default=True,
              help="Override classification picks: model:strategy:warm[,...].")
@click.option("--model-tag", default="", show_default=True,
              help="Suffix the manifest appends to model names for a variant sweep (e.g. _expr).")
@click.option("--target-label", default="Ω", show_default=True,
              help="Symbol in the tolerance tag, e.g. r for the exclusion ratio.")
@click.option("--legend-bottom", default=0.13, show_default=True,
              help="Axes-fraction height of the legend block's bottom edge "
                   "(keeps it above the reference lines).")
@click.option("--seeds-axis-override", default="", show_default=True,
              help="Per-model --min-seeds-axis, e.g. tabpfn_expr:4 (full model name).")
@click.option("--smooth", default=0, show_default=True,
              help="Rolling-average window in grid points (of 300) applied to each seed.")
@click.option("--output", required=True)
@click.option("--summary-json", default="", help="Where to dump the final values.")
def main(manifest, baseline_json, mcmc_yield, tolerance, min_seeds,
         min_seeds_axis, include_status, uncertainty, reg_picks, cls_picks, model_tag,
         target_label, legend_bottom, seeds_axis_override, smooth, output, summary_json):
    SMOOTH["window"] = smooth
    parse_axis_override(seeds_axis_override)
    reg_picks = parse_picks(reg_picks, REG_PICKS, model_tag)
    cls_picks = parse_picks(cls_picks, CLS_PICKS, model_tag)
    H._TARGET_LABEL_MATH = target_label
    df = pd.read_csv(manifest)
    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = df[df["status"].isin(allowed)]
    df = df[df["expected_run_dir"].map(
        lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists())]
    if df.empty:
        raise click.ClickException("no manifest rows with a state.pt")

    base = json.load(open(baseline_json))
    true_val = float(base["true_value"])
    p_valid = float(base["p_valid"]["p_valid"])
    prev = float(base["prevalence_per_attempt"][f"{tolerance:.4f}"])
    # Charge the initial chunk in attempt units, exactly as the main plotter does.
    H._DESIRED_P_VALID = p_valid
    click.echo(f"[baseline] p_valid={p_valid:.4f}, random per attempt at "
               f"{tolerance:.0%}: {prev:.5f}; MCMC per proposal: {mcmc_yield:.5f}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    H._setup_axes(axes, [tolerance, tolerance], true_val, "",
                  "In-band yield per simulator call")
    for ax in axes:
        ax.set_xlabel(r"Dataset size $|L|$", fontsize=14)
        # The band definition goes in the caption (2026-10-13), not in the plot.
        for t in list(ax.texts):
            t.remove()
    panels = [("regression", reg_picks), ("classification", cls_picks)]
    summary = {"tolerance": tolerance, "p_valid": p_valid,
               "random_per_attempt": prev, "mcmc_per_proposal": mcmc_yield,
               "x_axis": "labelled-set size (train + validation)"}
    for ax, (name, picks) in zip(axes, panels):
        summary[name] = _panel_curves(ax, df, picks, true_val, tolerance,
                                      min_seeds, min_seeds_axis, uncertainty)
        ax.axhline(prev, color="black", linestyle=":", linewidth=1.4)
        if mcmc_yield > 0:
            ax.axhline(mcmc_yield, color="black", linestyle="-.", linewidth=1.4)
        ax.text(0.02, 0.98, name, transform=ax.transAxes, ha="left", va="top",
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8", alpha=0.8))
    axes[1].set_ylabel("")

    models = [m for m in reg_picks if m in summary[panels[0][0]]
              or m in summary[panels[1][0]]]
    color_handles = [Line2D([0], [0], color=H.MODEL_COLORS.get(m, "gray"), lw=2.4,
                            label=H.MODEL_DISPLAY.get(m, m)) for m in models]
    style_handles = [
        Line2D([0], [0], color="black", linestyle=":", lw=1.4, label="random scan"),
    ]
    if mcmc_yield > 0:
        style_handles.append(Line2D([0], [0], color="black", linestyle="-.", lw=1.4,
                                    label="emcee MCMC"))
    # One untitled legend in the lower right of the right panel, above the two
    # reference lines, where no curve passes: the curves are high on the right
    # of that panel and low only at its left edge.
    for ax in axes:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, max(ymax, 0.05) * 1.05)
    fig.tight_layout()
    axes[1].legend(handles=color_handles + style_handles, loc="lower right",
                   bbox_to_anchor=(0.985, legend_bottom), ncol=2, fontsize=9.5,
                   frameon=True, framealpha=0.9, columnspacing=1.0,
                   handlelength=2.0)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    H._enforce_marker_policy(fig)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {output}")
    if summary_json:
        with open(summary_json, "w") as fh:
            json.dump(summary, fh, indent=2)
        click.echo(f"wrote {summary_json}")
    for name, _ in panels:
        for m, v in summary[name].items():
            click.echo(f"  {name:26} {m:12} final={v['final_mean']:.4f}"
                       f"±{(v['final_sem'] or 0):.4f} at it.{v['final_iteration']}"
                       f" ({v['n_seeds_at_final']}/{v['n_seeds']} seeds), |L|={v['L_final_mean']:.0f}"
                       f"  it.40={v['yield_at_iter_40']}")


if __name__ == "__main__":
    main()
