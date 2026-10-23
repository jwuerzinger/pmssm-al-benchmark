"""One-panel results figures: the best configuration per surrogate.

Main-body counterparts of the two-panel "heads" figures (which move to the
appendix). One figure per (target, metric), each a single panel holding every
surrogate once, in its best configuration (user's choice, 2026-10-13):

  relic density  Deep GP, Exact GP: regression (entropy_batch, warm)
                 DNN, Transformer:  classification (cls_entropy, cold)
                 TabPFN:            regression (tol_only_random)
  LHC exclusion  Deep GP:           regression (entropy_batch, warm)
                 DNN, Exact GP, Transformer: classification (cls_entropy)
                 TabPFN:            regression (top_k; 4 seeds, seed 1 stalled)

Metrics:
  yield     in-band yield per simulator call vs labelled-set size, with the
            random-scan line (and emcee MCMC on the relic density);
  accuracy  verdict accuracy on the static random evaluation set vs
            labelled-set size, active learner only;
  support   in-band support covered vs budget / reference dataset size, with the
            random-scan prefix curve, the raw emcee chains on the relic density
            and GPU-hour markers.

Curves are drawn by the heads scripts' own panel functions, so metric, grid,
seed rule (--min-seeds-axis) and smoothing match the appendix figures.

Usage
-----
    python scripts/plot_best_config.py --target DMRD --metric yield --output .../best_yield.png
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_hit_rate_trajectories_multiseed as H                       # noqa: E402
import plot_hits_per_desired_heads as HD                                # noqa: E402
import plot_accuracy_heads as AC                                        # noqa: E402
import plot_support_heads as SU                                         # noqa: E402
from coverage_saturation import _cells                                  # noqa: E402
from plot_support_efficiency import AXES, _band_mask, _ref_curve        # noqa: E402
from plot_prelim_support import _support_from_cache                     # noqa: E402
from pmssm.config import PARAM_ORDER                                    # noqa: E402

A = "/ptmp/jwuerzin/analysis"
TARGETS = {
    "DMRD": dict(
        tag="", manifest=f"{A}/all_runs/sweep_manifest_paper.csv",
        baseline=f"{A}/all_runs/random_baseline_prevalence.json",
        # MCMC reference lines dropped from the yield and support figures (user
        # decision 2026-10-21); was mcmc_yield=904840 / 48305152 and
        # mcmc_support_json=f"{A}/all_runs/support_efficiency_paper.json".
        mcmc_yield=0.0, mcmc_support_json="",
        pool="/ptmp/jwuerzin/data/18387358", cells=213, axis_override="",
        acc_ylim=(0.92, 1.0), acc_symbol=r"\Omega h^{2}\gtrless0.12",  # own range: 0.75-1 unreadable
        picks={"transformer": ("cls_entropy", "cold"), "dnn": ("cls_entropy", "cold"),
               "deep_gp": ("entropy_batch", "warm"), "exact_gp": ("entropy_batch", "warm"),
               "tabpfn": ("tol_only_random", "tabpfn")}),
    "ExpR": dict(
        tag="_expr", manifest=f"{A}/expr_runs/sweep_manifest_paper.csv",
        baseline=f"{A}/expr_runs/random_baseline_prevalence.json",
        mcmc_yield=0.0, mcmc_support_json="",
        pool="/ptmp/jwuerzin/data/260804", cells=1067, axis_override="tabpfn_expr:4",
        acc_ylim=(0.75, 1.0), acc_symbol=r"r\gtrless1",
        picks={"transformer": ("cls_entropy", "cold"), "dnn": ("cls_entropy", "cold"),
               "deep_gp": ("entropy_batch", "warm"), "exact_gp": ("cls_entropy", "warm"),
               "tabpfn": ("top_k", "tabpfn")}),
}
MODEL_ORDER = ["transformer", "dnn", "deep_gp", "exact_gp", "tabpfn"]
HEAD = {"cls_entropy": "classification", "entropy_batch": "regression",
        "tol_only_random": "regression", "top_k": "regression"}


def _ratio_panel(axr, df, picks, min_seeds_axis):
    """AL / random-baseline verdict accuracy per model, as the ratio of the
    seed-averaged curves (the convention chosen for the MSE ratio strip), on
    the AL curve's |L| grid; band from the two SEMs in quadrature."""
    for model, (s, w) in picks.items():
        sub = df[(df["model"] == model) & (df["strategy"] == s) & (df["warm_start"] == w)]
        al, base = [], []
        for d in sub["expected_run_dir"].dropna():
            try:
                c = AC._run_curves(d, "static_random")
            except Exception:  # noqa: BLE001
                continue
            if c is not None:
                al.append(c[0]); base.append(c[1])
        if len(al) < 2:
            continue
        grid, Ya = AC._on_grid(al, HD.axis_min(model, min_seeds_axis))
        if len(grid) == 0:
            continue
        Yb = np.full_like(Ya, np.nan)
        for k, (L, A, _) in enumerate(base):
            inside = (grid >= L[0]) & (grid <= L[-1])
            Yb[k, inside] = np.interp(grid[inside], L, A)
        Ya, Yb = HD.smooth_rows(Ya), HD.smooth_rows(Yb)
        ma, mb = np.nanmean(Ya, 0), np.nanmean(Yb, 0)
        na = np.sum(np.isfinite(Ya), 0); nb = np.sum(np.isfinite(Yb), 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            sa = np.nanstd(Ya, 0, ddof=1) / np.sqrt(na)
            sb = np.nanstd(Yb, 0, ddof=1) / np.sqrt(nb)
            r = ma / mb
            dr = r * np.sqrt((sa / ma) ** 2 + (sb / mb) ** 2)
        col = H.MODEL_COLORS.get(model, "gray")
        axr.plot(grid, r, color=col, lw=1.5)
        axr.fill_between(grid, r - dr, r + dr, color=col, alpha=0.15, lw=0)
    axr.axhline(1.0, color="black", linestyle=":", linewidth=1.2)


EFF_YLIM = (0.5, 15)  # lower-panel range, common to both targets


def _efficiency_panel(axr, df, picks, min_seeds_axis):
    """Data-efficiency factor (2026-10-15; combined forward/reverse below). For every random-baseline size |L| on
    the grid, the factor is |L| divided by the smallest |L'| at which the active
    learner's accuracy first reaches the baseline accuracy at |L|. Both curves
    are the smoothed seed means drawn in the top panel (same grid, same seed
    rule). Undefined, and left blank, where the active learner never reaches
    the baseline accuracy. The x axis is therefore the random-baseline size."""
    for model, (s, w) in picks.items():
        sub = df[(df["model"] == model) & (df["strategy"] == s) & (df["warm_start"] == w)]
        al, base = [], []
        for d in sub["expected_run_dir"].dropna():
            try:
                c = AC._run_curves(d, "static_random")
            except Exception:  # noqa: BLE001
                continue
            if c is not None:
                al.append(c[0]); base.append(c[1])
        if len(al) < 2:
            continue
        need = HD.axis_min(model, min_seeds_axis)
        ga, Ya = AC._on_grid(al, need)
        gb, Yb = AC._on_grid(base, need)
        if len(ga) == 0 or len(gb) == 0:
            continue
        ma = np.nanmean(HD.smooth_rows(Ya), 0)
        mb = np.nanmean(HD.smooth_rows(Yb), 0)
        # One definition per size x, chosen by which curve is ahead there
        # (2026-10-15), so the two cannot interleave where they cross:
        #   AL ahead:  x / (first AL size reaching the baseline accuracy at x)
        #   AL behind: (first baseline size reaching the AL accuracy at x) / x
        # Both are 1 at a crossing. Undefined (NaN) where the needed curve
        # never reaches the accuracy.
        lo, hi = max(ga[0], gb[0]), min(ga[-1], gb[-1])
        x = gb[(gb >= lo) & (gb <= hi)]
        a_x, b_x = np.interp(x, ga, ma), np.interp(x, gb, mb)
        y = np.full(len(x), np.nan)
        for k, xv in enumerate(x):
            if a_x[k] >= b_x[k]:
                hit = np.nonzero(ma >= b_x[k])[0]
                if len(hit):
                    y[k] = xv / ga[hit[0]]
            else:
                hit = np.nonzero(mb >= a_x[k])[0]
                if len(hit):
                    y[k] = gb[hit[0]] / xv
        # Beyond the overlap only the forward definition can extend (baseline
        # longer than AL is impossible to invert), so keep the overlap only.
        # Rolling mean of log(saving), same NaN-aware window as the top panel,
        # so a factor 2 up and a factor 2 down average out on the log axis.
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.exp(HD.smooth_rows(np.log(y)[None, :])[0])
        col = H.MODEL_COLORS.get(model, "gray")
        axr.plot(x, y, color=col, lw=1.5)
    axr.axhline(1.0, color="black", linestyle=":", linewidth=1.2)


def _manifest(path):
    df = pd.read_csv(path)
    df = df[df["status"].isin({"completed", "timeout", "running", "missing"})]
    return df[df["expected_run_dir"].map(lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists())]


@click.command()
@click.option("--target", type=click.Choice(list(TARGETS)), required=True)
@click.option("--metric", type=click.Choice(["yield", "accuracy", "support"]), required=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--smooth", default=15, show_default=True)
@click.option("--min-seeds-axis", default=5, show_default=True)
@click.option("--compute-marks", default="0.3,3,30", show_default=True)
@click.option("--pool-cache-dir", default=f"{A}/pool_cache", show_default=True)
@click.option("--lower-panel", type=click.Choice(["ratio", "efficiency"]), default="efficiency",
              show_default=True, help="With --ratio-panel: accuracy ratio, or data-efficiency factor.")
@click.option("--ratio-panel", is_flag=True, default=False,
              help="Accuracy only: AL curves on top, AL / random-baseline ratio below.")
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(target, metric, tolerance, smooth, min_seeds_axis, compute_marks, pool_cache_dir,
         lower_panel, ratio_panel,
         output, summary_json):
    cfg = TARGETS[target]
    HD.SMOOTH["window"] = smooth
    HD.parse_axis_override(cfg["axis_override"])
    picks = {m + cfg["tag"]: cfg["picks"][m] for m in MODEL_ORDER}
    df = _manifest(cfg["manifest"])
    base = json.load(open(cfg["baseline"]))
    true_val = float(base["true_value"])
    H._DESIRED_P_VALID = float(base["p_valid"]["p_valid"])
    rnd = float(base["prevalence_per_attempt"][f"{tolerance:.4f}"])
    marks = [float(v) for v in compute_marks.split(",") if v.strip()]

    # Single-column size (2026-10-13): drawn at 4.6 x 3.5 in, printed at the
    # ~3.4 in column width, so 12 pt labels come out at ~9 pt.
    axr = None
    if metric == "accuracy" and ratio_panel:
        fig, (ax, axr) = plt.subplots(2, 1, figsize=(4.6, 4.0), sharex=True,
                                      gridspec_kw=dict(height_ratios=[2.1, 1], hspace=0.06))
    else:
        fig, ax = plt.subplots(figsize=(4.6, 3.5))
    refs = []  # (label, linestyle) of reference lines drawn
    if metric == "yield":
        summary = HD._panel_curves(ax, df, picks, true_val, tolerance, 2, min_seeds_axis, "sem")
        ax.axhline(rnd, color="black", linestyle=":", linewidth=1.4)
        refs.append(("random scan", ":"))
        if cfg["mcmc_yield"] > 0:
            ax.axhline(cfg["mcmc_yield"], color="black", linestyle="-.", linewidth=1.4)
            refs.append(("emcee MCMC", "-."))
        ax.set_ylim(0, None)
        ax.set_ylabel("In-band yield per simulator call", fontsize=12)
    elif metric == "accuracy":
        dfa = df[df["expected_run_dir"].map(lambda d: (Path(d) / "accuracy_trajectory.json").exists())]
        # Random baselines included (user request 2026-10-13): dashed, same colour.
        summary = AC._panel(ax, dfa, picks, "static_random", 2, min_seeds_axis, "sem",
                            with_baseline=True)  # random baselines back in (2026-10-15)
        refs += [("active learning", "-"), ("random baseline", "--")]
        # Dashed baselines behind every solid AL curve (2026-10-15): the panel
        # draws model by model, so later baselines would cover earlier AL lines.
        for ln in ax.get_lines():
            ln.set_zorder(1.5 if ln.get_linestyle() == "--" else 2.5)
        if axr is not None:
            (_ratio_panel if lower_panel == "ratio" else _efficiency_panel)(
                axr, dfa, picks, min_seeds_axis)
        ax.set_ylim(*cfg["acc_ylim"])
        ax.set_ylabel("Class accuracy", fontsize=12)  # threshold stated in the caption
    else:
        ax_idx = [PARAM_ORDER.index(a) for a in AXES]
        edges, tmap, n_target, oX, oY, n_rows, _ = _support_from_cache(
            cfg["pool"], target, pool_cache_dir, tolerance, 12, 20, true_val, "in")
        if n_target != cfg["cells"]:
            raise click.ClickException(f"support has {n_target} cells, expected {cfg['cells']}")
        S = dict(edges=edges, tmap=tmap, n_target=n_target, n_total=n_rows)
        summary = SU._panel(ax, df, picks, "budget", S, ax_idx, true_val, tolerance, "in",
                            2, min_seeds_axis, "sem", marks)
        rcells = np.where(_band_mask(oY, true_val, tolerance, "in"), _cells(oX, edges), -1)
        rf, rc = _ref_curve(rcells, tmap, n_target, n_rows, 40, 2000)
        ax.plot(rf, rc, color="black", linestyle=":", linewidth=1.4)
        refs.append(("random scan", ":"))
        if cfg["mcmc_support_json"]:
            pan = json.load(open(cfg["mcmc_support_json"]))["panels"]["pool"]
            assert pan["n_target_cells"] == n_target and pan["dataset_calls"] == n_rows
            f = np.asarray(pan["reference"]["mcmc"]["fraction"])
            c = np.asarray(pan["reference"]["mcmc"]["coverage"])
            keep = (f <= 1.0) & (f >= 2000 / n_rows)
            ax.plot(f[keep], c[keep], color="black", linestyle="-.", linewidth=1.4)
            refs.append(("emcee MCMC", "-."))
        ax.set_xscale("log")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(f"Support covered ({n_target} cells)", fontsize=12)

    if metric == "support":
        ax.set_xlabel("Budget / reference dataset size", fontsize=12)
        ax.grid(alpha=0.3, which="both")
    else:
        # Thousands on the |L| axis: full numbers crowd a single-column panel.
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
        ax.set_xlabel(r"Dataset size $|L|$ [$10^3$]", fontsize=12)
        ax.grid(alpha=0.3)
    ax.tick_params(axis="both", labelsize=10)
    if axr is not None:
        # The shared |L| axis belongs to the ratio panel underneath.
        axr.set_xlabel(ax.get_xlabel(), fontsize=12)
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelbottom=False)
        axr.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
        axr.tick_params(axis="both", labelsize=10)
        axr.grid(alpha=0.3)
        axr.set_ylabel("AL / random" if lower_panel == "ratio" else "Equiv.-data ratio", fontsize=12)
        # Ticks at 0.95, 1.00, 1.05 on the common 0.95-1.06 range.
        if lower_panel == "ratio":
            axr.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
            axr.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
            axr.set_ylim(0.95, 1.06)  # common to both targets
        else:
            # Log axis common to both targets; ticks at 0.5, 1, 2, 5, 10.
            axr.set_yscale("log")
            axr.set_ylim(*EFF_YLIM)
            axr.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.5, 1, 2, 5, 10]))
            axr.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            axr.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        fig.align_ylabels([ax, axr])

    # Before the legend: the policy also strips markers from legend proxies.
    H._enforce_marker_policy(fig)
    shown = [m for m in MODEL_ORDER if m + cfg["tag"] in summary]
    handles = [Line2D([0], [0], color=H.MODEL_COLORS[m], lw=2.4,
                      label=H.MODEL_DISPLAY[m]) for m in shown]
    handles += [Line2D([0], [0], color="black", linestyle=ls, lw=1.4, label=l) for l, ls in refs]
    if metric == "support":
        handles += [Line2D([0], [0], linestyle="none", marker=SU.MARK_SHAPES[k], markersize=7,
                           markerfacecolor="white", markeredgecolor="black", label=f"{h:g} GPU h")
                    for k, h in enumerate(marks)]
    # Inside the axes (user request). Accuracy and support leave the lower right
    # free; the yield curves fill every corner, so those get headroom on top
    # and the legend sits there.
    if metric == "yield":
        # Common range for both targets (panels a and b of Fig. 1), with
        # headroom for the legend above the curves.
        ax.set_ylim(0, 0.19)
        loc = "upper left"
    else:
        loc = "lower right"
    # Support: one column at the far right, below both reference curves, which
    # are above 0.85 there; two columns reach left into the random-scan curve.
    ax.legend(handles=handles, loc=loc, fontsize=8.5, frameon=True, framealpha=0.92,
              ncol=1 if metric == "support" else 2, handlelength=1.8, columnspacing=0.9, borderpad=0.4, labelspacing=0.3)

    fig.tight_layout()
    # Same left/right margins for every figure, so axes line up across panels;
    # the ratio figure also needs its fixed vertical spacing back.
    fig.subplots_adjust(left=0.17, right=0.97)
    if axr is not None:
        fig.subplots_adjust(bottom=0.13, top=0.97, hspace=0.06)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Fixed canvas, no tight bbox (2026-10-14): every figure is exactly 4.6 in
    # (920 px) wide, so at equal printed width all share one scale in x and y.
    fig.savefig(out, dpi=200)
    plt.close(fig)
    click.echo(f"wrote {out}")
    if summary_json:
        json.dump({"target": target, "metric": metric, "picks": {m: list(v) for m, v in picks.items()},
                   "smooth": smooth, "min_seeds_axis": min_seeds_axis, "summary": summary},
                  open(summary_json, "w"), indent=2, default=str)


if __name__ == "__main__":
    main()
