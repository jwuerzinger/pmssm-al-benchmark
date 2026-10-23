"""One figure: in-band input marginals (3x3) plus the target marginal (right).

The paper's "holistic comparison" of the relic-density runs: for every
surrogate's best regression cell the seeds' final cumulative training sets are
pooled (state.pt, as in plot_al_input_target_diagnostics) and compared with
the valid random pool and the emcee MCMC reference.

  * nine panels on the left: density-normalised 1-D marginals of the free
    parameters, restricted to points inside the +-10% target band, which is
    what makes the samples comparable (an AL set is ~10% in-band, the
    reference ~64%);
  * one tall panel on the right: the target marginal of every full sample on
    shared log-log axes, with the band shaded; one legend (with sample
    sizes) above the grid.

Same data as the supplementary input_overlay_inband.png and omega_overlay.png,
with the paper's model set, display names and colours, and the picks chosen
by in-band yield (``--picks`` overrides, see plot_hits_per_desired_heads.py).

``--arms`` replaces the per-model picks by explicit cells, each with its own
line style, e.g. the regression and classification heads of one model side by
side: ``--arms transformer:entropy_batch:cold:solid,transformer:cls_entropy:cold:dashed``.

Usage
-----
    python scripts/figures/plot_distribution_overlay.py \
        --picks tabpfn:tol_only_random:tabpfn \
        --output .../distribution_overlay.png --summary-json .../distribution_overlay.json
    python scripts/figures/plot_distribution_overlay.py --no-pool \
        --arms transformer:entropy_batch:cold:solid,dnn:entropy_batch:cold:solid,\
transformer:cls_entropy:cold:dashed,dnn:cls_entropy:cold:dashed \
        --output .../distribution_overlay_heads.png
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
import numpy as np
import pandas as pd


import scripts.figures.plot_hit_rate_trajectories_multiseed as H                       # noqa: E402
from scripts.figures.plot_hits_per_desired_heads import REG_PICKS, parse_picks           # noqa: E402
from scripts.figures.plot_al_input_target_diagnostics import _pooled_cell_data, _arm_color  # noqa: E402
from scripts.analysis.analyse_runs import FREE_PARAM_NAMES, FREE_PARAM_INDICES            # noqa: E402
from pmssm.config import TARGET_CONFIG                                   # noqa: E402

# Axis labels in the paper's notation (Table 1); masses in GeV.
PRETTY = {
    "meL":  r"$m_{\tilde e_L}$ [GeV]",
    "meR":  r"$m_{\tilde e_R}$ [GeV]",
    "M_1":  r"$M_1$ [GeV]",
    "M_2":  r"$M_2$ [GeV]",
    "mu":   r"$\mu$ [GeV]",
    "At":   r"$A_t$ [GeV]",
    "Ab":   r"$A_b$ [GeV]",
    "Atau": r"$A_\tau$ [GeV]",
    "tanb": r"$\tan\beta$",
}
STRATEGY_LABEL = {
    "entropy_batch": "entropy batch", "cls_entropy": "classification entropy",
    "tol_only_random": "tol + rnd", "top_k": "top-k", "top_k_tol_only": "tol + top-k",
    "bald": "BALD",
}
LINESTYLES = {"solid": "-", "dashed": "--", "dotted": ":", "dashdot": "-."}
REFERENCES = ("random pool", "emcee MCMC")


def _arms_from_spec(spec: str) -> list[dict]:
    """[{key, model, strategy, warm, ls, label}] from model:strategy:warm[:style],..."""
    arms = []
    for item in (x.strip() for x in spec.split(",") if x.strip()):
        parts = item.split(":")
        if len(parts) not in (3, 4):
            raise click.BadParameter(f"arm {item!r}: expected model:strategy:warm[:style]")
        model, strat, warm = parts[:3]
        ls = LINESTYLES[parts[3]] if len(parts) == 4 else "-"
        arms.append(dict(key=f"{model}/{strat}/{warm}", model=model, strategy=strat, warm=warm,
                         ls=ls, label=f"{H.MODEL_DISPLAY.get(model, model)}, "
                                      f"{STRATEGY_LABEL.get(strat, strat)}"))
    return arms


def _arms_from_picks(picks: str, models: str) -> list[dict]:
    keep = [m.strip() for m in models.split(",") if m.strip()]
    pick_map = parse_picks(picks, REG_PICKS)
    return [dict(key=m, model=m, strategy=pick_map[m][0], warm=pick_map[m][1], ls="-",
                 label=H.MODEL_DISPLAY.get(m, m)) for m in keep]


@click.command()
@click.option("--manifest", default="/ptmp/jwuerzin/analysis/all_runs/sweep_manifest.csv", show_default=True)
@click.option("--picks", default="", show_default=True,
              help="Override regression picks: model:strategy:warm[,...].")
@click.option("--models", default="transformer,dnn,deep_gp,exact_gp,tabpfn", show_default=True)
@click.option("--arms", default="", show_default=True,
              help="Explicit cells model:strategy:warm[:solid|dashed|dotted|dashdot],... "
                   "(replaces --picks/--models; labels carry the strategy).")
@click.option("--pool/--no-pool", default=True, show_default=True, help="Draw the random pool.")
@click.option("--include-status", default="completed,timeout,running,missing", show_default=True)
@click.option("--pool-dir", default="/ptmp/jwuerzin/data/18387358", show_default=True)
@click.option("--cache-dir", default="/ptmp/jwuerzin/analysis/all_runs", show_default=True)
@click.option("--mcmc-data-dir", default="/ptmp/jwuerzin/data/neutralino_v4", show_default=True)
@click.option("--mcmc-max-samples", default=500_000, show_default=True)
@click.option("--target", default="DMRD", show_default=True)
@click.option("--tolerance", default=0.10, show_default=True)
@click.option("--require-neutralino-lsp/--no-require-neutralino-lsp", default=False, show_default=True)
@click.option("--output", required=True)
@click.option("--summary-json", default="")
def main(manifest, picks, models, arms, pool, include_status, pool_dir, cache_dir, mcmc_data_dir,
         mcmc_max_samples, target, tolerance, require_neutralino_lsp, output, summary_json):
    cfg = TARGET_CONFIG[target]
    true_val = float(cfg["true_value"])
    arm_list = _arms_from_spec(arms) if arms else _arms_from_picks(picks, models)

    allowed = {s.strip() for s in include_status.split(",") if s.strip()}
    df = pd.read_csv(manifest)
    df = df[df["status"].isin(allowed)]
    df = df[df["expected_run_dir"].map(lambda d: isinstance(d, str) and (Path(d) / "state.pt").exists())]

    # key -> arrays, and key -> (colour, linestyle, label)
    y_samples: dict[str, np.ndarray] = {}
    x_samples: dict[str, np.ndarray] = {}
    style: dict[str, tuple] = {}
    summary = {"target": target, "tolerance": tolerance, "samples": {}}
    for a in arm_list:
        sub = df[(df["model"] == a["model"]) & (df["strategy"] == a["strategy"])
                 & (df["warm_start"] == a["warm"])]
        X, y, n_used = _pooled_cell_data(list(sub["expected_run_dir"]), require_neutralino_lsp)
        if X is None:
            click.echo(f"[warn] {a['key']}: no runs", err=True)
            continue
        x_samples[a["key"]] = np.asarray(X)
        y_samples[a["key"]] = np.asarray(y).ravel()
        style[a["key"]] = (H.MODEL_COLORS.get(a["model"], "gray"), a["ls"], a["label"])
        summary["samples"][a["label"]] = {"model": a["model"], "strategy": a["strategy"],
                                          "warm": a["warm"], "n_seeds": n_used, "n_pooled": int(len(y))}
        click.echo(f"[overlay] {a['key']}: {n_used} seeds, {len(y)} pooled points")

    ref_names = [n for n, on in zip(REFERENCES, (pool, bool(mcmc_data_dir) and cfg.get("has_mcmc_reference", False))) if on]
    if pool:
        Xb, Yb = H._load_xy_full(pool_dir, target, Path(cache_dir))
        x_samples["random pool"] = np.asarray(Xb)[:, FREE_PARAM_INDICES]
        y_samples["random pool"] = np.asarray(Yb).ravel()
        style["random pool"] = (_arm_color("random pool", ref_names), "-", "random pool")
        summary["samples"]["random pool"] = {"n_pooled": int(len(y_samples["random pool"]))}
    if "emcee MCMC" in ref_names:
        from pmssm.data import load_mcmc_data  # noqa: PLC0415
        Xm, Ym = load_mcmc_data(data_dir=mcmc_data_dir, target=target,
                                require_neutralino_lsp=require_neutralino_lsp,
                                max_samples=mcmc_max_samples or None)
        Xm = Xm.numpy() if hasattr(Xm, "numpy") else np.asarray(Xm)
        Ym = (Ym.numpy() if hasattr(Ym, "numpy") else np.asarray(Ym)).ravel()
        x_samples["emcee MCMC"] = Xm[:, FREE_PARAM_INDICES]
        y_samples["emcee MCMC"] = Ym
        style["emcee MCMC"] = (_arm_color("emcee MCMC", ref_names), "-", "emcee MCMC")
        summary["samples"]["emcee MCMC"] = {"n_pooled": int(len(Ym))}

    # In-band restriction for the input marginals.
    inband = {}
    for key, X in x_samples.items():
        y = y_samples[key]
        sel = np.isfinite(y) & (np.abs(y - true_val) / true_val < tolerance)
        inband[key] = X[sel]
        summary["samples"][style[key][2]]["n_inband"] = int(sel.sum())

    # Two thirds of the original 8.8 in height (user request, 2026-10-12): the
    # marginals read fine as squat panels and the figure then takes less page.
    fig = plt.figure(figsize=(15, 5.9))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 1.35], wspace=0.10, hspace=0.50,
                          left=0.03, right=0.95, bottom=0.10, top=0.86)
    axes = [fig.add_subplot(gs[j // 3, j % 3]) for j in range(len(FREE_PARAM_NAMES))]
    ax_t = fig.add_subplot(gs[:, 3])

    for j, (ax, pname) in enumerate(zip(axes, FREE_PARAM_NAMES)):
        lo = min(float(np.nanmin(X[:, j])) for X in inband.values() if len(X))
        hi = max(float(np.nanmax(X[:, j])) for X in inband.values() if len(X))
        bins = np.linspace(lo, hi, 51)
        for key, X in inband.items():
            color, ls, label = style[key]
            x = X[:, j]
            x = x[np.isfinite(x)]
            ax.hist(x, bins=bins, density=True, histtype="step", lw=1.6, color=color,
                    linestyle=ls, label=label)
        ax.set_xlabel(PRETTY.get(pname, pname), fontsize=14)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=11)
        ax.grid(alpha=0.3)
        ax.margins(x=0)

    bins = np.logspace(-4, 0, 81)
    ax_t.axvspan(true_val * (1 - tolerance), true_val * (1 + tolerance), color="gold",
                 alpha=0.25, lw=0, label=f"target ±{int(tolerance * 100)}%")
    vmin, vmax = np.inf, 0.0
    for key, y in y_samples.items():
        color, ls, label = style[key]
        y = np.asarray(y, dtype=float).ravel()
        y = y[np.isfinite(y) & (y > 0)]
        dens, _ = np.histogram(np.clip(y, bins[0], bins[-1]), bins=bins, density=True)
        ax_t.stairs(dens, bins, lw=1.8, color=color, linestyle=ls, label=f"{label} (N={len(y):,})")
        pos = dens[dens > 0]
        if len(pos):
            vmin, vmax = min(vmin, pos.min()), max(vmax, pos.max())
    ax_t.axvline(true_val, color="black", ls=":", lw=1.2)
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    if np.isfinite(vmin) and vmax > 0:
        ax_t.set_ylim(vmin * 0.5, vmax * 3.0)
    ax_t.set_xlabel(cfg.get("label") or target, fontsize=14)
    # Ticks and label on the right: the left edge abuts the parameter grid.
    ax_t.yaxis.tick_right()
    ax_t.yaxis.set_label_position("right")
    ax_t.set_ylabel("density", fontsize=14)
    ax_t.tick_params(axis="both", labelsize=11)
    ax_t.grid(alpha=0.3, which="both")
    # One legend above the grid; inside the target panel it hid the MCMC tail.
    handles, labels = ax_t.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4 if len(labels) > 6 else 3, fontsize=11, frameon=True, framealpha=0.9,
               columnspacing=1.4)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    H._enforce_marker_policy(fig)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"wrote {out}")
    if summary_json:
        json.dump(summary, open(summary_json, "w"), indent=2)
        click.echo(f"wrote {summary_json}")
    for name, v in summary["samples"].items():
        click.echo(f"  {name:36} N={v['n_pooled']:>9,}  in-band={v['n_inband']:>7,}"
                   + (f"  ({v['strategy']}/{v['warm']}, {v['n_seeds']} seeds)" if "strategy" in v else ""))


if __name__ == "__main__":
    main()
