"""Cache per-point predictions on the static-random eval set, from checkpoints.

Why this exists
---------------
``accuracy_trajectory.json`` stores one accuracy number per iteration, so any
other metric on the same eval set (e.g. an accuracy weighted by proximity to the
threshold) needs the per-point predictions, which no run writes. This script
reproduces them once, on a GPU, and stores them so every such metric can then
be computed on the CPU without touching a model again.

For each paper configuration (``plot_best_config.TARGETS[target]["picks"]``),
seed, role (AL and random baseline) and selected iteration, the model is
rebuilt exactly as the accuracy plots rebuild it (``_load_iter_model``; TabPFN
is re-fitted in context with ``evaluate_uq._predict_tabpfn``) and its output in
transformed space is predicted on the run's static-random eval set (100k pool
rows carved by ``_static_random_indices``). The sign of the output at the
target threshold is the verdict for both heads.

Output: ``<analysis>/<runs>/static_preds/<run_dir_name>.npz`` with one float32
array per ``<role>_<iteration>``, plus ``_y_true_t.npy`` (the eval set's true
transformed target) next to them. Each npz also stores the unweighted accuracy
it implies, so the cache can be checked against ``accuracy_trajectory.json``.

Usage (GPU node)
----------------
    python scripts/static_predictions_cache.py --target DMRD --model transformer
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import plot_hit_rate_trajectories_multiseed as H                  # noqa: E402
from plot_best_config import TARGETS                              # noqa: E402
from pmssm.data import TARGET_CONFIG, transform_y                 # noqa: E402

A = "/ptmp/jwuerzin/analysis"
OUT = {"DMRD": f"{A}/all_runs/static_preds", "ExpR": f"{A}/expr_runs/static_preds"}


def _iters(n_iters: int, every: int) -> list[int]:
    """1, every, 2*every, ... and the last iteration (1-based)."""
    its = {1, n_iters} | set(range(every, n_iters + 1, every))
    return sorted(i for i in its if 1 <= i <= n_iters)


def _predict(model_type, role, iter_dir, sets, run_kwargs, target, X_eval, device):
    X_tr, Y_tr = sets[role]["train"]
    X_va, Y_va = sets[role]["val"]
    if model_type == "tabpfn":
        from evaluate_uq import _predict_tabpfn  # noqa: PLC0415
        return np.asarray(_predict_tabpfn(X_tr, Y_tr, X_eval, device, target)["mean"],
                          dtype=np.float32)
    model = H._load_iter_model(model_type, role, iter_dir, X_tr, Y_tr, X_va, Y_va,
                               run_kwargs, device)
    if model is None:
        return None
    try:
        if model_type in ("transformer", "dnn"):
            from pmssm.data import compute_stats  # noqa: PLC0415
            stats = compute_stats(X_tr, Y_tr.unsqueeze(-1), torch.arange(len(X_tr)))
            y = H._predict_transformer_t(model, X_eval, stats, device)
        else:
            y = H._predict_gp_t(model, model_type, X_eval, *H._GP_NORM,
                                jitter=float(run_kwargs.get("jitter", 1e-3) or 1e-3),
                                num_samples=int(run_kwargs.get("gp_num_samples", 8) or 8),
                                device=device)
        return np.asarray(y, dtype=np.float32)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@click.command()
@click.option("--target", type=click.Choice(list(TARGETS)), required=True)
@click.option("--model", "models", default="transformer,dnn,deep_gp,exact_gp,tabpfn",
              show_default=True, help="Comma list of bare model names.")
@click.option("--seeds", default="1,2,3,4,5", show_default=True)
@click.option("--roles", default="al,baseline", show_default=True)
@click.option("--every", default=10, show_default=True, help="Iteration stride.")
@click.option("--include-status", default="completed,timeout,running,missing,stalled",
              show_default=True)
@click.option("--pool-cache-dir", default=f"{A}/pool_cache", show_default=True)
@click.option("--static-eval-size", default=100_000, show_default=True)
@click.option("--initial-reserved", default=2000, show_default=True)
@click.option("--device", default="cuda", show_default=True)
def main(target, models, seeds, roles, every, include_status, pool_cache_dir,
         static_eval_size, initial_reserved, device):
    cfg = TARGETS[target]
    out_dir = Path(OUT[target])
    out_dir.mkdir(parents=True, exist_ok=True)
    from pmssm.data import build_norm_tensors  # noqa: PLC0415
    H._GP_NORM = build_norm_tensors()

    X_full, Y_full = H._load_xy_full(cfg["pool"], target, Path(pool_cache_dir))
    idx = H._static_random_indices(len(Y_full), initial_reserved, static_eval_size)
    X_eval = torch.from_numpy(np.asarray(X_full[idx], dtype=np.float32))
    Y_eval = torch.from_numpy(np.asarray(Y_full[idx], dtype=np.float32))
    y_true_t = transform_y(Y_eval, target=target).view(-1).numpy().astype(np.float32)
    np.save(out_dir / "_y_true_t.npy", y_true_t)
    thr = float(TARGET_CONFIG[target]["threshold"])
    click.echo(f"[eval] {target}: {len(idx)} rows, positive fraction "
               f"{float((y_true_t >= thr).mean()):.4f}")
    empty = torch.empty(0, 19)

    df = pd.read_csv(cfg["manifest"])
    df = df[df["status"].isin({s.strip() for s in include_status.split(",")})]
    seed_set = {int(s) for s in seeds.split(",")}
    for model in [m.strip() for m in models.split(",") if m.strip()]:
        strat, warm = cfg["picks"][model]
        sub = df[(df["model"] == model + cfg["tag"]) & (df["strategy"] == strat)
                 & (df["warm_start"] == warm) & (df["seed"].isin(seed_set))]
        for _, row in sub.iterrows():
            run_dir = Path(row["expected_run_dir"])
            if not (run_dir / "state.pt").exists():
                click.echo(f"[skip] {run_dir.name}: no state.pt")
                continue
            state = torch.load(run_dir / "state.pt", weights_only=False, map_location="cpu")
            run_kwargs = H._parse_run_kwargs_from_log(run_dir)
            run_kwargs.setdefault("target", target)
            n_iters = len(list(state["al_n_train"]))
            out_path = out_dir / f"{run_dir.name}.npz"
            cache = dict(np.load(out_path)) if out_path.exists() else {}
            click.echo(f"[run] {model} seed {row['seed']}: {run_dir.name} ({n_iters} it.)")
            for it in _iters(n_iters, every):
                todo = [r for r in roles.split(",") if f"{r}_{it}" not in cache]
                if not todo:
                    continue
                sets = H._build_eval_sets(run_dir, state, int(row["seed"]), X_full, Y_full,
                                          X_eval, Y_eval, empty, torch.empty(0), it - 1)
                if sets is None:
                    click.echo(f"  it {it}: eval sets not reconstructable")
                    continue
                for role in todo:
                    t0 = time.time()
                    # The run's own model type: the exclusion Exact GP classification
                    # arm is a Laplace GP classifier (laplace_gpc), not an exact GP.
                    mtype = str(run_kwargs.get("model_type") or model)
                    try:
                        y = _predict(mtype, role, run_dir / f"iteration_{it:03d}", sets,
                                     run_kwargs, target, X_eval, device)
                    except Exception as exc:  # noqa: BLE001
                        click.echo(f"  it {it} {role}: failed: {type(exc).__name__}: "
                                   f"{str(exc)[:120]}")
                        continue
                    if y is None:
                        click.echo(f"  it {it} {role}: no checkpoint")
                        continue
                    cache[f"{role}_{it}"] = y
                    cache[f"n_L_{it}"] = np.array(int(state["al_n_train"][it - 1])
                                                  + int(state["al_n_val"][it - 1]))
                    acc = float(np.mean((y >= thr) == (y_true_t >= thr)))
                    cache[f"acc_{role}_{it}"] = np.array(acc)
                    click.echo(f"  it {it:>3} {role:8} acc {acc:.4f} ({time.time() - t0:.0f}s)")
                np.savez(out_path, **cache)


if __name__ == "__main__":
    main()
