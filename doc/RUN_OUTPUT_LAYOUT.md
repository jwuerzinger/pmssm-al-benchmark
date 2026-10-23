# What a run directory contains

Every pipeline writes one directory (`--output-dir`), which it also resumes
from. A finished 200-iteration run:

```
active_learning_<model>_<strategy>_<warm>_seed<N>_<sweep id>/
├── active_learning.log            the run log: configuration, per-iteration metrics,
│                                  generation statistics, the validity-fraction line
├── state.pt                       the loop state after the last finished iteration:
│                                  cumulative X, Y, feature arrays, the AL and baseline
│                                  index sets, per-iteration metric lists; written
│                                  atomically after every iteration (pmssm/resume.py)
├── summary.json                   written once at the very end: timestamp, config,
│                                  iterations, final_dataset_size, al_metrics,
│                                  baseline_metrics; its presence is what marks a
│                                  run `completed` in the manifests
├── accuracy_trajectory.json       per-iteration verdict accuracy of the AL and the
│                                  baseline model on static_random, mcmc, train and
│                                  val (pmssm/accuracy.py; also the cache the figure
│                                  scripts refresh)
├── lengthscales.csv               GP runs with --track-lengthscales: ARD lengthscales
│                                  per iteration
├── plots/hist_dataset.png         the ingested pool's target histogram
├── gifs/*.gif                     one animation per per-iteration plot type, AL beside
│                                  baseline (pmssm/gifs.py, made at the end of the run)
└── iteration_NNN/
    ├── selected_points.csv        the batch chosen this iteration, with its scores
    ├── al_training.log            the two training workers' logs
    ├── baseline_training.log
    ├── al_model_checkpoint.pt     the weights (or GP state) after this iteration;
    ├── baseline_model_checkpoint.pt   TabPFN has none and refits from state.pt
    ├── plots/
    │   ├── al/                    losses, true-versus-predicted, candidate uncertainty,
    │   ├── baseline/              input and target histograms, parallel coordinates
    │   └── scatterplots_iter_NNN.png
    ├── ntuples/                   the labelled models of this iteration as ROOT ntuples,
    │                              one per generation worker (flattened by the housekeeping)
    ├── smodels_best_analysis.json exclusion target only: the most sensitive analysis
    │                              per labelled point
    └── debris.tar                 everything else the simulator left (SLHA spectra,
                                   SPheno logs, Snakemake state), packed by
                                   pmssm/iteration_housekeeping.py
```

A run that was interrupted holds `state.pt` and the iterations up to the last
finished one and no `summary.json`; resuming continues from `state.pt` and
appends iterations in place. The housekeeping never touches the newest
iteration directory, so an in-progress iteration keeps its unpacked
workspaces.

Runs started through the sweep launcher follow the naming convention above
and have a row in a manifest (`doc/CAMPAIGN.md`); runs started by hand are
named after `--output-dir` plus a timestamp. The per-run GIFs can be remade
with `python -m pmssm.gifs <run dir>`.
