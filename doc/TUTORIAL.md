# Tutorial: one active-learning loop from start to finish

This walkthrough runs the whole loop on a small data sample, without the
physics chain, and ends with the kind of figure the paper is made of. It
takes about half an hour on a CPU (the two runs of step 4 can run side by side) and a few minutes on a GPU. Every command
below is also in `scripts/tutorial/run_tutorial.sh`, so

```bash
pixi run -e rocm tutorial        # -e cuda on NVIDIA; `pixi run tutorial` inside an activated env
```

does all of it in one go and leaves its results in `tutorial_runs/`.

**What is different from a production run.** Production labels every selected
candidate by simulating it (SPheno, micrOMEGAs, SModelS through the
`Run3ModelGen` submodule, `--generate-data`). Here the loop runs in its
*oracle* configuration (`--candidate-source mcmc`): candidates are drawn from a
sample of the paper's reference posterior, whose relic densities are already
known, so a selected point is labelled by lookup. The paper uses this
configuration as the theoretical limit of acquisition; for a tutorial it means
the loop can be exercised end to end before the simulator is built. The sizes are also
cut down (1,500 initial models, 100 selections from 5,000 candidates per
iteration, 5 iterations, 80 training epochs; production is 2,000, 500 of
1,000,000, 40 to 200 iterations and up to 10,000 epochs).
Nothing you see here is a physics result.

## 1. Set up

```bash
git clone --recurse-submodules git@github.com:jwuerzinger/pmssm-al-benchmark.git
cd pmssm-al-benchmark
pixi install -e rocm                   # or -e cuda
(cd Run3ModelGen && pixi run build)    # the physics chain; see its README
bash scripts/tutorial/fetch_data.sh    # unpacks the 12 MB sample shipped in data/tutorial/
```

Both submodules are required: `al_pmssmwithgp` holds the GP models and the
`entropy_batch` selection code, `Run3ModelGen` is the physics chain the loop
labels with and the source of every dataset (fetching it needs a CERN
account). Steps 2 to 5 run in the oracle configuration and do not launch the
simulator, so they work while the `Run3ModelGen` build is still running; step
6 needs it. The sample ships with the repository as
`data/tutorial/pmssm-al-tutorial-data-v1.tar.gz` (checksum beside it); the
fetch script unpacks it into two ROOT files with the same `susy` tree and
branches as the real data (`data/tutorial/README.txt` records their
provenance):

| file | content |
|---|---|
| `data/tutorial/pool/ntuple.tutorial.0.root` | 50,000 models drawn uniformly from the relic-density pool: the 19 input parameters, Ω h², the Higgs mass and the LSP type the validity cuts need |
| `data/tutorial/mcmc/ntuple.tutorial.0.root` | 100,000 rows drawn uniformly over the emcee reference posterior, walker repeats included |

`TUTORIAL_DATA_URL` (a URL or a local path) points the fetch script at
another copy of the tarball; `TUTORIAL_DATA` changes where it unpacks.

## 2. Run one loop

```bash
STAMP=$(date +%Y%m%d_%H%M%S)
python pipelines/active_learning_transformer.py \
    --target DMRD --data-dir data/tutorial/pool --mcmc-data-dir data/tutorial/mcmc \
    --candidate-source mcmc --mcmc-max-samples 100000 \
    --n-samples 1500 --static-eval-size 5000 \
    --n-iterations 5 --n-select 100 --n-candidates 5000 --entropy-pool-size 500 \
    --mc-samples 10 --epochs 80 --patience 25 --no-warm-starting --seed 1 --gpu-ids 0 \
    --selection-strategy entropy_batch \
    --output-dir tutorial_runs/active_learning_transformer_entropy_batch_cold_seed1_${STAMP}
```

Reading the flags in groups:

- **Data.** `--data-dir` is the pool the initial labelled set and the random
  baseline draw from; `--mcmc-data-dir` is the reference. `--target DMRD`
  selects the relic density, with the boundary Ω h² = 0.12 at zero in the
  training space t = log(Y / 0.12) (`doc/TARGETS.md`).
- **Budget.** 1,500 models to start, 5 iterations of 100 selections. The
  static evaluation set of 5,000 models is reserved from the pool before
  anything is trained and never touched again.
- **Acquisition.** `entropy_batch` scores the 5,000 candidates with 10
  MC-dropout passes, keeps the 500 of largest proximity-weighted variance
  near the boundary and picks a diverse batch of 100 from their predictive
  covariance (`doc/ACQUISITION.md`). The tolerance cut and proximity
  weighting are on at their defaults.
- **Training.** 80 epochs with early stopping, patience 25, from scratch each
  iteration (`--no-warm-starting`, the paper's choice for the neural
  surrogates).
- **Reproducibility.** `--seed 1` fixes the split, the candidate draws and
  the initialisation. A run directory whose name ends in a timestamp is used
  as given; otherwise the pipeline appends the strategy, warm mode, seed and
  time itself.

The log (also saved as `active_learning.log` in the run directory) first
reports the ingest, including the validity filter and the fraction of models
that survive it, then `Oracle mode: ... candidates / ... eval`, and then one
block per iteration: the training of the active-learning model and of the
random baseline, the candidate draw (`Oracle: drew ...`), the selection and
the sizes of both training sets.

## 3. Read the run directory

```
tutorial_runs/active_learning_transformer_entropy_batch_cold_seed1_<stamp>/
├── active_learning.log         the log above
├── state.pt                    the loop state after the last iteration (resume point)
├── summary.json                written at the end: config, per-iteration metrics
├── accuracy_trajectory.json    verdict accuracy per iteration on each evaluation set
├── plots/hist_dataset.png      the target histogram of the ingested pool
├── gifs/                       one animation per plot type, AL beside baseline
└── iteration_001 ... 005/
    ├── selected_points.csv     the batch of that iteration with its scores
    ├── al_model_checkpoint.pt, baseline_model_checkpoint.pt
    ├── al_training.log, baseline_training.log
    └── plots/al/, plots/baseline/   losses, true-versus-predicted, candidate
                                     uncertainty, input and target histograms
```

`doc/RUN_OUTPUT_LAYOUT.md` describes every file. Two things worth opening:
`iteration_003/plots/al/al_candidate_uncertainty_iter3.png` shows where the
surrogate was uncertain and what it picked, and the labelled-set growth is one
line of Python away:

```python
import torch
s = torch.load("tutorial_runs/<run>/state.pt", weights_only=False, map_location="cpu")
print(s["al_n_train"])          # labelled-set size after each iteration
```

A production run writes exactly this layout, plus the simulator's ntuples and
packed workspaces per iteration; the figure scripts read nothing else.

## 4. A second run, and a manifest

Run the same loop with greedy selection:

```bash
python pipelines/active_learning_transformer.py ... --selection-strategy top_k \
    --output-dir tutorial_runs/active_learning_transformer_top_k_cold_seed1_${STAMP}
```

Every figure script discovers runs through a manifest, one row per run with
its cell (model, strategy, warm start, seed) and status. The sweep launcher
writes these rows at submission; for runs started by hand,
`build_probe_manifest.py` reconstructs them from the directory names:

```bash
python scripts/campaign/build_probe_manifest.py --output-root tutorial_runs \
    --pattern "active_learning_*_${STAMP}" --out tutorial_runs/manifest.csv
cat tutorial_runs/manifest.csv
```

This is why the directories above follow the
`active_learning_<model>_<strategy>_<warm>_seed<N>_<stamp>` convention.

## 5. Figures

The single-run figure, hit rate against iteration for one run with its random
baseline:

```bash
python scripts/figures/plot_hit_rate_single_run.py \
    --run-dir tutorial_runs/active_learning_transformer_entropy_batch_cold_seed1_${STAMP} \
    --seed 1 --model-label transformer --strategy-label entropy_batch --warm-label cold \
    --baseline-data-dir data/tutorial/pool --cache-dir tutorial_runs/cache \
    --output-dir tutorial_runs/figures
```

and the manifest-driven comparison, the script behind the paper's main
figures:

```bash
python scripts/figures/plot_hit_rate_trajectories_multiseed.py \
    --manifest tutorial_runs/manifest.csv --output-dir tutorial_runs/figures \
    --baseline-data-dir data/tutorial/pool --mcmc-data-dir data/tutorial/mcmc \
    --pool-cache-dir tutorial_runs/cache --mcmc-yield-json "" \
    --min-seeds 1 --min-seeds-axis 1 --tolerances 0.10,0.20,0.50
```

In `tutorial_runs/figures/` you get `hit_rate_single_run.png` and
`hits_per_desired_single_run.png` for the first run, and for the comparison
`hit_rate_best_per_model.png`, `hit_rate_strategy_<strategy>.png`,
`hit_rate_model_transformer.png` and their `hits_per_desired_*` twins, one
panel per tolerance. Solid lines are the active-learning model, dashed lines
the random baseline, the horizontal line the pool's own in-band fraction; the
`hits_per_desired` figures divide by the number of simulator calls the
labelled points would have cost, which is the paper's cost-aware metric.
Beside the figures the script writes `random_baseline_prevalence.json` and
`data_efficiency_best_per_model.json`, the numbers behind the curves.

With one seed per cell and five iterations the curves are noisy and the two
strategies may well cross; the paper's figures average five seeds over 40 to
200 iterations. `--min-seeds 1` is what lets a single seed through at all.

## 6. From here to the production loop

- **Labelling by simulation.** Drop `--candidate-source mcmc`, pass
  `--generate-data`, and the loop launches `Run3ModelGen` workers for every
  selected batch (`--gen-workers`); the exclusion target `ExpR` adds SModelS
  and needs its database (`doc/TARGETS.md`). The oracle configuration exists
  only for the relic density, because only it has a reference posterior.
- **Production sizes and the cluster.** The per-model scripts in `slurm/al/`
  carry the paper's settings and read overrides from `AL_*` variables; one
  run is one `sbatch` (`doc/SLURM.md`). A run resumes from its `state.pt`
  with `--resume-from` and `--n-additional-iterations`.
- **Many runs.** The sweep launcher submits the model x strategy x warm-start
  grid at several seeds and writes the manifest rows itself; the chase loop
  resumes everything to a target iteration count; the regeneration job
  renders every figure of the paper from the manifests (`doc/CAMPAIGN.md`,
  `doc/REPRODUCING_THE_PAPER.md`).
- **Other surrogates.** `pipelines/active_learning_gp.py` (exact and deep
  GPs, needs the `al_pmssmwithgp` submodule), `active_learning_dnn.py` and
  `active_learning_tabpfn.py` (needs a TabPFN licence token) take the same
  data and acquisition flags; `doc/CLI_REFERENCE.md` lists every option.

## If something goes wrong

- *No GPU*: the pipelines fall back to the CPU; set `OMP_NUM_THREADS` to the
  cores you can spare. `--gpu-ids 0,1` runs the two models on two devices.
- *`n_samples exceeds available data`*: fewer models survived the validity
  cuts than requested; lower `--n-samples` or `--static-eval-size`.
- *`MCMC candidate pool exhausted`*: iterations x `--n-select` exceeded the
  oracle reservoir (90 % of the reference sample); lower either.
- *`--candidate-source=mcmc requires --mcmc-data-dir`*: the oracle needs the
  reference; `--no-mcmc-eval` and the oracle exclude each other.
- *`No module named 'gp_pipeline'`*: the submodules are not checked out;
  `git submodule update --init`.
- *The figure script skips a run*: its row in the manifest is not
  `completed`; a run is complete only once `summary.json` exists.
