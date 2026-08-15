# E7 — real-model feasibility sweep (MIT-BIH ResNet)

## What this tests

Whether Theorem 1's qualitative prediction — lower participation, higher
convergence floor; more privacy noise, higher floor — holds on a real,
already-validated nonconvex model, not only the synthetic multinomial-
logistic setting used in E2–E6. MIT-BIH ResNet was chosen because it is
the one domain with a decisive, well-powered baseline result already
established (8 seeds, statistically significant convergence under
favorable conditions), so this experiment adds participation dropout and
DP noise on top of a model already known to work, rather than testing
two things (does the model converge at all; does dropout/privacy hurt
it) at once.

## Status: script written, not executed

`run_e7_mitbih.py` implements the federated round loop, DP mechanics
(participation sampling, clipping, Gaussian noise, aggregation —
`dp_wrapper.py`, tested against a toy model, see `test_dp_wrapper.py`),
checkpointing, resumability, and CPU thread capping. It has **not** been
run against the real model or data, and cannot be from this environment
(no GPU, no network route to the R770).

## Before running: integration checklist

The script marks every point requiring confirmation with
`TODO(confirm)`. Each was based on names referenced in prior work on
this codebase, not verified directly against the current source. Check
each before running:

1. **Model import and constructor.** `from models import ResNet1D_MITBIH`
   and its constructor arguments (input length, channel count, number
   of classes) — confirm the actual class name and signature.
2. **Data loader.** `from data_mitbih import load_mitbih_clients` and its
   return type (a dict keyed by client id, or a list) — confirm the
   actual interface, including how a single client's batches are
   iterated.
3. **Local training step** (`local_train_step`) — confirm loss function,
   optimizer, and per-client batching match the existing pipeline's own
   local-update logic. If they differ, results here will not be
   comparable to the established 8-seed baseline.
4. **Evaluation** (`evaluate_global_model`) — confirm the held-out split
   and metric match existing convention.
5. **Learning rate default** (`--lr 0.001`) — confirm this is still the
   current stable value for this model on this domain.

None of these are guesses picked at random — each is grounded in a
specific reference from prior work on this codebase — but none should be
trusted without a direct check against current source.

## Before running: recommended pilot

Run a short pilot (10–20 rounds, one K value, sigma=0) first to confirm
the integration points work end-to-end and to check whether the
floor stabilizes within the default 150 rounds — that round count is
carried over from this project's own synthetic experiments and has not
been validated for this real model. If the floor is still moving
substantially at round 150, extend before running the full sweep, not
after.

## Launch

```bash
cd /path/to/mitbih/codebase   # wherever fl_harness.py / models.py live
tmux new -s e7_mitbih_sweep
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 run_e7_mitbih.py \
    --K_grid 5 10 20 \
    --n_seeds 2 \
    --rounds 150 \
    --local_epochs 1 \
    --lr 0.001
# detach: Ctrl-b d
```

Resumable: if interrupted, re-running the same command skips completed
(K, sigma_frac, seed) cells (checked against `results/e7/e7_mitbih_results.csv`)
and resumes any in-progress cell from its last checkpoint
(`results/e7/ckpt_*.pt`, saved every 25 rounds by default).

## Grid and why it's smaller than a first instinct would suggest

- **K in {5, 10, 20}**, not a participation *rate* swept against a large
  population — MIT-BIH's real client pool is small (on the order of
  44–47 patients), so K is set directly, matching the existing
  pipeline's own convention (which selects a fixed-size K per round, not
  independent Bernoulli(p) participation — see the sampling-model note
  in `run_e7_mitbih.py`'s docstring; this is a materially different
  process from what this project's theoretical results assume, and
  should be described as such, not glossed over as "the same as p").
- **sigma in {0, sigma_max/2, sigma_max}**, 3 values, not a finer sweep —
  enough to see the direction and rough shape without a large grid.
- **2 seeds**, not the established 8 — reduced for budget reasons,
  stated explicitly rather than presented as matching the established
  standard. If the qualitative pattern is clear at 2 seeds, extending to
  match the 8-seed standard is a natural follow-up, not a requirement to
  get a first, directionally useful result.

3 K-values × 3 sigma-values × 2 seeds = 18 runs. Per-run wall-clock time
has not been measured for this specific model; the closest available
reference point on the same hardware (Icentia11k ResNet, a comparably-
scaled model) ran roughly 107 minutes per seed at a much larger round
count and dataset. This is not a reliable estimate for MIT-BIH's smaller
dataset and shorter default round count here, and should be treated as
an order-of-magnitude anchor only — time the pilot run before committing
to the full grid.

## After running

Paste `results/e7/e7_mitbih_results.csv` and the corresponding
`EXPERIMENT_LOG.md`-style entry (command, environment, wall-clock time,
result summary) back for analysis — floor-vs-K ordering, floor-vs-sigma
ordering, and a Pearson-r fit against 1/K matching the statistic already
reported for the synthetic scenarios in the manuscript's own Section 11.
