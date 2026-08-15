# FMCL Paper 2A — Convergence & Privacy Simulation

Supporting simulation code, derivations, and experiment records for
**"Convergence and Privacy Guarantees for Federated Learning Under
Correlated Device Dropout"** (FMCL theoretical series, Paper 2A).

This repo exists to make every quantitative claim added to the manuscript
during its novelty/technical-contribution revision traceable to an exact
script, parameter set, and commit. Read `EXPERIMENT_PLAN.md` first — it is
the pre-registered plan for every experiment before any of them were run —
then `EXPERIMENT_LOG.md` for what was actually executed and when.

## Why this exists

A review of the original manuscript draft found that its most novel claim
(Theorem 2, the joint privacy–accuracy feasibility condition) was validated
only by a single plugged-in numerical example, while its least novel claim
(Theorem 1's 1/p convergence-floor scaling) carried the paper's only real
simulation. The experiments in this repo are designed to correct that
imbalance: tighten Theorem 2 (E1), make an idealizing assumption explicit
(E2), give Theorem 2 an actual feasibility-boundary simulation (E3),
generalize the correlated-participation model (E4), attempt an empirical
correlation estimate (E5), package the feasibility condition for practical
use (E6), and — the most consequential single addition — test the theory
against a real nonconvex model on real data using Paper 3's existing GPU
infrastructure (E7).

## Repo layout

```
EXPERIMENT_PLAN.md         pre-registered plan (write before running)
EXPERIMENT_LOG.md          append-only record of what was actually run
ENVIRONMENT.md             exact software environment, per track
requirements.txt           pinned Python dependencies (Track A)
src/common/                shared library: FedProx update, RDP accounting,
                            participation generators, synthetic data —
                            used by every experiment, tested once
src/e1_.../ ... e7_.../    one directory per experiment in EXPERIMENT_PLAN.md
results/e1/ ... e7/        raw output (CSV/JSON) + figures per experiment
docs/derivations/          new proofs (amplification, inexactness, Markov
                            floor) as standalone markdown, meant to be
                            checked independently of the manuscript prose
server_jobs/               R770-bound scripts (Track B), tmux launch
                            commands, and the exact skip-if-exists /
                            logging contract — not executed from this repo
manuscript_patches/        notes on which manuscript section/equation each
                            result feeds into, and the E8 structural-repair
                            checklist
```

## Compute tracks

- **Track A**: runs on a local, CPU-only development machine with no access to the
  R770 or its jump server. Covers all theory derivations and every
  synthetic/convex-scale simulation.
- **Track B**: needs the R770's GPUs and Paper 3's already-built
  FEMNIST-ResNet / MIT-BIH pipelines. Scripts are written and validated
  against toy data here, then handed off for Pijush to run via `tmux`,
  following the same unattended-execution discipline established for
  Paper 3 (this is a Texas state government system; server time is not
  spent on interactive debugging).

## Reproducing a result

Every experiment directory has its own `run.py` (or numbered scripts) and
a `README.md` stating the exact command. Results are never hand-edited;
regenerate from the script if a number looks wrong, and if the regenerated
number differs from what's committed, that's a bug report, not a rounding
choice.

## Status

See `EXPERIMENT_PLAN.md` §1 for the live status table (Planned / Running /
Complete / Blocked) as of the last commit.
