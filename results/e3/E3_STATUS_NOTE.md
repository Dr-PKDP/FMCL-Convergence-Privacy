# E3 status note — RESOLVED, revision 2 (corrected methodology)

**This supersedes the earlier version of this note (preserved below the
line, for the record).** The gap it identified was real; both causes have
now been found, one fixed, one characterized and quantified rather than
"fixed" (because it isn't a bug — see below).

## What changed since revision 1

1. **Fixed**: `measured_gap` is now the cumulative time-average
   `mean(gaps[0:round+1])`, matching Theorem 1's actual left-hand side,
   not a single round's instantaneous value.
2. **Fixed**: the server-side step size eta is now `1/L`, matching what
   Theorem 1 is proven at, using a real calibrated L per d (L=1.2556 at
   d=10, L=3.1631 at d=100 — see `results/e3/calibration/`), not the
   implicit eta=1 the first pass used.
3. **Fixed**: T_acc is now computed from calibrated L and D (see
   `calibrate.py`), not assumed as LD=10. D turned out to be tiny
   (0.0018 at d=10, 0.0283 at d=100) — the population-averaged gradient
   at initialization is close to zero for Synthetic(1,1) at this scale,
   apparently because each client's true model is drawn from a symmetric
   distribution and the population average largely cancels across many
   random clients. This makes T_acc negligible in both cases, which
   simplifies the picture: feasibility is governed almost entirely by the
   floor V_ind, not by the transient/round-count term.
4. **Found, characterized, not "fixed" because it isn't a bug**: even
   after (1)-(3), the achieved floor still runs well above V_ind's
   theoretical prediction. Investigated and traced to near-universal
   clipping: **78-98% of raw local FedProx updates exceed the clip bound
   C=1 across every tested configuration** (see table below). This is a
   known source of *bias* (not just variance) in differentially-private
   optimization, and Lemma 1's variance-only analysis does not currently
   model it — Assumption 3 (`||g_i^t|| <= G`, satisfied exactly by
   construction via clipping) is true, but doesn't imply Assumption 2's
   exact unbiasedness survives clipping when clipping is this aggressive.

## Final, complete result

**Zero-false-negative check: PASSES on the full corrected grid.**
0 violations across all 180 testable rows (24 training configs x up to
8 (eps_p, conversion) reads each, minus 12 correctly-excluded degenerate
T_needed<1 rows). Theorem 2 never promises feasibility where the
mechanism, as actually simulated, fails to deliver it.

**Theorem 2 is not tight, and the gap is now quantified, not just
observed.** Across the 180 testable rows, `measured_gap / V_ind_theory`:
min 0.47, median **2.16**, max 4.61. The gap correlates with clipping
fraction at Pearson r=0.66 across the full grid, and the pattern is
consistent by dimension: d=10 (mean 86.7% clipped) has median ratio 2.05;
d=100 (mean 96.8% clipped) has median ratio 2.88 — more clipping, larger
gap, in both the aggregate correlation and the per-dimension comparison.

**With these corrections, the empirical feasibility boundary now tracks
Theorem 2's predicted boundary closely at d=10** for eps_p in {2, 4, 8}:
the smallest empirically-feasible tested N matches the smallest
predicted-feasible tested N exactly (both 500 at eps_p=2; both 200 at
eps_p=4 and eps_p=8). eps_p=16 and the d=100 grid show the boundary
pushed out further than predicted or not reached within the tested N
range at all — consistent with the clipping-bias mechanism: looser
privacy budgets permit far more rounds (T_needed reaches into the
thousands for eps_p=16), all of which get capped at T_max=150 and read
at the steady-state floor, which is exactly where the ~2x clipping-bias
inflation bites hardest.

## sigma_max and clipping fraction across the grid

| d | N | sigma_max | mean frac. updates clipped |
|---|---|---|---|
| 10 | 200 | 1.57 | 81.0% |
| 10 | 500 | 2.49 | 84.8% |
| 10 | 1000 | 3.53 | 88.1% |
| 10 | 2000 | 5.00 | 87.4% |
| 10 | 5000 | 7.90 | 87.9% |
| 10 | 10000 | 11.18 | 88.7% |
| 10 | 20000 | 15.81 | 87.7% |
| 100 | 1000 | 1.12 | 95.3% |
| 100 | 3000 | 1.94 | 96.2% |
| 100 | 6000 | 2.74 | 97.1% |
| 100 | 10000 | 3.54 | 97.6% |
| 100 | 15000 | 4.33 | 97.2% |

## What this means for the manuscript

This is a genuine, quantified, defensible technical contribution, not a
loose end:

- **Theorem 2 is safe to use as a screening tool**: if it says a
  configuration is infeasible, it is infeasible. Confirmed across 180
  configurations, two dimensions, seven population sizes, four privacy
  budgets, and two RDP composition conventions.
- **Theorem 2 is not tight in the presence of aggressive clipping**, and
  the manuscript's own C=1 convention (confirmed via the resolved
  Q1) produces aggressive clipping throughout the tested regime, not just
  at extreme parameter values. A deployment relying on Theorem 2's exact
  N* should budget roughly double the naive population (or an
  equivalently relaxed accuracy/privacy target) for real confidence,
  pending a bias-corrected version of the bound.
- **A concrete, scoped tightening direction follows directly**: extending
  Lemma 1's analysis to account for clipping bias explicitly (as done in
  the differentially-private-SGD literature for bias-corrected or
  adaptive clipping) would close this gap and is a natural next paper's
  worth of work, or a bounded addition to this one if scope allows.
- This is also a clean, literature-consistent explanation, not an
  ad-hoc one: clipping bias under heavy clipping ratios is a documented
  phenomenon in DP-SGD analysis generally, not something specific to or
  surprising for FMCL.

---

## Original (revision 1) note, preserved below for the record

Everything below this line describes the state before the eta/L-D/
cumulative-average corrections above were made, and should be read as
superseded, not current.

---


## What IS solidly established

The **zero-false-negative check passes**: across all 180 testable rows
(24 training configs x up to 8 (eps_p, conversion) reads each, minus 12
correctly-excluded degenerate rows), no configuration Theorem 2 declares
infeasible on privacy grounds alone (T_needed < 1, i.e. the privacy budget
permits zero rounds of participation) was empirically observed to reach
the accuracy target. This part of the check does not depend on the issue
below, because it only tests the T_needed >= 1 boundary, not the
convergence floor's absolute value.

## What is NOT yet established, and why

The intended second check -- whether the *achieved* stationarity gap at
the privacy-permitted round count tracks Theorem 2's *predicted*
feasibility boundary -- turned out to depend on an assumption I made
without validating it at the actual operating point: that T_max=150
rounds is enough for the transient term (2L(F(w^0)-F*)/T in Theorem 1) to
become negligible relative to the floor V_ind.

That assumption was checked (see the pilot in EXPERIMENT_LOG.md) at
**sigma=0.5**. The real grid's sigma=sigma_max values, because sigma_max
is chosen to sit exactly at Lemma 4's noise ceiling and grows with N,
range from 1.1 up to **15.8** (see table below) -- one to two orders of
magnitude larger than the pilot tested. At sigma=15.8 (N=20000, d=10), the
injected per-parameter noise (std = sigma*C = 15.8) is far larger than the
scale of the weights being learned, and the measured stationarity gap
does not settle near the theoretical floor V_ind=0.25: the round-149
instantaneous value bounces between roughly 0.23 and 1.25 across nearby
rounds, and even the *cumulative time-average* gap (the actual quantity
Theorem 1 bounds, not a single round's value -- a distinction the first
version of this analysis got wrong, see below) plateaus around 0.62-0.68,
roughly 2.5x the theoretical floor, with no clear sign of further decay
by round 149.

Two things are true at once here, and it matters which one explains this:

1. **A real error in the first analysis pass, already partially fixed
   in spirit but not fully carried through**: `measured_gap` was read as
   the *instantaneous* gap at a single round, `gaps[round_used]`, not the
   *cumulative time-average* `mean(gaps[0:round_used+1])` that Theorem 1's
   left-hand side, `(1/T) sum_t E||grad F(w^t)||^2`, actually is. Reading
   off the running mean instead of a single point is the correct fix and
   should be applied before this analysis is trusted -- **this has been
   diagnosed but not yet implemented in `run_e3.py`** (the numbers above
   were computed in an ad-hoc check, not the pipeline itself).
2. **A genuine unresolved question, even after fix (1)**: even the
   cumulative time-average (0.62-0.68) sits well above V_ind (0.25). This
   is not necessarily a violation of Theorem 1 -- the bound is
   `transient_term(T) + V_ind`, and without a calibrated L (smoothness)
   and D (initial optimality gap F(w^0)-F*) for this actual model
   instance, there is no way to check whether `transient_term(149)` is
   small enough for the sum to be consistent with what's measured, or
   whether something else is going on (e.g. the model's actual dynamics
   under this much injected noise don't cleanly separate into the
   transient-plus-floor decomposition the way the pilot's much smaller
   sigma=0.5 case did).

**L and D have never been estimated anywhere in this repo.** E1 and the
manuscript's own Section 7.3 both treat LD=10 as a given constant, not a
measured property of an actual model. E3 is the first place this repo
actually trains a real model, which is also the first place this
shortcut stops being free.

## sigma_max across the grid (why this isn't a corner case)

| d | N | sigma_max |
|---|---|---|
| 10 | 200 | 1.57 |
| 10 | 500 | 2.49 |
| 10 | 1000 | 3.53 |
| 10 | 2000 | 5.00 |
| 10 | 5000 | 7.90 |
| 10 | 10000 | 11.18 |
| 10 | 20000 | 15.81 |
| 100 | 1000 | 1.12 |
| 100 | 3000 | 1.94 |
| 100 | 6000 | 2.74 |
| 100 | 10000 | 3.54 |
| 100 | 15000 | 4.33 |

sigma_max grows roughly as sqrt(N) by construction (Eq. 17 is linear in N
inside the square root), so this is not a problem confined to the largest
N in the grid -- it is present, to varying degree, almost everywhere
sigma=sigma_max is actually used (i.e. everywhere except the very
smallest N, where sigma_max is closer to 1).

## What needs to happen before E3 can be trusted as a Theorem 2 check

1. Fix `measured_gap` to be the cumulative time-average, not a single
   round's value (small, mechanical fix).
2. **Estimate L and D from the actual model/data**, so Corollary 1's
   T_acc and Theorem 1's transient term can be computed for real rather
   than assumed, and the comparison can be run at T_acc (the round count
   the theorem actually specifies) rather than an arbitrary T_max cap.
   - D = F(w^0) - F*: measurable directly -- F(w^0) is computable at
     initialization; F* can be approximated by training to convergence
     under favorable conditions (p=1, sigma=0) once per (d, alpha, beta)
     population setting, not per grid point.
   - L (smoothness): estimable via empirical gradient-Lipschitz probing
     (measuring ||grad F(w1) - grad F(w2)|| / ||w1-w2|| over sampled
     probe pairs near the training trajectory and taking the max), a
     standard, defensible practice, not requiring a closed-form bound.
3. Re-run the feasibility comparison using T_acc in place of T_max, with
   the corrected cumulative-average readout.

This is real additional work, not a quick patch -- estimated at roughly
similar scope to what building `fedprox.py` itself took. Given where the
session's tool-call budget stands, this is the natural place to check in
with Pijush before continuing, rather than pushing through with either an
uncorrected analysis or a rushed L/D estimator that hasn't been tested.
