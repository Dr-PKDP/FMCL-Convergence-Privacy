# EXPERIMENT_PLAN.md — FMCL Paper 2A (Convergence & Privacy Guarantees)

**Status:** PRE-REGISTERED. This document is written and committed *before* any
experiment listed in it is executed. Any deviation from what is written here
(parameter change, dropped run, changed success criterion) must be recorded
as a dated amendment in the "Amendments" section at the bottom, never by
silently editing the original entry. This mirrors the discipline used for
Paper 3's EXPERIMENT_LOG.md and exists for the same reason: if a reviewer or
someone later asks "why does the paper report N* = X", the answer must be
traceable to a specific script, commit, and log line, not reconstructed from
memory.

**Source manuscript:** `PaperA_ConvergencePrivacy_structure_1.docx`
("Convergence and Privacy Guarantees for Federated Learning Under Correlated
Device Dropout" — FMCL Paper 2A, split from the FMCL theoretical series).

**Origin of this plan:** a novelty/technical-contribution review conducted
2026-08-14 identified that the manuscript's most novel claim (Theorem 2,
the joint privacy–accuracy feasibility condition) is validated only by a
single plugged-in numerical example, while its least novel claim (Theorem 1's
1/p floor-scaling) carries the paper's only real simulation. The experiments
below are designed to correct that imbalance and to close specific gaps the
manuscript itself already flags in Sections 12–13.

---

## 0. Two-track compute constraint (record this before anything else)

The local development machine used for Track A work has network egress
restricted to an allow-list of software-registry domains (PyPI, npm, GitHub,
crates.io, Ubuntu archives). It cannot reach the R770 jump server
(`129.106.31.39` → `129.106.31.17`) — confirmed by direct connection test on
2026-08-14 (`/dev/tcp` connect to port 22 timed out / refused). It also has
no GPU, 1 vCPU, and 3.9 GB RAM.

Consequently:

- **Track A (local):** experiments cheap enough to run on 1 CPU core —
  synthetic populations, N ≤ a few thousand, convex or small models,
  closed-form/Monte-Carlo checks. Executed on the local machine, logged in
  `EXPERIMENT_LOG.md`, committed to this repository.
- **Track B (R770, GPU):** experiments needing real datasets/models at the
  scale Paper 3's infrastructure already supports (FEMNIST-ResNet, MIT-BIH
  ResNet). Scripts are written and locally syntax/logic-validated against
  tiny toy data first (so no server time is spent debugging), packaged for
  unattended `tmux` execution with skip-if-exists logic, and run on the R770
  separately, with results pasted back in `EXPERIMENT_LOG.md`-style format
  for analysis and write-up.

This split is recorded once here and referenced by ID (Track A / Track B) in
every experiment entry below rather than re-justified each time.

---

## 1. Experiment index

| ID | Title | Paper claim addressed | Track | Status |
|----|-------|------------------------|-------|--------|
| E1 | Subsampling amplification for between-device participation | Theorem 2 conservatism (§13.1) | A (theory + numerical check) | **DONE, fully closed.** See Amendment 1, `docs/derivations/e1_amplification.md`, `results/e1/`, and the resolved Q1 in `manuscript_patches/OPEN_QUESTIONS_FOR_AUTHOR.md`. |
| E2 | FedProx inexactness residual, explicit form | Assumption 2 (§13.1) | A | **DONE.** Theorem 1' proven (exact bias cancellation at eta=1/L). Bound never violated across 2400 measurements. Headline finding: at the manuscript's own default (mu=0.1, E=5), inexactness contributes ~2.9x V_ind -- NOT negligible. E=10 brings parity, E=20 negligible. Full writeup: `results/e2/E2_RESULTS.md`. |
| E3 | Theorem 2 feasibility-boundary simulation, synthetic scale | Theorem 2 (C2) — currently unvalidated | A | **DONE.** Zero-false-negative check passes (180/180 testable rows). Theorem 2 is safe but not tight: median 2.16x conservatism gap vs. theoretical V_ind, mechanistically traced to near-universal clipping bias (78-98% of updates clipped throughout the grid, r=0.66 correlation with the gap). Full writeup: `results/e3/E3_STATUS_NOTE.md`. |
| E4 | Markov-modulated correlated participation | Theorem 1b / Assumption 4 (§13.1, §2.2) | A | **DONE.** Theorem 1b'' proven (floor ≤ V_corr + (λG/p)²). Found and resolved a real subtlety along the way (naive single-round argument was wrong; manuscript's own §6 verification methodology is structurally blind to this effect, demonstrated concretely). Headline: population size dilutes ρ̄'s cost but NOT temporal correlation's cost. Full writeup: `results/e4/E4_RESULTS.md`. |
| E5 | Empirical ρ̄ from public device-availability data | §13.3 empirical gap | A (contingent on data access) | **DONE — feasibility scan succeeded.** Real trace found (FLASH, Yang et al. WWW'21, BSD-2-Clause, 1000 devices). Headline: empirical ρ̄ ≈ 0.09-0.11, **68-77x** the O(1/N) threshold — population averaging does NOT survive in this real population. Also yields empirical λ≈0.42-0.58 (ties directly to E4). Full writeup: `results/e5/E5_RESULTS.md`. |
| E6 | Feasibility nomograph for healthcare-scale models | Translational packaging of Theorem 2 | A | **DONE.** 360-cell grid (d up to 100,000, bracketing realistic compact-model sizes), basic + Path-A-tightened N*. Deliberately does NOT bake E2-E5's findings into the table (false-precision risk); states them as explicit safety margins instead. Full writeup: `results/e6/E6_RESULTS.md`. |
| E7 | Real-model feasibility-boundary experiment (FEMNIST-ResNet / MIT-BIH ResNet1D) | Theorem 1 + Theorem 2, nonconvex regime | B (R770 GPU) | **Script + mechanics complete, validated locally (7/7 checks). NOT executed** — requires resolving TODO(confirm) integration points and running on the R770. Target corrected to MIT-BIH ResNet specifically (FEMNIST-ResNet is unvalidated even under favorable conditions — demoted). See `src/e7_neural_feasibility/README.md`. |
| E8 | Manuscript structural repair (section numbering, table captions) | N/A — hygiene, not an experiment | — | Planned |

Each experiment below is specified to the level needed to re-run it
identically from a clean checkout: fixed parameters, swept parameters, seed
policy, sample size, output schema, and a **quantitative, pre-committed
success criterion** — stated before the run, not fitted after seeing results.

---

## E1 — Subsampling amplification for between-device participation

**Gap.** §13.1 of the manuscript states plainly that Theorem 2 omits privacy
amplification by subsampling because it is unclear whether the standard
amplification result — built for sampling *data points* from a fixed dataset
— transfers to FMCL's sampling of *devices* from a population each round.
This is flagged as the paper's main source of conservatism.

**Hypothesis.** FMCL's per-round device draw (each device independently
included with probability p, i.e. Poisson/Bernoulli sampling of *clients*)
is mechanically the same sampling primitive the standard RDP amplification
theorem (Wang, Balle & Kasiviswanathan 2019, ref [32]) uses for Poisson
subsampling of *records*, provided the privacy unit is redefined from
"one training example" to "one device-round contribution." Under that
redefinition the amplification bound should apply directly, tightening
Proposition 1 / Lemma 3 by replacing the linear-in-T RDP composition with
the amplified form, and shrinking the feasible-population threshold N*
reported in §7.3's numerical illustration (d=100, ε_acc=0.5, ε_p=8 → N≈10^5).

**Method (theory).** Re-derive Proposition 1 under Poisson-subsampled RDP:
state the privacy unit substitution explicitly, check the independence
condition the amplification theorem requires against Assumption 4
(independent Bernoulli participation — satisfied for Theorem 1's setting;
flag explicitly that it does *not* trivially extend to Theorem 1b's
correlated setting, which is a boundary this derivation must state rather
than paper over), and produce Theorem 2′ with the amplified constant.

**Method (numerical check, Track A).** Not a training simulation — a
closed-form comparison. For a grid of (p, σ, T_i) compute (a) the current
paper's non-amplified ε_i(T_i) from Eq. 15, and (b) the amplified ε_i(T_i)
from the new derivation, using the numerical RDP-to-DP conversion routine
implemented from scratch (no external DP library — the initial design assumed no
internet access to install `opacus`/`dp-accounting` beyond PyPI, and a
from-scratch implementation is auditable against the closed forms directly).
Cross-check the from-scratch implementation against Proposition 1's existing
closed form at p=1 (no subsampling), where the two must coincide exactly.

**Grid.** p ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}; σ ∈ {0.5, 0.8, 1.0, 1.5,
2.0, 3.0}; T_i ∈ {10, 50, 100, 500, 1000}; δ = 1e-5 fixed (matches §7.3).

**Success criterion (pre-committed).**
1. Sanity check: amplified and non-amplified ε_i(T_i) agree to within 1e-6
   relative error at p = 1.0 for every (σ, T_i) pair (this is a correctness
   gate on the implementation, not a scientific finding).
2. Reports, for the same (d, ε_acc, ε_p, δ, LD) operating point used in the
   manuscript's §7.3 illustration, the amplified population threshold N*′
   alongside the original N*≈10^5, with the ratio N*/N*′ as the headline
   number.
3. States explicitly, as an output artifact, the independence-assumption
   boundary (works cleanly under Theorem 1's iid participation; requires a
   stated extra assumption or is left open under Theorem 1b's correlated
   participation) — this boundary statement is a required output regardless
   of the numerical result, per §13.1's own framing of the open question.

**Est. runtime:** minutes (grid is O(7×6×5)=210 points, closed-form only).

---

## E2 — FedProx inexactness residual, explicit form

**Gap.** Assumption 2 treats the FedProx local solve as producing an update
whose participation-randomness mean exactly equals ∇F(w^t), with the true
proximal-solve inexactness "absorbed into the residual floor" without a
closed form. §13.1 names this a load-bearing idealization and an open
problem.

**Hypothesis.** For a finite local-step budget E (number of local SGD steps
towards the proximal objective, matching the manuscript's "five local
steps" in §11.1) and proximal coefficient μ, the inexactness of the FedProx
local solve relative to the exact proximal-point solution shrinks
geometrically in E at a rate governed by μ and the local smoothness L_i,
following the standard inexact-proximal-point analysis; this yields an
explicit additive term δ(E, μ, L_i) to append to Theorem 1's floor V_ind
rather than leaving it unnamed.

**Method (theory).** Derive δ(E, μ, L_i) from the contraction property of E
steps of gradient descent on the μ-strongly-convex-in-w proximal objective
minw Fi(w) + (μ/2)||w − w^t||². State the result as Lemma (new) and show the
revised Theorem 1′: floor ≤ V_ind + δ(E, μ, L_i), recovering the original
Theorem 1 as the E → ∞ (exact local solve) limit.

**Method (numerical validation, Track A).** Directly measure the inexactness
gap on the same convex multinomial-logistic local objective the manuscript
already uses (Section 11's model — reusing it here is intentional: this
experiment audits an assumption the manuscript's *own* existing model
depends on). For each client, solve the proximal objective (i) to near-exact
optimality via many GD steps as ground truth and (ii) via E ∈ {1, 2, 5, 10,
20, 50} local steps (the manuscript's own setting, E=5, is inside this grid),
and record ||local update after E steps − exact proximal solution||.

**Grid.** μ ∈ {0.01, 0.1, 0.5, 1.0} (0.1 matches manuscript's §11.1 setting);
E ∈ {1, 2, 5, 10, 20, 50}; 20 clients from the existing Scenario-C data
generator (§11.1's Synthetic(1,1)), 5 seeds each.

**Success criterion.**
1. Measured inexactness decays with E at a rate matching the derived
   geometric bound's exponent to within a factor of 2 (log-linear regression
   of log(gap) on E; compare fitted slope to the theoretical contraction
   rate log(1/(1+μ/L)) or the derived equivalent).
2. Produces a usable numeric statement: at the manuscript's own operating
   point (μ=0.1, E=5), report δ as a fraction of V_ind's other terms at the
   population sizes/participation rates already used in §11 — i.e., is the
   idealization in Assumption 2 negligible or material at the paper's own
   simulated scale? This is reported either way; a null result (idealization
   is fine, gap is negligible) is a valid and reportable outcome, not a
   failure.

**Est. runtime:** minutes, single core (20 clients × 6 E-values × 4 μ-values
× 5 seeds × small logistic regression).

---

## E3 — Theorem 2 feasibility-boundary simulation, synthetic scale

**Gap.** This is the central gap identified in the review: Theorem 2, the
paper's most novel result, is validated by nothing but one plugged-in
numerical example. No sweep exists showing the *predicted* feasibility
boundary (Eq. 19) tracks an *observed* boundary from an actual DP-FedProx
training+accounting run.

**Hypothesis.** For fixed (d, ε_acc, target δ), sweeping population size N
and per-device privacy budget ε_p, there exists an observed
feasible/infeasible boundary (does training empirically reach the
stationarity target within the actual privacy budget, tracked round-by-round
with the real RDP accountant from E1) that coincides with the boundary
predicted by Theorem 2's condition (Eq. 19), up to the log-scale tolerance
appropriate to a bound (not an equality) — i.e., every point Theorem 2
declares *infeasible* is empirically infeasible (no false negatives, since
Theorem 2 is a sufficient condition and must never promise something that
fails), and the empirically-feasible region should not extend far beyond
what Theorem 2 declares feasible (bounded false-negative rate quantifies how
conservative the bound is in practice — this is the direct empirical
counterpart of E1's theoretical tightening).

**Method (Track A, synthetic/logistic scale — same model family as the
manuscript's existing §11 simulation, extended to test Theorem 2 rather than
only Theorem 1).** Multinomial logistic regression, d ∈ {10, 50, 100, 200}
(the manuscript's §7.3 illustration used d=100; sweeping around it tests
whether the boundary's *shape* matches, which is what a feasibility
condition should be judged on), K=4 classes, FedProx local update (E=5,
μ=0.1, matching §11.1), Gaussian mechanism per Section 4, RDP accounting via
E1's implementation (both non-amplified, to match the current manuscript
exactly, and amplified, to show the E1 tightening's practical effect on this
same boundary).

For each (N, ε_p) grid point at fixed d and target ε_acc:
1. Compute the Theorem 2 predicted feasibility (Eq. 19: feasible / infeasible)
   using the manuscript's existing closed forms.
2. Run actual FedProx training with the corresponding σ = σ_max (Lemma 4) for
   T = T_acc rounds (Theorem 1's round requirement), tracking (a) whether the
   measured stationarity gap (avg over final 30 rounds, matching §11.3's own
   convention) reaches ε_acc, and (b) the actual cumulative per-device
   privacy loss via the real RDP accountant, checked against ε_p.
3. Label the grid point empirically feasible only if both (a) and (b) hold.

**Grid.** d ∈ {10, 100} (two points, primary=100 to match manuscript,
secondary=10 for a second regime); N ∈ {200, 1000, 5000, 20000, 100000}
(log-spaced, bracketing the manuscript's own N≈10^5 illustration); ε_p ∈
{1, 2, 4, 8, 16} (δ=1e-5 fixed); ε_acc = 0.5 fixed (matches manuscript);
p = 0.5 fixed (matches manuscript). 3 seeds per grid point (population size
makes single-client training cheap; the number of grid points is what costs
time, not per-run cost).

**Compute budget check.** 2 (d) × 5 (N) × 5 (ε_p) × 3 (seeds) = 150 runs.
Each run trains a d-dimensional multinomial logistic model for up to
T_acc rounds — T_acc itself depends on d via the L(F⁰−F*) term, must be
computed per-config, not assumed; this is recorded in the run manifest.
Runtime per run on 1 CPU core is expected to be seconds to low minutes;
the full grid is expected to fit in a single run, but if not, this
plan explicitly authorizes running it across multiple runs with
skip-if-exists resumption (same discipline as Paper 3's R770 runs), logging
partial completion honestly rather than silently truncating the grid.

**Success criterion (pre-committed).**
1. **Zero false negatives** at the non-amplified boundary: every grid point
   Theorem 2 labels infeasible must be empirically infeasible. A single
   violation is a serious, reportable finding (it would mean Theorem 2 as
   stated is not actually sufficient) and must not be smoothed over — if
   found, this must halt and be flagged for review before any manuscript
   language is drafted from this experiment.
2. Quantify the conservatism gap: for each d, find the empirical boundary
   N_empirical(ε_p) and compare to N*(ε_p) predicted by Theorem 2 and to
   N*′(ε_p) predicted by the E1-amplified Theorem 2′. Report the ratio
   N*/N_empirical and N*′/N_empirical as the headline result — this is the
   number that goes in the paper as "Theorem 2 is conservative by a factor
   of approximately X at this operating point, and the amplified version
   narrows that to Y."
3. Full grid (points, pass/fail, measured stationarity, measured ε_i) saved
   as a single CSV — this is the artifact a reviewer could re-plot.

**Est. runtime:** likely tens of minutes to a few hours on 1 CPU core,
depending on T_acc at small ε_acc; will be measured and logged, with actual
per-run wall time recorded in the log so future runs can be estimated
accurately rather than guessed.

---

## E4 — Markov-modulated correlated participation

**Gap.** Theorem 1b's ρ̄ is an average-pairwise-correlation *scalar*
summary; the manuscript itself calls this "exact for the equicorrelated
model... an approximation for arbitrary correlated participation," and
names Rodio et al.'s Markov-chain treatment as strictly more general but
not coupled to a privacy budget. §13.1 lists non-stationary/correlated
availability as open.

**Hypothesis.** Modeling each device's availability as a two-state
(available/unavailable) Markov chain with device-level stationary
availability p and a persistence parameter τ (self-transition probability,
τ=0.5 recovering the iid Bernoulli case of Theorem 1, τ→1 approaching
"available all day or none of the day" block correlation) yields a
convergence floor whose inflation factor is again computable in closed
form, reducing to Theorem 1b's (1+(N−1)ρ̄) under an equicorrelation
special case and generalizing beyond it — importantly, a Markov model
supplies *within-device temporal* correlation (a device available now is
more likely available next round), which is explicitly named in §13.1 as
"a separate and, for long campaigns, potentially more important effect"
that ρ̄ does not capture at all.

**Method (theory).** Derive the aggregation-error variance under the
Markov-modulated participation model (paralleling the existing Lemma 1/
Lemma 2 proof structure: unbiasedness from marginals, variance from the
stationary pairwise/cross-time covariance of the chain). State the new
Lemma/Theorem, show the reduction to Theorem 1b at the appropriate limit,
and identify the new quantity (equivalent of ρ̄) the Markov model
contributes: a *within-device* autocorrelection term absent from the
current model.

**Method (numerical validation, Track A — same style as the manuscript's
existing §6.4 Monte Carlo check, extended).** Simulate the aggregation step
in isolation (not full training — this directly extends the manuscript's own
existing verification method, which the manuscript describes as testing
"the variance identities directly, free of the optimization noise that
obscures them in a full training run") at N=200 (matching the manuscript's
existing check), drawing participation indicators from the two-state Markov
chain instead of iid/equicorrelated Bernoulli, across a grid of persistence
values, and compare empirical aggregation-error second moment against the
new closed form.

**Grid.** p ∈ {0.1, 0.2, 0.5, 1.0} (matches manuscript's existing grid);
τ (self-transition/persistence) ∈ {0.5 (=iid), 0.6, 0.7, 0.85, 0.95}; 10^5
independent realizations per grid point (matches manuscript's existing
Monte Carlo sample size exactly, for direct comparability).

**Success criterion.**
1. At τ=0.5 (the iid special case), the Markov-model closed form must
   coincide with Lemma 1's existing V_ind to within the same 0.2% tolerance
   the manuscript already reports for its own Monte Carlo check — this is
   the correctness gate.
2. For τ > 0.5, empirical aggregation-error variance matches the new closed
   form within 1% (looser than 0.2% because within-device temporal
   correlation is a smaller-sample-per-point effect than the original
   check; this looser tolerance is stated here in advance, not chosen after
   seeing results).
3. Reports how much additional floor inflation the within-device temporal
   channel contributes beyond what ρ̄ alone predicts at matched "looks
   similar in aggregate" operating points — directly answering whether
   §13.1's suspicion ("potentially more important effect") is borne out
   quantitatively.

**Est. runtime:** minutes to tens of minutes (grid is 4×5=20 points ×
10^5 draws of a simple two-state chain).

---

## E5 — Empirical ρ̄ from public device-availability data (contingent)

**Gap.** §13.3: "The value of ρ̄ for a real FMCL device population has never
been measured."

**Method — feasibility scan first, execution second.** Before committing to
this experiment, the first step is to search for a legitimately obtainable public
dataset of smartphone charging/connectivity/availability traces (e.g.,
device-analyzer-style corpora, published federated-learning availability
traces accompanying papers such as Yang et al.'s on-device participation
studies) reachable from within the local machine's allowed domains (GitHub raw
content, PyPI-packaged datasets). This is genuinely uncertain — most such
corpora (Device Analyzer, LiveLab, NetSense) require data-use agreements
the local machine cannot execute. **If no such dataset is reachable, this
experiment is marked BLOCKED, not silently dropped**, and the manuscript
keeps its existing honest statement that ρ̄ has not been measured, with a
note that a scan was attempted and what was found.

**If a dataset is found:** compute pairwise participation-indicator
correlation across devices at the trace's native time resolution, resampled
to the round length representative of FMCL scheduling (the manuscript's own
"overnight charging window" framing), report the empirical ρ̄, and check it
against the O(1/N) condition Theorem 1b identifies as the regime where
population-averaging benefit survives.

**Success criterion.** Either (a) a defensible empirical ρ̄ estimate with
its data source fully documented and licensing terms checked, or (b) an
honest BLOCKED status with the search trail recorded, so a repeat effort
doesn't redo the same failed search.

**Est. runtime:** search, tens of minutes; analysis if unblocked, minutes.

---

## E6 — Feasibility nomograph for healthcare-scale models

**Gap/motivation.** Pure translational packaging: Theorem 2 is a condition,
not a lookup table. A reviewer (and the sibling healthcare
paper) benefits from seeing N* computed across a realistic grid rather than
one illustrative point.

**Method (Track A, pure computation, no training).** Using the manuscript's
own closed forms (Eq. 17–19) plus, once available, E1's amplified variant,
compute N*(d, ε_acc, ε_p) over a grid representative of the compact models
named across the FMCL series (e.g., d spanning small CNN/ResNet1D
head-plus-adapter parameter counts relevant to ECG/arrhythmia classifiers,
rather than d=100 as an arbitrary illustration) at fixed δ=1e-5 and a small
number of (LD) initial-optimality-gap settings bracketing plausible values.
Render as a heatmap/contour figure (matplotlib, no external service) plus
the underlying CSV table.

**Grid.** d ∈ {10, 50, 100, 500, 1000, 5000} (log-spaced, bracketing
realistic compact-model dimension counts); ε_acc ∈ {0.1, 0.5, 1.0}; ε_p ∈
{1, 2, 4, 8, 16}; δ=1e-5 fixed; LD ∈ {1, 10, 50}.

**Success criterion.** A complete, reproducible grid (script + CSV + figure)
that a reader can regenerate from the repo in under a minute, with every
manuscript-quoted N* traceable to a specific row.

**Est. runtime:** seconds to low minutes (closed-form grid, no training).

---

## E7 — Real-model feasibility-boundary experiment (Track B, R770)

**Gap.** Everything in the current manuscript and in E1–E6 above is either
theory or a convex/synthetic-scale check. The manuscript's own §11.4 admits
"the model is convex per client and small... the data is synthetic... each
is a gap that only a real deployment can close." A nonconvex, real-dataset
run is the strongest single addition available and Paper 3's infrastructure
(H200 GPUs, FEMNIST-ResNet already built and verified, MIT-BIH pipelines
already working) makes it achievable without new infrastructure work.

**Hypothesis.** The qualitative predictions of Theorem 1 (1/p floor scaling)
and the E1/E3-refined Theorem 2 feasibility boundary hold in direction and
rough magnitude on a real nonconvex model (FEMNIST-ResNet or MIT-BIH
ResNet1D with per-device DP-SGD), even though the formal bounds were derived
under Assumption 2's convex-surrogate idealization — i.e., the theory's
practical usefulness extends beyond the regime it was proven in, which is
the claim a Q1 reviewer will actually want tested.

**Method.** Full specification lives in `src/e7_neural_feasibility/` as a
standalone, tmux-ready script following Paper 3's established pattern
(per-run config hash, skip-if-exists, exact package versions logged,
`torch.set_num_threads` / OMP pinning per Paper 3's established CPU-thread
fix if run on shared hardware, results appended to a CSV rather than
overwritten). Two candidate model/dataset pairs, matching what Paper 3
already has built and verified:

- **FEMNIST-ResNet** (already built and verified per Paper 3 state, not yet
  run) — natural per-writer non-IID partition, image classification,
  moderate d.
- **MIT-BIH ResNet1D_PTB** (already confirmed to converge decisively in
  Paper 3, 8 seeds, statistically significant) — physiological time series,
  directly relevant to the FMCL healthcare narrative, and reuses a model
  Paper 3 has already validated converges under favorable conditions, which
  is exactly the baseline this experiment needs before adding dropout and
  privacy noise on top.

Sweep: participation rate p ∈ {0.2, 0.5, 0.8, 1.0}; privacy noise σ ∈
{0, σ_max/2, σ_max} (σ_max computed per Lemma 4 using measured G=C and a
measured/estimated L for the real model — this measurement is itself a new
artifact, since L is currently only a symbol in the manuscript, never
estimated); 3 seeds minimum per config (Paper 3's own established practice
for statistical significance uses up to 8; this experiment defaults to 3 for
budget reasons and documents the gap explicitly rather than silently
under-powering it relative to Paper 3's own standard).

**Success criterion.** (a) Floor ordering by p replicates qualitatively
(lower p → higher measured floor) as in the manuscript's existing synthetic
Fig. 3/4/5; (b) quantify whether the 1/p linear-scaling fit (Pearson r, same
statistic the manuscript already reports) holds at a comparable strength to
the synthetic scenarios' r=0.949–0.999, or degrades — a degraded but still
positive r is a valid, reportable, non-catastrophic outcome that directly
answers the manuscript's own open question rather than needing to hit an
arbitrary bar.

**What can be done without server access:** write the complete script;
validate its control flow and every formula against E1–E4's already-tested
implementations by running it on a tiny CPU-only synthetic stand-in dataset
(a few hundred samples, 2 clients, 3 rounds) to catch bugs before any server
time is spent; produce the exact `tmux` launch command and the exact
skip-if-exists / logging contract the R770 run should follow, matching
`EXPERIMENT_LOG.md`'s existing schema so results paste back in directly.
**This is not executed on the R770 from the local machine.**

**Est. runtime (on R770, informational only, to be confirmed by actual
run):** rough order-of-magnitude estimate given Paper 3's own reported
per-seed runtimes for these exact models — low hours per full sweep on one
H200, refined once first actual wall-clock numbers are available.

---

## E8 — Manuscript structural repair (not an experiment; hygiene)

Section numbering currently jumps 1→2→4→6→7→11→12→13→14 with sections
3/5/8/9/10 absent (residue of the paper being split from a larger
structure), and table captions are mismatched (a table introduced as
"Table 8" is captioned "Table 7. Standing assumptions..." directly beneath
it; "Table 1" is used for two different tables). This must be renumbered
contiguously before submission regardless of which experiments above are
incorporated. Tracked here so it isn't forgotten once the more interesting
experimental work is underway — logged as a checklist item in
`manuscript_patches/`, not a simulation, so it gets no seed/grid/success
criterion, only a done/not-done state.

---

## Reporting discipline for all experiments above

- Every run writes a machine-readable record (CSV/JSON row) *and* a
  human-readable line in `EXPERIMENT_LOG.md` before any prose summary is
  written from it.
- No result is described in prose (here, in chat, or eventually in the
  manuscript) until its underlying artifact exists on disk and is committed.
- A negative or null result (bound not tight, mechanism doesn't transfer,
  amplification boundary genuinely doesn't hold for correlated
  participation) is reported with the same weight as a positive one. This
  plan treats "we checked and it doesn't work" as a valid, useful outcome —
  consistent with stating limitations honestly rather than soft-pedaling
  them, which is the standing practice for this paper series.

## Amendments

### Amendment 1 — 2026-08-14, before E1 execution

Two corrections discovered while building `src/common/privacy.py`, before
any E1 result was generated (so this amendment precedes, and shapes, E1's
actual run rather than being a post-hoc excuse):

1. **PyPI is in fact reachable from the local machine** (it is on the documented
   allow-list; the original plan text for E1 said "no internet access to
   install opacus/dp-accounting beyond PyPI," which is self-contradictory —
   PyPI access is exactly what's needed and exists). `dp-accounting`
   (Google's audited RDP/PLD accounting library, used in production DP
   systems) installs cleanly. E1's numerical work now uses this library
   as an independent cross-check on the from-scratch closed-form
   reproduction, rather than the from-scratch implementation being the only
   source of the amplification number. This is strictly better practice
   than originally planned (a bespoke-only implementation of a subtle
   subsampled-RDP bound is exactly the kind of thing worth having a second,
   independently-authored implementation to check against).

2. **A second, orthogonal tightening was found that the original E1 design
   did not anticipate.** Reproducing Proposition 1 (Eq. 15) against
   `dp_accounting`'s `RdpAccountant` at matched parameters (T_i=100, σ=1.0,
   δ=1e-5) gave a persistent ~2% gap that did *not* close under a denser
   RDP-order grid, ruling out numerical resolution as the cause. Reading the
   library source (`rdp_privacy_accountant.py`, `compute_epsilon`) shows why:
   it implements the **improved RDP→DP conversion of Canonne, Kamath &
   Steinke (arXiv:2004.00010, Prop. 12)** — ε(λ) = RDP(λ) + log(1−1/λ) −
   log(δλ)/(λ−1) — not the **basic Mironov (2017) conversion** — ε(λ) =
   RDP(λ) + log(1/δ)/(λ−1) — that the manuscript's Proposition 1 derivation
   explicitly uses (citing the same Mironov 2017 reference, [12], for the
   RDP mechanism but the *older* conversion bound for the composition step).
   This means there are now **two independent, orthogonal tightening paths**
   for Theorem 2, not one:
   - **Path A (conversion formula):** swap Mironov's basic bound for the
     Canonne–Kamath–Steinke bound. This requires *no* new assumption about
     FMCL's participation model — it is a strict improvement available
     for free, and should be adopted regardless of what happens with
     amplification.
   - **Path B (subsampling amplification):** the originally planned
     device-level Poisson-subsampling argument, which *does* require the
     participation-model boundary discussion in the original E1 entry above.

   E1's scope is revised to quantify both paths **separately** before
   quantifying them together, so the manuscript can report each
   contribution's individual magnitude rather than one conflated number.
   This is a strengthening of E1, not a redefinition of its goal (the goal —
   tighten Theorem 2, report the new N* — is unchanged); it is recorded here
   because the *method* changed materially from what was pre-registered.

No other experiment (E2–E8) is affected by this amendment.

### Amendment 2 — 2026-08-14, before E3 execution

Building `src/common/fedprox.py` (vectorized multinomial-logistic FedProx
training) and calibrating its throughput on the local machine's single CPU core
showed the originally-planned E3 grid (d×N×eps_p×seeds run as independent
full training runs) would take multiple hours — e.g. d=100, N=30,000 alone
costs ~500-600s per 200-round run, and the plan called for that cell
repeated across 5 eps_p values and 3 seeds.

Two changes, both recorded here before generating any E3 result so the
final numbers are traceable to the design that produced them, not
reverse-justified afterward:

1. **eps_p no longer requires a separate training run.** Re-reading Lemma
   4 (Eq. 17): sigma_max depends only on (N, d, eps_acc, p) — *not* on
   eps_p. eps_p only enters afterward, through T_priv(sigma_max, eps_p),
   a closed-form (or accountant) calculation with no training involved.
   So for a fixed (N, d, seed), one training run at sigma=sigma_max,
   recording the full per-round stationarity trajectory, is enough to
   answer the feasibility question for *every* eps_p in the grid: read the
   trajectory's value at round T_priv(sigma_max, eps_p)/p (capping at a
   fixed T_max once the floor has visibly stabilized, since Theorem 1's
   floor is asymptotic and Section 11's own pilot -- reproduced here --
   shows it stabilizes by ~T=100). This collapses the grid from
   N x eps_p x seeds independent runs to N x seeds runs, a 4-5x reduction,
   with no loss of information (eps_p is evaluated post-hoc from the same
   recorded trajectory, not approximated).

2. **d=10 and d=100 swap primary/secondary roles from the original plan.**
   The original plan called d=100 primary (matching the manuscript's own
   §7.3 headline illustration) and d=10 secondary. Measured cost scales
   roughly with d^0.85 in addition to linearly with N (d=100 costs ~4.4x
   d=10 at matched N), so d=10 is designated the primary, fine-grained
   grid here, with d=100 run as a coarser confirmatory check at fewer N
   points -- enough to confirm the trend generalizes to the manuscript's
   own dimension, not to characterize it as finely. This is purely a
   compute-budget decision, not a scientific one: the manuscript's own
   headline number (N*~15,255 at d=100) is still checked, just with fewer
   surrounding grid points than d=10 gets.

Revised grid (see `src/e3_theorem2_feasibility/run_e3.py` for the
executed version, which is authoritative if this text and that script
ever diverge):
- Shared: eps_acc=0.5, p=0.5, delta=1e-5, mu=0.1, local_steps=5, lr=0.5,
  C=1 (per the resolved Q1), T_max=150 rounds (stabilization confirmed by
  pilot at T~100-150; see EXPERIMENT_LOG.md pilot entry), eval_every=1.
- eps_p grid (evaluated post-hoc, no extra runs): {2, 4, 8, 16}.
- d=10 (primary): N in {200, 500, 1000, 2000, 5000, 10000, 20000}, 2 seeds.
- d=100 (secondary): N in {1000, 3000, 6000, 10000, 15000}, 2 seeds.
- Both the manuscript's basic (Mironov) and the Path-A-tight T_priv are
  evaluated against the same recorded trajectories, so E3 also functions
  as an empirical check on E1's Path A tightening, not just on Theorem 2
  as originally stated.

If time budget allows, this can be extended with more N points or a
third seed in a follow-up pass; skip-if-exists logic in the run script
(row already present in the output CSV for that exact (N,d,seed) key) is
what makes that safe to do incrementally across multiple runs
without re-running completed cells.
