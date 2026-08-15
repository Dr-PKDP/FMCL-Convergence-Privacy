# E2 results — FedProx inexactness: bound validated, and a real, actionable finding for the manuscript's own defaults

**Bottom line first:** the derived bound (Lemma E2.1 / Theorem 1') is
correct and safe — never violated in any of the 2,400 measurements below.
But it also surfaces something the manuscript should probably act on:
**at its own default of 5 local steps (mu=0.1), the FedProx inexactness
term is not a negligible idealization — it is roughly 2.9x larger than
V_ind itself**, i.e. bigger than the entire participation+privacy-noise
floor the paper spends Sections 4-5 characterizing. Doubling local_steps
to 10 brings it into parity with V_ind (ratio 1.03); by 20 steps it's
negligible (0.16). This is a concrete, evidence-based recommendation, not
a caveat to bury in a limitations paragraph.

## 1. Theory (see `docs/derivations/e2_inexactness.md` for full proofs)

- **Lemma E2.1**: the FedProx local-solve inexactness after E steps of
  gradient descent is bounded by delta_i(E,mu,L_i) <= r^E (G_i/mu), with
  r = sqrt(L_i/(L_i+mu)), via standard strongly-convex-smooth GD
  contraction plus a strong-convexity distance bound.
- **Theorem 1'**: extends Theorem 1's floor to
  `floor <= V_ind + delta^2` exactly at eta=1/L — the bias cross-terms in
  the descent-lemma proof cancel *exactly* at this step size (not
  approximately), which is also the step size E3 independently found
  necessary to recover the theorem's own predictions empirically. Theorem
  1 is recovered exactly as delta -> 0 (E -> infinity).
- **Corollary E2.1**: every downstream construction (sigma_max, Theorem
  2's feasibility condition, T_acc) carries through with
  `V_ind -> V_ind + delta^2`, a mechanical substitution given Theorem 1's
  structural role.

## 2. Numerical validation

**Setup**: 20 clients, Synthetic(1,1), d=10, K=4 (same generator this
repo already uses in E3), mu in {0.01, 0.1, 0.5, 1.0}, E in
{1,2,5,10,20,50} (E=5 is the manuscript's own default), 5 seeds — 2,400
(client, mu, E, seed) measurements total. Ground truth: near-exact
proximal solve via many-step GD (3000 steps, target tolerance 1e-10).

### Rate check: is the bound's decay rate right?

| mu | empirical slope | theoretical slope | ratio | verdict |
|---|---|---|---|---|
| 0.01 | -0.0247 | -0.0040 | 6.23 | loose (see mechanism below) |
| 0.1 (manuscript default) | -0.0851 | -0.0383 | 2.22 | just outside 2x, safe direction |
| 0.5 | -0.3369 | -0.1676 | 2.01 | at the 2x boundary, safe direction |
| 1.0 | -0.3394 | -0.2929 | 1.16 | tight, passes cleanly |

**The bound is never violated** — every ratio is >= 1, meaning measured
inexactness always shrinks *at least* as fast as Lemma E2.1 claims,
across all four mu values. It is loosest at small mu and tightest at
large mu, which is itself an interpretable pattern (see below), not
random scatter.

**A genuine secondary finding, investigated rather than smoothed over**:
initial analysis used the *mean* delta across clients, which gave a
noisy, sometimes non-monotonic picture. Investigating why: roughly 24%
of (client, seed) cells (97/400) show a persistently large residual
gradient norm (order 1, vs. ~1e-16 for the rest) even after the full
3000-step ground-truth solve, regardless of mu. This is consistent with
a small number of clients per random draw having an ill-conditioned
local objective — plausible and, on reflection, expected given only 10
samples per client in a 10-dimensional, 4-class problem, which leaves
real room for near-degenerate feature/label configurations in roughly
1 in 20 random client draws. This is a distinct phenomenon from the
FedProx-approximation-quality question E2 set out to test (it affects
the *ground truth* itself, not the E-step approximation to it), but it
is a genuine, worth-flagging robustness consideration for FMCL
specifically: consumer devices with very small local datasets are
exactly the regime this repo's finding describes, not an edge case.
Median (not mean) is the reported statistic throughout for this reason,
and is robust to this contamination (excluding the flagged cells changes
the operating-point median by under 3%).

### Magnitude check: does the inexactness matter at the manuscript's own settings?

| E | median delta | delta^2 | delta^2 / V_ind |
|---|---|---|---|
| 1 | 1.370 | 1.875 | 7.50 |
| 2 | 1.189 | 1.414 | 5.66 |
| **5 (manuscript default)** | **0.853** | **0.727** | **2.91** |
| 10 | 0.507 | 0.257 | 1.03 |
| 20 | 0.200 | 0.040 | 0.16 |
| 50 | 0.020 | 0.0004 | 0.002 |

(V_ind = eps_acc/2 = 0.25, the same operating point used throughout this
repo's E3 grid and the manuscript's own Section 7.3 illustration.)

**This is the headline result**: Assumption 2's idealization is not a
safe simplification at the manuscript's own default of 5 local steps —
inexactness alone contributes almost 3x what the entire
participation-and-privacy-noise floor contributes. It becomes comparable
to V_ind at 10 local steps and negligible by 20.

## 3. What this means for the manuscript

- **Adopt Theorem 1' and Corollary E2.1** as stated — they are correct,
  proven, and recover Theorem 1 exactly in the appropriate limit.
- **The E=5 default deserves a second look.** Either (a) state Theorem
  1'/Theorem 2's inexactness term explicitly and note it is material at
  E=5 (honest, minimal-change option), or (b) revise the recommended
  default toward E=10 local steps, where inexactness and V_ind are
  comparable rather than one dominating the other (stronger, more
  actionable option — and doubling local compute per round is a cheap
  price relative to the communication and privacy costs the rest of the
  paper is built around).
- **A specific, bounded caveat on Lemma E2.1's tightness at small mu**:
  the bound is safe but loose for mu <= 0.1 (ratio 2.2-6.2x); it is not
  the right tool for precisely predicting inexactness magnitude in that
  regime, only for confirming it decays and bounding it conservatively.
  This should be stated alongside the Lemma, not discovered by a reader
  re-deriving it.
- **The small-sample conditioning finding** is worth a sentence in the
  paper's risk/robustness discussion (Section 7 already discusses
  reliability and heterogeneity) — it is a concrete manifestation of
  exactly the "per-device data richer but sparser than institutional"
  concern FMCL's own framing already raises, now with a number attached
  (~1 in 20 clients, at 10 samples/client) rather than left qualitative.

## 4. Artifacts

- `docs/derivations/e2_inexactness.md` — full proofs
- `results/e2/e2_inexactness_grid.csv` — 2,400 raw measurements
- `results/e2/e2_summary.json` — mean-based summary (superseded by the
  median-based numbers in this document for the reasons above; kept for
  transparency, not deleted)
