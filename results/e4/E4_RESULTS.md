# E4 results — within-device temporal correlation: resolved, mechanism identified, bound validated but loose

**Bottom line first:** the manuscript's own speculation ("within-device
temporal correlation is a separate and, for long campaigns, potentially
more important effect") is correct, and now has a mechanism and a
number attached to it. The key qualitative finding: **population size
can rescue you from cross-device correlation's cost (rho-bar's term, if
rho-bar = O(1/N)) but cannot rescue you from within-device temporal
correlation's cost** — the (lambda*G/p)^2 penalty in Theorem 1b'' does
not shrink with N, unlike rho-bar's term. This is a precise, useful
distinction for a deployment operator, not just "both hurt."

## 1. Theory (full derivation: `docs/derivations/e4_markov_correlation.md`)

- Modeled per-device availability as a 2-state Markov chain parametrized
  by (p, lambda), lambda = lag-1 autocorrelation, lambda=0 recovering
  Assumption 4's i.i.d. baseline exactly.
- **Found and resolved a real subtlety**: a naive argument ("single-round
  cross-sectional variance is unaffected by a device's own temporal
  memory") is only half right. The debiasing factor 1/(Np) is calibrated
  to the MARGINAL p, but under Markov correlation, a device's
  CONDITIONAL participation probability given the realized history
  deviates from p — bounded but nonzero (Lemma E4.1/E4.2). This is
  exactly why the manuscript's own existing single-round Monte Carlo
  verification (Section 6) is structurally blind to this effect: it
  draws from the marginal directly, never conditioning on history.
- **Theorem 1b''**: floor <= V_corr + (lambda*G/p)^2, via the identical
  bias-variance-cancellation mechanism as E2's Theorem 1' (exact
  cross-term cancellation at eta=1/L). Recovers Theorem 1b exactly as
  lambda -> 0. Gives the paper a coherent, unified structure: three
  separate stated idealizations (subsampling amplification in E1's
  investigation, FedProx inexactness in E2, temporal correlation here)
  all resolve into the same additive-floor-correction pattern, from the
  same underlying proof mechanism.

## 2. Numerical validation (all four checks pass or confirm as designed)

**Checks 1-2 (Lemma E4.1 zero-mean, Lemma E4.2 bound)**: 15/15 pass for
both, across p in {0.1,0.2,0.5} x lambda in {0.1,0.2,0.35,0.5}. The bound
is never violated. Under RANDOM (non-adversarial) gradient directions,
observed ||beta_t|| sits consistently 13-23x below the worst-case bound
lambda*G/p (table in the raw results) — meaning the bound, like the
manuscript's own Cauchy-Schwarz-based rho-bar term (which uses the
identical worst-case technique, <g_i,g_j> <= G^2), is a valid safety
bound but not tight under typical, non-adversarially-aligned local
update directions. This is not a new weakness E4 introduces; it is
inherited from the same proof style the paper already uses elsewhere.

**Check 3 (single-round null, manuscript's own methodology)**: confirms
directly and concretely that lambda has NO detectable effect on a
single-round Monte Carlo check drawing from the marginal (relative error
under 1.8% across all tested (p,lambda) pairs, no trend with lambda) —
exactly as the theory predicts, and exactly why this effect needed a
different verification approach than the one already in the manuscript.

**Check 4 (real multi-round training, the project's fedprox.py
harness)**: floor increases monotonically with lambda in actual training
(lambda=0: ~0.0026-0.0028; lambda=0.2: ~0.0038-0.0044; lambda=0.4:
~0.0046-0.0053, averaged over 2 seeds each) — **confirms the qualitative
direction correctly**. The (lambda*G/p)^2 term substantially
*overestimates* the magnitude of this inflation (predicting 0.16 and
0.64 respectively, vs. observed differences on the order of 0.001-0.003)
— consistent with, and roughly the right order of magnitude given,
checks 1-2's finding that the underlying per-round bound is loose by
~15-20x under random directions (a ~15-20x linear looseness compounds to
~200-400x once squared, which is the right ballpark for the gap observed
here).

## 3. What this means for the manuscript

- **Adopt Theorem 1b'' as a valid, safe extension** — it is proven,
  reduces correctly to Theorem 1b, and gives the precise mechanistic
  answer (bounded, N-independent penalty) to a question the manuscript
  currently only poses.
- **State the tightness honestly, the same way E3's clipping-bias
  finding is stated**: the bound is a worst-case guarantee, loose by
  roughly an order of magnitude (squared: two orders) under typical,
  non-adversarial participation-history/gradient-direction alignment.
  This is consistent with, not a departure from, the existing rho-bar
  term's own proof technique.
- **The clean, quotable finding for the paper**: scale (larger N) helps
  against cross-device correlation but not against within-device
  temporal correlation — this is a genuinely new, useful distinction for
  a deployment operator's decision-making, and is worth a sentence in
  Section 7's discussion of what a device population can and cannot buy.
- **A natural connection to E5** (empirical rho-bar): if E5 finds a
  usable participation trace dataset, the same trace could estimate
  lambda alongside rho-bar, giving both halves of the correlation
  picture from one data source rather than treating them as separately
  unmeasured quantities.

## 4. Artifacts

- `docs/derivations/e4_markov_correlation.md` — full derivation
- `src/common/participation.py` — reusable iid/Markov generators
  (validated against target marginal and lag-1 autocorrelation before use)
- `results/e4/e4_results.json` — all four checks' raw output
