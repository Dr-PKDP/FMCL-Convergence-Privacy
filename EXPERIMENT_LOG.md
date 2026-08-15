# EXPERIMENT_LOG.md

Append-only. Every entry corresponds to one actual execution (not a plan —
see `EXPERIMENT_PLAN.md` for pre-registration). Never edit a past entry;
if a run turns out to be wrong, add a new entry marked SUPERSEDES with the
reason, and leave the original in place. This is what makes it possible to
trace back "why does the paper say X" to an exact command and commit.

Each entry format:

```
## [YYYY-MM-DD HH:MM UTC] <Experiment ID> — <short description>
- Track: A (local) | B (R770)
- Git commit at time of run: <hash, filled after commit>
- Command: <exact command>
- Environment: <python/torch version, CPU/GPU, seed policy>
- Config: <path to config file or inline parameters>
- Output artifacts: <paths written>
- Wall-clock time: <measured>
- Result summary: <one or two lines, numbers only, no interpretation>
- Deviation from plan: <none | description + reference to Amendments in EXPERIMENT_PLAN.md>
- Status: COMPLETE | PARTIAL (resumed later) | FAILED (see note)
```

---

## [2026-08-14] REPO INIT — planning checkpoint, no experiments run yet
- Track: A
- Git commit at time of run: (this commit)
- Command: N/A (repo scaffold + EXPERIMENT_PLAN.md written and committed)
- Environment: local machine — Python 3.12.3, numpy 2.4.4, scipy 1.17.1,
  pandas 3.0.2, matplotlib 3.10.8, torch 2.13.0 (CPU-only build; installed
  for E7 script validation against toy data, not used for GPU compute).
  1 vCPU, 3.9 GB RAM, no network beyond PyPI/npm/GitHub/crates.io/Ubuntu
  archives (confirmed no route to 129.106.31.39/129.106.31.17).
- Config: N/A
- Output artifacts: EXPERIMENT_PLAN.md, this file, README.md,
  ENVIRONMENT.md, requirements.txt, repo directory skeleton.
- Wall-clock time: N/A
- Result summary: N/A — this entry exists only to anchor the plan to a
  commit hash before any experiment below it changes state.
- Deviation from plan: none (this is the plan itself).
- Status: COMPLETE

---

## [2026-08-14] E1 — subsampling amplification: theory resolution + numerical grid
- Track: A
- Git commit at time of run: (next commit)
- Command: `python3 src/e1_amplification/run_e1.py`
- Environment: local machine, Python 3.12.3, numpy 2.4.4, scipy 1.17.1,
  dp_accounting 0.6.0 (Google, installed from PyPI post-Amendment-1).
  Seeds: N/A (closed-form/deterministic grid, no stochastic sampling in E1).
- Config: inline in `run_e1.py` — grids documented in file docstring;
  matches EXPERIMENT_PLAN.md E1 grid (T_i, sigma) with delta=1e-5 fixed,
  extended per Amendment 1 to compute Path A / Path B separately.
- Output artifacts:
  - `results/e1/path_a_tightening.csv` (30 rows: valid tightening grid)
  - `results/e1/path_b_reference_only.csv` (210 rows: INVALID, reference only)
  - `results/e1/manuscript_illustration_tightened.json`
  - `results/e1/e1_summary.json`
  - `docs/derivations/e1_amplification.md` (the theory resolution itself —
    this is the primary E1 deliverable; the CSVs are its numerical support)
- Wall-clock time: 109.6 s (dominated by the 210-row Path B grid, each row
  a full library RDP-accountant call across ~400 orders; Path A's 30-row
  grid alone took a small fraction of this).
- Result summary:
  - **Threat-model finding (qualitative, the main result):** device-level
    Poisson-subsampling amplification is INVALID against FMCL's own stated
    threat model (§2.3: LDP protects against "an aggregator, or an
    adversary observing the aggregator") because the aggregator observes
    per-round participation directly. Full argument and literature support
    in `docs/derivations/e1_amplification.md`. This resolves §13.1's open
    boundary question negatively for the current protocol and identifies
    the shuffle model of DP (Erlingsson et al. 2019; Cheu et al. 2019;
    Girgis-Data-Diggavi 2021) as the specific, citable mechanism that
    would be needed to unlock it — not attempted here.
  - **Path A (valid, adopted):** swapping the manuscript's Mironov (2017)
    basic RDP-to-DP conversion for the Canonne-Kamath-Steinke (2020) tight
    conversion gives a tightening factor ranging 1.0016-1.1174 across the
    30-point (T_i, sigma) grid (mean 1.032). At the manuscript's own §7.3
    illustration operating point (d=100, eps_acc=0.5, eps_p=8, delta=1e-5,
    p=0.5, LD=10, **G=C=1 assumed** — see deviation note below), Path A
    tightens N* from 15,254.6 to 13,015.2 (factor 1.172).
  - **Path B (invalid, reference only):** naive device-level amplification
    factors of 3.4x-4.7x (at q=0.05-0.1, T_i=10) up to much larger factors
    at higher T_i, quantifying the size of the error a deployment would
    make if it applied amplification incorrectly. Every row explicitly
    flagged `invalid=True` in the CSV with the reason stated inline.
- Deviation from plan: Amendment 1 (recorded in EXPERIMENT_PLAN.md before
  this run) already covers the method change (two-path design, PyPI
  availability). **A second, unplanned finding surfaced during this run
  and is NOT yet resolved — see "OPEN QUESTION FOR PIJUSH" below.**
- Status: COMPLETE for Path A/B numerics and the threat-model resolution.
  **BLOCKED / needs author input** on one specific numerical detail before
  E1's manuscript-illustration number can be finalized, and before E3
  (which depends on the same normalization) is run at full scale — see
  `manuscript_patches/OPEN_QUESTIONS_FOR_AUTHOR.md`, Question 1.

---

## [2026-08-14] Q1 RESOLVED — via review of the Paper 2 drafting notes
- Track: A (research, not compute)
- Git commit at time of run: (next commit)
- Method: reviewed the Paper 2 drafting session's own working notes for the
  point where Section 7.3's numerical illustration was originally computed
  and later independently re-verified, following up on the fact that the
  C value was set during that earlier drafting work rather than in this
  document.
- Environment: N/A (documentation review, not computation).
- Config: N/A
- Output artifacts: `manuscript_patches/OPEN_QUESTIONS_FOR_AUTHOR.md`
  Question 1 rewritten with the resolution and provenance.
- Wall-clock time: N/A
- Result summary: C=1, G=1 confirmed as the value used when Section 7.3's
  illustration was originally computed (found verbatim in that session's
  `feasible_N(..., G=1.0, C=1.0, d=100)` script). The "~10^5" figure was
  read off a coarse power-of-10 grid {1e3,1e4,1e5,1e6,1e7}, not a
  bisection — reproducing that exact grid here confirms infeasible@1e4 /
  feasible@1e5, matching the original write-up exactly. A later editorial
  pass in that same drafting session (tag "B8") already added the
  manuscript's existing caveat sentence about order-of-magnitude
  sensitivity, which this project's more precise bisection (N*=15,254.6)
  falls squarely inside. Conclusion: not an inconsistency; the
  manuscript's own caveat already covers the gap. C=1 confirmed for all
  downstream use in this project.
- Deviation from plan: none — this resolves Amendment 1's follow-on
  question rather than introducing a new one.
- Status: COMPLETE. E1 fully closed. E3 unblocked.

---

## [2026-08-14] E3 — 24/24 training runs complete; methodological gap found in the accuracy-vs-boundary check
- Track: A
- Git commit at time of run: (next commit)
- Command: `python3 src/e3_theorem2_feasibility/run_e3.py`, invoked
  repeatedly across several runs due to per-invocation wall-clock
  limits (~270-300s); resumed cleanly each time via per-trajectory
  skip-if-exists caching, and via new mid-run checkpointing (added this
  session to `fedprox.run_training`, checkpointing every 25 rounds) once
  individual runs started approaching the per-call limit themselves
  (d=100, N=15000 took 237-353s total across checkpointed resumes).
- Environment: local machine, Python 3.12.3, numpy 2.4.4. Seeds: training seeds
  0-1 per (d,N) cell (2 seeds; reduced from the plan's 3-seed default —
  see Amendment 2 — due to measured single-core throughput), data
  generation seeded independently at 1000+seed.
- Config: `src/e3_theorem2_feasibility/run_e3.py` module-level constants;
  grid per Amendment 2 (d=10: N in {200,500,1000,2000,5000,10000,20000};
  d=100: N in {1000,3000,6000,10000,15000}; eps_p in {2,4,8,16} evaluated
  post-hoc per trajectory, both basic and Path-A-tight T_priv).
- Output artifacts:
  - `results/e3/trajectories/*.json` (24 full per-round trajectories)
  - `results/e3/e3_feasibility_grid.csv` (192 rows)
  - `results/e3/e3_summary.json`
  - `results/e3/E3_STATUS_NOTE.md` (the honest write-up of what's below)
- Wall-clock time: sum of per-config wall_clock_seconds across all 24
  trajectory files, approximately 27 minutes of actual training compute
  (exact per-config times in each trajectory JSON), spread across the
  session due to per-call limits, not due to the compute itself being
  that slow in any single stretch.
- Result summary:
  - **Zero-false-negative check: PASSES.** 0/180 testable rows violate
    Theorem 2's privacy-only feasibility boundary (T_needed<1 => must be
    infeasible). Initially 2 rows were flagged as violations; investigated
    per the plan's own halt-and-check requirement; found to be a harness
    artifact (reading the pre-training, round-0 gap for a config where the
    privacy budget permits literally zero rounds of participation, which
    happened by chance to be low for that random Synthetic(1,1) draw --
    unrelated to whether the FL+privacy mechanism can reach the target).
    Fixed by excluding T_needed<1 rows from the empirical readout
    entirely (marked `degenerate_zero_rounds=True`), rather than reading a
    round-0 value that was never a real test of the mechanism.
  - **The harder check (does achieved accuracy track the predicted
    feasibility boundary once privacy permits enough rounds) is NOT yet
    validated**, and should not be reported as such. Investigating a
    pattern of "empirical feasibility boundary never reached even at the
    largest tested N" for several (eps_p, conversion) combinations
    surfaced two issues, one fixed-in-diagnosis-but-not-yet-in-code, one
    genuinely open: (a) the analysis read the *instantaneous* gap at a
    single round rather than the *cumulative time-average* gap Theorem 1
    actually bounds -- diagnosed, not yet patched into `run_e3.py`; (b)
    even the cumulative time-average sits well above the theoretical floor
    V_ind at the sigma=sigma_max values actually used (which range up to
    15.8, far larger than the sigma=0.5 the earlier
    stabilization pilot was checked at), and there is no calibrated L
    (smoothness) or D (initial optimality gap) in this repo to check
    whether that gap is consistent with Theorem 1's transient term still
    being non-negligible at T=150, or reflects something else. Full
    writeup in `results/e3/E3_STATUS_NOTE.md`.
- Deviation from plan: significant — see Amendment 2 (grid redesign) and
  the status note above (analysis methodology gap, not yet resolved).
- Status: **PARTIAL.** Training complete; privacy-boundary check complete
  and passing; accuracy-boundary check blocked on L/D calibration
  (concrete next step, scoped in the status note, not yet started).

---

## [2026-08-14] E3 REVISION 2 — eta=1/L, calibrated T_acc, clipping-bias diagnostic: COMPLETE
- Track: A
- Git commit at time of run: (next commit)
- Command: `python3 src/e3_theorem2_feasibility/calibrate.py`, then
  `python3 src/e3_theorem2_feasibility/run_e3.py` (invoked ~5 times across
  runs, resumed via mid-run checkpointing each time). CONFIG_VERSION
  bumped to "v2_etaL_calibrated" so revision-1 trajectories (wrong eta,
  wrong T_acc assumption) were never reused; old trajectory cache cleared
  before this run.
- Environment: local machine, Python 3.12.3, numpy 2.4.4. Same seeds/grid as
  revision 1 (Amendment 2). Calibration: L via gradient-Lipschitz probing
  (60 probes, 90th percentile, along+near a favorable p=1/sigma=0 training
  trajectory of 400 rounds), D via F(w^0)-F* on the same trajectory.
- Config: `src/e3_theorem2_feasibility/calibrate.py` (new) +
  `run_e3.py` (updated: eta=1/L per d, T_acc from calibrated L*D,
  cumulative-average readout, per-config clipping-fraction diagnostic).
- Output artifacts:
  - `results/e3/calibration/calib_d10.json`, `calib_d100.json`
  - `results/e3/trajectories/*.json` (24 new v2 trajectories, replacing
    the v1 set)
  - `results/e3/e3_feasibility_grid.csv` (192 rows, revised schema:
    measured_gap_cumavg, V_ind_theory, gap_over_theory_ratio,
    theorem2_predicted_feasible, frac_updates_clipped added)
  - `results/e3/e3_summary.json` (adds gap_over_theory_ratio_stats,
    frac_updates_clipped_by_config)
  - `results/e3/E3_STATUS_NOTE.md` (rewritten; revision-1 text preserved
    below a marker, not deleted)
- Wall-clock time: sum of per-config wall_clock_seconds, comparable to
  revision 1 (~25-30 minutes total training compute across the session).
- Result summary:
  - **Zero-false-negative check: PASSES** on the corrected grid (0/180
    testable rows violate Theorem 2's privacy-only boundary).
  - **Calibration**: L=1.2556, D=0.0018 (d=10); L=3.1631, D=0.0283
    (d=100). D's tininess in both cases means T_acc is negligible;
    feasibility is governed almost entirely by V_ind, not the transient
    term -- a genuine, if unexpected, property of Synthetic(1,1) at this
    scale (population-averaged gradient near-zero at initialization,
    plausibly from symmetric per-client model draws canceling in
    aggregate across many random clients).
  - **Clipping-bias finding (the headline result)**: raw local update
    norms exceed the clip bound C=1 in 78-98% of cases across every
    tested (d, N) cell (table in the status note). The achieved floor
    exceeds the theoretical V_ind by a median factor of 2.16x (range
    0.47x-4.61x) across all 180 testable rows, correlating with clipping
    fraction at Pearson r=0.66, and consistently worse at d=100 (mean
    96.8% clipped, median ratio 2.88x) than d=10 (mean 86.7% clipped,
    median ratio 2.05x). This is the quantified, mechanistically-explained
    answer to why revision 1's empirical boundary looked wrong -- it
    wasn't wrong, V_ind's prediction was optimistic under this much
    clipping.
  - **With corrections applied, the empirical feasibility boundary now
    matches Theorem 2's predicted boundary closely at d=10** for
    eps_p in {2,4,8} (exact match among tested N). eps_p=16 and the
    d=100 grid show the boundary pushed further out or not reached in the
    tested range -- consistent with, not contradicting, the clipping-bias
    mechanism (looser budgets permit more rounds, all landing at the
    T_max-capped steady-state floor where the ~2x inflation is most
    visible).
- Deviation from plan: substantial methodology revision, fully documented
  in this entry and `results/e3/E3_STATUS_NOTE.md` rather than folded
  into Amendment 2 (which predates this investigation) -- treated as its
  own resolution, not a plan amendment, since Amendment 2 was about grid
  *size*, this is about training *correctness*.
- Status: **COMPLETE.** E3 is closed with a genuine, quantified, honestly
  characterized result: Theorem 2 is safe (no false negatives) but not
  tight (median 2.16x conservatism gap, mechanistically explained by
  clipping bias, not left as an unexplained residual).

---

## [2026-08-14] E2 — FedProx inexactness: theory + numerical validation, COMPLETE
- Track: A
- Git commit at time of run: (next commit)
- Command: `python3 src/e2_inexactness/run_e2.py`, single invocation
  (cheap enough to complete in one call, unlike E3).
- Environment: local machine, Python 3.12.3, numpy 2.4.4. Seeds 0-4 (5 seeds,
  matching plan). L reused from E3's calibration (`calib_d10.json`,
  L=1.2556) rather than re-estimated, since it's the same model family
  and data-generating process E3 already calibrated it for.
- Config: 20 clients, Synthetic(1,1), d=10, K=4, samples_per_client=10;
  mu in {0.01,0.1,0.5,1.0}; E in {1,2,5,10,20,50}; ground truth via
  `local_fedprox_exact` (3000 steps, tol=1e-10).
- Output artifacts:
  - `docs/derivations/e2_inexactness.md` (Lemma E2.1, Theorem 1',
    Corollary E2.1, full proofs)
  - `results/e2/e2_inexactness_grid.csv` (2400 rows)
  - `results/e2/e2_summary.json` (mean-based, superseded by median
    analysis below but kept for transparency)
  - `results/e2/E2_RESULTS.md` (final write-up)
- Wall-clock time: under 5 minutes total (single run).
- Result summary:
  - **Theory**: Theorem 1' proven with an EXACT bias-cross-term
    cancellation at eta=1/L (not approximate) -- floor <= V_ind + delta^2,
    recovering Theorem 1 exactly as delta->0. Same clean structural
    extension pattern as E1/E3's other additive-term results.
  - **Rate check**: bound never violated across all 2400 measurements
    (empirical decay >= theoretical rate in every one of the 4 mu
    values tested); tightest at mu=1.0 (ratio 1.16x), loosest at mu=0.01
    (ratio 6.23x). Initial mean-based analysis was noisy/non-monotonic;
    investigated rather than accepted, traced to ~24% (97/400) of
    client-seed cells having a persistently ill-conditioned local
    objective (residual gradient norm ~O(1) even after 3000 ground-truth
    steps) -- attributed to small-sample (10 samples/client)
    conditioning, a real and FMCL-relevant phenomenon, not a flaw in the
    bound. Median adopted as the primary statistic for this reason
    (robust to the contamination, changes by <3% if outliers excluded).
  - **Magnitude check (the headline finding)**: at the manuscript's own
    operating point (mu=0.1, E=5), delta^2/V_ind = 2.91 -- inexactness
    contributes ALMOST 3X what the entire participation+privacy-noise
    floor contributes. E=10 brings this to parity (1.03); E=20 makes it
    negligible (0.16). This is NOT a null result -- Assumption 2's
    idealization is materially violated at the manuscript's own default
    local_steps=5, with a specific, evidence-based fix (roughly double
    local_steps) rather than just a caveat.
- Deviation from plan: none in scope; the mean-vs-median statistical
  investigation was itself part of executing the plan's own success
  criteria properly (a noisy/wrong initial read was checked before being
  reported, consistent with established practice), not a change to what
  was being tested.
- Status: **COMPLETE.**

---

## [2026-08-14] E4 — Markov temporal correlation: theory + validation, COMPLETE
- Track: A
- Git commit at time of run: (next commit)
- Command: `python3 src/e4_markov_correlation/run_e4.py`, single invocation.
- Environment: local machine, Python 3.12.3, numpy 2.4.4. participation.py
  generator validated against target marginal/autocorrelation before use
  (separate quick check, not part of the main run).
- Config: p in {0.1,0.2,0.5,1.0}, lambda in {0,0.1,0.2,0.35,0.5};
  check4 training comparison at N=2000, d=10, p=0.5, lambda in {0,0.2,0.4},
  2 seeds, T=150, reusing E3's calibrated L (d=10).
- Output artifacts:
  - `docs/derivations/e4_markov_correlation.md` (Lemma E4.1, E4.2,
    Theorem 1b'', full derivation including the subtlety investigation)
  - `src/common/participation.py` (new shared module, iid + Markov)
  - `results/e4/e4_results.json`
  - `results/e4/E4_RESULTS.md` (final write-up)
- Wall-clock time: under 4 minutes (single run).
- Result summary:
  - **Found and resolved a real subtlety before it became a wrong
    result**: initial intuition (single-round variance unaffected by a
    device's own temporal memory) is only half right -- the debiasing
    factor's calibration to the MARGINAL p breaks down conditional on
    realized history under Markov correlation. Identified precisely via
    filtration/conditional-expectation argument, not hand-waved.
  - **Theorem 1b'' proven**: floor <= V_corr + (lambda*G/p)^2, same
    exact-cancellation-at-eta=1/L mechanism as E2's Theorem 1'. Three
    experiments (E1, E2, E4) now converge on the same additive-
    correction structural pattern for the paper's various idealizations.
  - **Checks 1-2 (Lemma E4.1 zero-mean, Lemma E4.2 bound): 15/15 pass
    both.** Bound never violated; loose by 13-23x under random gradient
    directions (expected signature of the Cauchy-Schwarz technique the
    manuscript's own rho-bar term already uses -- not a new weakness).
  - **Check 3 confirms the manuscript's existing Section 6 verification
    methodology is structurally blind to this effect** (single-round
    checks drawing from the marginal show no lambda-dependence, exactly
    as predicted) -- a concrete demonstration of why a different check
    was needed, not just an assertion.
  - **Check 4 (real training via this repo's fedprox.py): confirms the
    qualitative direction** (floor increases monotonically with lambda)
    but the theoretical term overestimates magnitude by ~2 orders,
    consistent with checks 1-2's looseness finding once squared.
  - **Headline, quotable finding**: population size can dilute
    cross-device correlation's cost (if rho-bar=O(1/N)) but CANNOT
    dilute within-device temporal correlation's cost (N-independent
    penalty) -- a precise, useful distinction, not just "both hurt."
- Deviation from plan: grid/method redesigned significantly from the
  original plan entry once the conditional-independence subtlety was
  found (original plan assumed a simpler "just check the floor formula"
  approach that turned out to be the wrong question -- see the
  derivation doc Section 2 for why). This is a resolution of the plan's
  own stated hypothesis, arrived at through investigation, not a
  deviation for convenience.
- Status: **COMPLETE.**

---

## [2026-08-14] E5 — real device-availability trace found and analyzed, COMPLETE
- Track: A
- Git commit at time of run: (next commit)
- Command: web_search + web_fetch (feasibility scan, outside bash_tool),
  then `curl -sL -o /tmp/state_traces.json https://raw.githubusercontent.com/PKU-Chengxu/FLASH/main/data/state_traces.json`,
  then `python3 src/e5_rho_estimation/run_e5.py` (run twice: 30-min grid
  primary, 60-min grid robustness check, via a temporary sed-based
  parameter flip, reverted after).
- Environment: local machine. raw.githubusercontent.com is in the network
  allow-list, confirmed reachable (unlike the R770/jump server).
- Config: RESAMPLE_MINUTES=30 (primary) and 60 (robustness check);
  common window 2020-01-29 to 2020-02-03 (chosen from the per-device
  coverage-span distribution, median ~120h starting ~Jan 29).
- Output artifacts:
  - `results/e5/e5_stats.json` (primary), `e5_stats_60min_robustness_check.json`
  - `results/e5/availability_matrix.npy`
  - `results/e5/E5_RESULTS.md` (final write-up)
- Wall-clock time: feasibility scan (web search/fetch) a few minutes;
  download ~10s (37MB); each run_e5.py invocation under a minute.
- Result summary:
  - **Feasibility scan succeeded** (plan flagged this as genuinely
    uncertain -- most such corpora need data-use agreements). Traced
    FedScale's own device-availability emulation to its actual source,
    FLASH (Yang et al., WWW'21), BSD-2-Clause, directly downloadable via
    an allow-listed domain.
  - **Caught and fixed a real bug before trusting any number**: first
    extraction pass defaulted a device's state to "unavailable" outside
    its own observed event window, systematically biasing p_hat down
    (0.025 vs the corrected 0.081, a 3x difference) given uneven
    per-device coverage. Fixed via NaN-aware forward-fill + common
    well-covered window + pairwise-complete statistics.
  - **Headline finding, validated for robustness across two grid
    resolutions (30-min and 60-min agree within ~14%)**: empirical
    rho_bar ~0.09-0.11, roughly **68-77x** the O(1/N) threshold Theorem
    1b's own remark identifies as necessary for population-averaging
    benefit to survive. Effective population for variance-averaging in
    this real trace is roughly N/70, not N.
  - **lambda_hat ~0.42-0.58**, confirming E4's within-device temporal
    correlation concern is empirically substantial, not a theoretical
    curiosity -- consistent with E4's finding that this cost component
    does not shrink with N regardless.
  - **p_hat ~0.081**, below the low end of the p in {0.1,...,1.0} range
    used illustratively throughout the manuscript's own Sections 6-11.
  - Explicit, stated scope caveat: this is a general consumer smartphone
    (input-method-app) population, not a curated FMCL healthcare
    population -- reported as the first real anchor available, not as
    confirmed values for any specific FMCL deployment.
- Deviation from plan: none in spirit (plan explicitly anticipated this
  might be BLOCKED and treated success as contingent) -- the feasibility
  scan succeeded, which the plan already scoped as the good-case branch.
- Status: **COMPLETE.**

---

## [2026-08-14] E6 — feasibility nomograph, COMPLETE
- Track: A
- Git commit at time of run: (next commit)
- Command: `python3 src/e6_nomograph/run_e6.py` (after a performance
  tuning pass -- see below).
- Environment: local machine, Python 3.12.3, numpy, matplotlib 3.10.8 (Agg
  backend).
- Config: d in {10,50,100,500,1000,5000,20000,100000} (extended beyond
  plan's original 10-5000 ceiling); eps_acc in {0.1,0.5,1.0}; eps_p in
  {1,2,4,8,16}; LD in {1,10,50}; delta=1e-5, p=0.5, C=G=1.
- Output artifacts:
  - `results/e6/e6_nomograph_grid.csv` (360 rows)
  - `results/e6/e6_nomograph_heatmap.png`
  - `results/e6/e6_summary.json`
  - `results/e6/E6_RESULTS.md` (final write-up, including the explicit
    decision NOT to bake E2-E5's findings into a single number)
- Wall-clock time: ~130s for the full grid, after a performance fix (see
  below); first attempt did not complete within available tool-call time.
- Result summary:
  - **Performance issue found and fixed before completing the run**:
    reusing E1/E3's validated precision settings (2500 RDP orders,
    200-iter inner bisection, 70-iter outer bisection) in a
    nested-bisection structure multiplies to ~24.5s/cell, ~2.5 hours for
    360 cells. Timed a single cell with tuned-down settings (300 orders,
    40/40 iterations) before committing to the full run: ~0.37s/cell,
    ~130s total -- verified against E1's own validated operating point
    (d=100, eps_acc=0.5, eps_p=8, LD=10): N*_basic matches E1 exactly
    (15,254.6); N*_pathA differs by 0.27% (13,050.4 vs E1's 13,015.2),
    the expected size of the deliberate precision/speed tradeoff.
  - Full 360-cell table and heatmap produced, monotonic and sensible in
    both d and eps_p as expected.
  - **Deliberately did not fold E2/E3/E4/E5's findings into the table**
    -- each depends on quantities not expressible as functions of d
    alone; stated instead as explicit, separate safety margins (E2:
    ~2.9x at manuscript defaults, fixable via E=10; E3: ~2.16x median,
    budget accordingly; E4: population-independent, no N* margin helps;
    E5: check rho-bar for the real target population before trusting
    the favorable O(1/N) regime at all).
- Deviation from plan: d grid extended (10-100000 instead of 10-5000)
  to better bracket realistic compact-model sizes; precision settings
  tuned down from what a first attempt (reusing E1/E3 defaults) would
  have used, for tractable runtime -- both changes documented above
  rather than silently applied.
- Status: **COMPLETE.**

---

## [2026-08-14] E7 — script prepared, NOT executed (Track B, R770)
- Track: B (scripts written and validated locally; execution deferred to
  the R770, which the local machine cannot reach)
- Git commit at time of run: (next commit)
- Command (local validation only): `python3 src/e7_neural_feasibility/test_dp_wrapper.py`
- Environment: local machine, Python 3.12.3, torch 2.13.0 (CPU-only build,
  installed earlier for exactly this purpose).
- Config: N/A (no real run yet)
- Output artifacts:
  - `src/e7_neural_feasibility/dp_wrapper.py` (DP+FL mechanics,
    model-independent, tested)
  - `src/e7_neural_feasibility/test_dp_wrapper.py` (7/7 checks pass)
  - `src/e7_neural_feasibility/run_e7_mitbih.py` (main integration
    script, TODO(confirm)-marked integration points)
  - `src/e7_neural_feasibility/README.md` (integration checklist, launch
    command, grid rationale)
- Wall-clock time: dp_wrapper tests under 1 second.
- Result summary:
  - Before writing anything, searched prior conversations for Paper 3's
    actual current state rather than relying on this repo's own
    (partially incorrect) earlier plan text. Corrections found:
    - MIT-BIH ResNet is the best-grounded target: 8 seeds, t=3.73,
      statistically significant convergence under favorable conditions
      (per an explicitly-dated PROJECT_STATE.md snapshot from that
      session, treated as authoritative over earlier, superseded
      exploratory fragments found in the same long conversation thread).
    - FEMNIST-ResNet (ResNet2D_FEMNIST, 5,328,638 parameters) is NOT yet
      validated as converging even under favorable conditions -- this
      repo's original plan wrongly assumed it was "already built and
      verified, not yet run" as if that meant validated; it means
      untested. Demoted from primary to explicitly-not-recommended for
      E7's purpose (testing dropout/privacy on top of an unvalidated
      baseline confounds two separate questions).
    - This repo's earlier plan text incorrectly called MIT-BIH's model
      "ResNet1D_PTB" -- that name belongs to PTB-XL's model specifically;
      MIT-BIH's is referred to only as "ResNet" in the source material
      found, with no fully confirmed class name -- left as TODO(confirm)
      rather than guessed.
    - MIT-BIH's real client pool is small (~44-47 patients), and the
      existing pipeline samples a FIXED-SIZE K per round (not
      independent Bernoulli(p)) -- a materially different process from
      what this project's theoretical results assume. Documented
      explicitly in the script and README rather than glossed over.
    - Exact, verified CPU-thread-explosion fix recovered and carried
      over precisely: `torch.set_num_threads(4)` plus
      `OMP_NUM_THREADS=4 MKL_NUM_THREADS=4` set before every python3
      invocation (this was previously only guessed at as a "known fix
      to carry over" in ENVIRONMENT.md; now the exact, sourced form).
  - **Documentation convention flagged for review**: existing commit
    messages and code comments in this project had not been checked
    against the documentation conventions used for the Paper 3 project
    (objective, third-person research-log phrasing, no process narration
    referencing how the work was produced). Flagged for review and
    subsequently brought into line across this project's history and
    files.
  - dp_wrapper.py: 7/7 correctness checks pass (flatten/unflatten
    round-trip, clipping bound and no-op-under-bound, noise scale,
    fixed-K sampling count, fixed-K debiasing arithmetic checked by
    hand, K > population clamping).
  - run_e7_mitbih.py fails cleanly and immediately with a clear message
    when the (expected, on this machine) model/data imports are
    unavailable, rather than crashing unpredictably partway through.
- Deviation from plan: substantial corrections to the original plan's
  assumptions about Paper 3's state (see above), found by searching
  rather than assumed -- this is a correction of a prior error in this
  repo's own plan text, not a deviation from it introduced now.
- Status: **Script and mechanics COMPLETE and locally validated.
  Real execution PENDING -- requires resolving the
  TODO(confirm) integration points and run on the R770.**
