# E6 results — feasibility nomograph

**What this is and isn't.** This is a lookup table and heatmap for
Theorem 2's population threshold N*(d, eps_acc, eps_p; LD), computed
exactly (well, to nomograph precision — see below) under two conditions:
the manuscript's own basic (Mironov) conversion, and E1's Path A
(Canonne-Kamath-Steinke) tightening. **It deliberately does NOT bake
E2's inexactness term, E3's clipping-bias factor, E4's temporal-
correlation term, or E5's empirical rho-bar into a single "corrected"
N*.** Each of those depends on quantities that are not functions of d
alone (L_i, per-model clip-vs-update-norm ratios, lambda, rho-bar), so
folding them into this table would manufacture false precision — a
single number that looks authoritative but silently assumes specific
values for things that vary by deployment. Instead, Section 3 below
states them as separate, explicit safety margins to apply on top.

## 1. Grid and method

d in {10, 50, 100, 500, 1000, 5000, 20000, 100000} (extended beyond the
original plan's 10-5000 ceiling to bracket realistic compact-model
parameter counts — a real small CNN or ResNet1D head is typically tens
of thousands to low hundreds of thousands of parameters, not a few
thousand); eps_acc in {0.1, 0.5, 1.0}; eps_p in {1,2,4,8,16}; LD in
{1,10,50}; delta=1e-5, p=0.5, C=G=1 throughout (per the resolved Q1).
360 grid cells, each solved by nested bisection (outer: population size
N; inner, for Path A: RDP order optimization + privacy-round inversion).

**A real performance problem, found and fixed before running the full
grid**: the first version used E1/E3's validated precision settings
(2500 RDP orders, 200-iteration inner bisection, 70-iteration outer
bisection) — appropriate for a correctness-critical result, but the
nested-bisection structure multiplies these into ~24.5 seconds per grid
cell, ~2.5 hours for the full 360-cell grid. Tuned down to
nomograph-appropriate precision (300 orders, 40-iteration inner and
outer bisections — still ~2^-40 relative precision, far beyond what a
lookup table needs) after timing a single cell first rather than
guessing. Full grid now runs in ~130 seconds. Verified the tuned-down
settings didn't sacrifice accuracy where it matters: the grid cell
matching E1's own validated operating point (d=100, eps_acc=0.5,
eps_p=8, LD=10) gives N*_basic=15,254.6 (exact match to E1's number) and
N*_pathA=13,050.4 (E1 had 13,015.2 — 0.27% difference, the expected size
of the deliberate precision/speed tradeoff, negligible for this purpose).

## 2. Artifacts

- `results/e6/e6_nomograph_grid.csv` — full 360-row table
- `results/e6/e6_nomograph_heatmap.png` — log10(N*) heatmap at
  eps_acc=0.5, LD=10 (the manuscript's own §7.3 operating point), Path-A
  values, across the full (d, eps_p) grid
- `results/e6/e6_summary.json` — grid parameters for reproducibility

Every number in the CSV is reproducible by rerunning
`src/e6_nomograph/run_e6.py` and is traceable to a specific row —
any N* quoted in the manuscript from this table should cite the exact
(d, eps_acc, eps_p, LD) row it came from.

## 3. Safety margins to apply on top (not baked into the table)

Read alongside the table, not instead of it:

| Source | Finding | Suggested margin |
|---|---|---|
| E2 (FedProx inexactness) | At the manuscript's own default (mu=0.1, E=5 local steps), inexactness contributes ~2.9x V_ind | Use E=10 local steps (brings parity, ratio 1.03) rather than applying a margin, if compute allows — a fix, not just a caveat |
| E3 (clipping bias) | Achieved floor exceeds V_ind by a median factor of 2.16x under near-universal clipping at C=1 (78-98% of updates clipped throughout the tested grid) | Budget ~2x the table's N* for real confidence |
| E4 (temporal correlation) | Additional floor term (lambda*G/p)^2, does NOT shrink with N (unlike rho-bar's term) | Population-independent; no margin on N* helps — mitigate lambda directly (e.g. scheduling diversity) instead |
| E5 (empirical rho-bar) | Real device populations can show rho-bar ~70x the O(1/N) threshold | Check rho-bar for the actual target population before trusting the O(1/N)-regime assumption behind Theorem 1b's favorable case at all |

None of these compound simply (they're not independent multipliers to
multiply together blindly), but the practical takeaway for a deployment
planner is: **treat this table's N* as a floor, not a target** — E2's
and E3's findings alone suggest budgeting roughly 2-3x the tabulated
value is prudent until model/deployment-specific corrections are
available.

## 4. Worked examples for the healthcare series

For the compact ECG/arrhythmia-style models named elsewhere in the FMCL
series (d roughly in the tens of thousands), at eps_acc=0.5, LD=10:

| d | eps_p=2 | eps_p=8 |
|---|---|---|
| 20,000 | N* (Path A) ~ 10^7.5 | N* (Path A) ~ 10^6.4 |
| 100,000 | N* (Path A) ~ 10^8.2 | N* (Path A) ~ 10^7.1 |

(exact values in the CSV) — these are large populations, consistent
with the manuscript's own framing that high-dimensional models under
tight targets need substantial device populations; this table now lets
that claim be checked at the specific dimension a given healthcare model
actually has, rather than only at the illustrative d=100 point.
