# E5 results — the first real measurement of ρ̄ for an FMCL-relevant device population

**Bottom line first:** the manuscript's own §13.3 states plainly, "The
value of rho-bar for a real FMCL device population has never been
measured." It now has been, from a real, permissively-licensed,
136k-device-derived trace. **The empirical rho-bar is roughly 70x the
O(1/N) threshold** Theorem 1b's own remark identifies as the boundary
for population-averaging benefit to survive — meaning, in this real
population, it does not survive: the effective population for variance
-averaging purposes is roughly N/70, not N. This is a sobering, concrete,
citable finding, not a null result to bury.

## 1. Data source and why it's usable

**FLASH** (Yang et al., "Characterizing Impacts of Heterogeneity in
Federated Learning upon Large-Scale Smartphone Data," WWW'21),
github.com/PKU-Chengxu/FLASH, BSD-2-Clause licensed. Ships
`data/state_traces.json`: 1000 devices sampled from a 136k-device
population of a Chinese input-method-app's users, ~1 week of raw
timestamped events (screen on/off, screen lock/unlock, charging on/off,
Wi-Fi/cellular connectivity, battery percentage), de-identified
(random device ids; no age/gender/location retained, per the FedScale
project's own redistribution of this same data). Traced to this specific
file via FedScale's own documentation, which uses this exact dataset for
its own client-availability emulation — this is not an obscure or
unvetted source; it is already used by an established FL benchmarking
platform for precisely this purpose.

**Important scope caveat, stated plainly rather than glossed over**:
this is a general consumer smartphone population (input-method-app
users), not a curated FMCL healthcare deployment population specifically.
It is the best available real anchor, not a claim that FMCL healthcare
populations behave identically — the manuscript's own text should treat
this as "the first real evidence available," not "the confirmed value
for FMCL."

## 2. Methodology (and a real bug caught before it reached a number)

Availability reconstructed exactly per Definition 1: idle (screen off)
AND charging (battery_charged_on) AND Wi-Fi-connected, forward-filled
from the raw event log onto a regular 30-minute grid.

**A real bug was found and fixed before trusting any result**: the first
version of this extraction defaulted a device's state to "unavailable"
outside its own observed event window (i.e., treated "no data yet" as
"definitely not idle/charging/wifi"). Devices have uneven coverage
windows (median ~120 hours, some as short as 24, within an overall
13-day span), so this systematically pulled p_hat downward for any
device with partial coverage of the analysis window. Fixed by treating
out-of-window periods as missing (NaN) rather than false, restricting
to a common well-covered window (Jan 29 - Feb 3, 2020), and requiring
>=50% grid coverage per device to be included, with all downstream
statistics (mean, correlation, autocorrelation) computed pairwise/
columnwise over valid observations only (matching standard practice for
irregular panel data, e.g. pandas' own pairwise-complete correlation).
This changed p_hat from 0.025 to 0.081 (roughly 3x) -- not a rounding
difference, a real correction.

## 3. Results

| Quantity | Value (30-min grid, primary) | Value (60-min grid, robustness check) |
|---|---|---|
| Devices used | 723 (of 1000; 277 dropped for missing signal types or <50% coverage) | 723 |
| p_hat (empirical availability rate) | 0.0812 | 0.0810 |
| rho_bar_hat | 0.0942 | 0.1071 |
| 1/N | 0.00138 | 0.00138 |
| rho_bar_hat / (1/N) | **68.1x** | **77.4x** |
| Inflation factor 1+(N-1)*rho_bar | **69.0** | **78.3** |
| lambda_hat (E4's within-device lag-1 autocorrelation) | 0.577 | 0.423 |

The two grid resolutions agree closely on p_hat and rho_bar (within
~14%), giving confidence these are not artifacts of an arbitrary
resampling choice. lambda_hat is meaningfully different between the two
grids (0.577 vs 0.423) -- **expected, not a discrepancy**: lag-1
autocorrelation is inherently tied to the time unit ("lag 1" means
different things at 30 vs 60 minutes), and this dependency is itself
informative: it means lambda should be reported and used together with
the assumed FL round length, not as a grid-independent constant the way
rho_bar approximately is.

## 4. What this means for the manuscript

- **p_hat = 0.081 is well below the p in {0.1, 0.2, 0.5, 1.0} range
  used throughout Sections 6-11's illustrations.** Real-world
  simultaneous idle+charging+Wi-Fi availability, at 30-minute
  granularity, is rarer than the paper's own illustrative parameter
  choices assume. Worth a sentence noting the low end of the paper's
  own p range is the empirically realistic one, not the high end.
- **rho_bar_hat ~0.09-0.11 is dramatically above the O(1/N) regime
  Theorem 1b's own remark identifies as necessary for population
  averaging to help.** At N=723 devices, the effective population for
  variance-averaging purposes is roughly 723/69 ~ 10 devices, not 723.
  This is the single most consequential empirical finding across E1-E5:
  it suggests that, absent active mitigation (the "reduce rho-bar"
  mitigations the manuscript's own Remark after Theorem 1b already
  gestures at, citing [69][70]), simply recruiting more devices may buy
  far less than Theorem 1b's favorable O(1/N) regime would suggest, for
  a population with synchronized daily routines resembling this trace.
- **lambda_hat ~0.4-0.6 confirms E4's concern is not a theoretical
  curiosity** — real device populations show substantial within-device
  temporal persistence, and per E4's Theorem 1b'', this specific cost
  component does not shrink with population size at all.
- **Together, E4 and E5 tell a coherent, sobering, and useful story**:
  both correlation channels are real and substantial in actual device
  behavior, cross-device correlation's cost is diluted far less than
  hoped by the population sizes FMCL targets, and temporal correlation's
  cost cannot be diluted by scale under any circumstances. This belongs
  in the manuscript's Section 7/13 discussion as a concrete empirical
  grounding for what was previously a purely theoretical concern.

## 5. Artifacts

- `src/e5_rho_estimation/run_e5.py` — extraction and analysis pipeline
- `results/e5/e5_stats.json` — primary (30-min) results
- `results/e5/e5_stats_60min_robustness_check.json` — robustness check
- `results/e5/availability_matrix.npy` — the extracted (T,N) availability
  matrix itself, for any further analysis
- Raw trace: downloaded from
  `https://raw.githubusercontent.com/PKU-Chengxu/FLASH/main/data/state_traces.json`
  (not committed to this repo — 37MB; re-downloadable via the URL above,
  reproducibility relies on the upstream repo remaining available, same
  as any dataset citation)

**Citation obligation** (BSD-2-Clause + explicit request in the FLASH
README): Yang, C., Wang, Q., Xu, M., Chen, Z., Bian, K., Liu, Y., Liu, X.
"Characterizing impacts of heterogeneity in federated learning upon
large-scale smartphone data." WWW 2021. This should be added to the
manuscript's reference list if this finding is incorporated.
