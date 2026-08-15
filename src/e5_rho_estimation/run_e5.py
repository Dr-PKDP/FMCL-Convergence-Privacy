"""
E5 -- empirical rho-bar (and, as a bonus tying directly to E4, lambda)
from a real device-availability trace.

Data source: FLASH (Yang et al., "Characterizing Impacts of Heterogeneity
in Federated Learning upon Large-Scale Smartphone Data," WWW'21),
https://github.com/PKU-Chengxu/FLASH, data/state_traces.json.
BSD-2-Clause licensed. 1000 devices, ~1 week (2020-01-28 to 2020-02-04),
sampled from a 136k-device population of a Chinese input-method-app's
users. De-identified (random device ids; no age/gender/location kept
per the FedScale README that also redistributes this data).

This is the FIRST real measurement, as far as this repo is aware, of
rho-bar (Definition 3) and of the within-device lambda E4 introduces,
directly answering the manuscript's own stated gap (S13.3): "The value
of rho-bar for a real FMCL device population has never been measured."

Availability definition, matching Definition 1 exactly: idle (screen
off) AND charging (battery_charged_on) AND connected via Wi-Fi.

Run: python3 src/e5_rho_estimation/run_e5.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

OUT_DIR = ROOT / "results" / "e5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRACE_PATH = Path("/tmp/state_traces.json")  # downloaded via curl, see EXPERIMENT_LOG.md
RESAMPLE_MINUTES = 30  # grid resolution -- see run script docstring for justification
# Common window chosen from the per-device coverage distribution (median
# span ~120h starting ~Jan 29; see EXPERIMENT_LOG.md for the check that
# motivated this): tight enough that most devices have real data
# throughout, avoiding the default=False-outside-a-device's-own-window
# bias an earlier version of this script had (documented in the log).
WINDOW_START = "2020-01-29 00:00:00"
WINDOW_END = "2020-02-03 00:00:00"


def parse_device_events(messages: str):
    """Parse the raw newline-separated event log into typed events."""
    idle_events = []     # (timestamp, is_idle)  -- screen_off=idle, screen_on=not idle
    charge_events = []   # (timestamp, is_charging)
    wifi_events = []     # (timestamp, is_wifi)
    for line in messages.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        ts_str, ev = parts[0].strip(), parts[1].strip()
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ev == "screen_off":
            idle_events.append((ts, True))
        elif ev == "screen_on":
            idle_events.append((ts, False))
        elif ev == "battery_charged_on":
            charge_events.append((ts, True))
        elif ev == "battery_charged_off":
            charge_events.append((ts, False))
        elif ev == "wifi":
            wifi_events.append((ts, True))
        elif ev in ("2G", "Unknown"):
            wifi_events.append((ts, False))
        # BATTERY_PCT (numeric %) events ignored -- not needed for the
        # idle/charging/wifi availability definition used here.
    return idle_events, charge_events, wifi_events


def events_to_series_nan_aware(events, grid: pd.DatetimeIndex) -> np.ndarray:
    """Forward-fill onto the grid, but leave NaN OUTSIDE this device's own
    observed [first_event, last_event] window rather than assuming a
    default state -- a device with no data in that stretch is UNKNOWN,
    not necessarily unavailable, and treating unknown-as-unavailable was
    a real bug in an earlier version of this script (see
    EXPERIMENT_LOG.md), producing a systematically downward-biased p_hat."""
    if not events:
        return np.full(len(grid), np.nan)
    events = sorted(events, key=lambda x: x[0])
    ts = pd.DatetimeIndex([e[0] for e in events])
    vals = np.array([e[1] for e in events], dtype=float)
    s = pd.Series(vals, index=ts)
    s = s[~s.index.duplicated(keep="last")]
    reindexed = s.reindex(s.index.union(grid)).ffill().reindex(grid)
    out = reindexed.values.astype(float)
    first_t, last_t = ts.min(), ts.max()
    out[(grid < first_t) | (grid > last_t)] = np.nan
    return out


def build_availability_matrix(trace: dict, grid: pd.DatetimeIndex) -> tuple[np.ndarray, list]:
    """Returns (T, N) float availability matrix (1.0/0.0/nan) and device ids kept."""
    device_ids = sorted(trace.keys(), key=lambda x: int(x))
    cols = []
    kept_ids = []
    for did in device_ids:
        idle_ev, charge_ev, wifi_ev = parse_device_events(trace[did]["messages"])
        if not idle_ev or not charge_ev or not wifi_ev:
            continue
        idle = events_to_series_nan_aware(idle_ev, grid)
        charging = events_to_series_nan_aware(charge_ev, grid)
        wifi = events_to_series_nan_aware(wifi_ev, grid)
        # AND of three states, NaN-propagating: available=1 requires all
        # three known AND true; unavailable=0 if any known-false; NaN if
        # any of the three is unknown at that instant AND none is known-false.
        with np.errstate(invalid="ignore"):
            any_false = (idle == 0) | (charging == 0) | (wifi == 0)
            all_true = (idle == 1) & (charging == 1) & (wifi == 1)
        col = np.where(all_true, 1.0, np.where(any_false, 0.0, np.nan))
        # require at least 50% grid coverage (non-NaN) to keep the device
        if np.isnan(col).mean() > 0.5:
            continue
        cols.append(col)
        kept_ids.append(did)
    A = np.array(cols).T  # (T, N)
    return A, kept_ids


def compute_stats(A: np.ndarray) -> dict:
    """A may contain NaN (unknown state) -- all statistics below use
    only pairwise/columnwise-valid (non-NaN) observations, matching how
    irregular panel data is standardly handled (e.g. pandas' default
    pairwise-complete correlation)."""
    T, N = A.shape

    valid_frac_per_device = 1.0 - np.isnan(A).mean(axis=0)
    p_hat = float(np.nanmean(A))  # mean over all valid (i,t) entries

    # rho-bar: pairwise |correlation| across devices, NaN-aware (pandas
    # .corr() with min_periods handles this correctly out of the box).
    df = pd.DataFrame(A)
    corr = df.corr(min_periods=int(0.3 * T)).values  # need >=30% overlap to trust a pair
    n_cols = corr.shape[0]
    off_diag_mask = ~np.eye(n_cols, dtype=bool)
    pairwise_vals = corr[off_diag_mask]
    pairwise_vals = pairwise_vals[~np.isnan(pairwise_vals)]
    rho_bar_hat = float(np.abs(pairwise_vals).mean())
    n_pairs_used = len(pairwise_vals)

    # lambda: lag-1 autocorrelation per device (within-device across
    # time), NaN-aware, averaged across devices with enough valid data.
    lambdas = []
    for j in range(N):
        col = A[:, j]
        x0, x1 = col[:-1], col[1:]
        mask = ~(np.isnan(x0) | np.isnan(x1))
        if mask.sum() < 30:
            continue
        x0v, x1v = x0[mask] - x0[mask].mean(), x1[mask] - x1[mask].mean()
        den = np.sqrt((x0v ** 2).sum() * (x1v ** 2).sum())
        if den > 1e-9:
            lambdas.append((x0v * x1v).sum() / den)
    lambda_hat = float(np.mean(lambdas)) if lambdas else None

    return dict(
        T=T, N=N, mean_valid_frac_per_device=float(valid_frac_per_device.mean()),
        p_hat=p_hat, rho_bar_hat=rho_bar_hat, n_pairs_used_for_rho_bar=n_pairs_used,
        lambda_hat=lambda_hat, n_devices_used_for_lambda=len(lambdas),
        one_over_N=1.0 / N,
        rho_bar_over_one_over_N=rho_bar_hat / (1.0 / N),
        inflation_factor_1_plus_N_minus_1_rho=1 + (N - 1) * rho_bar_hat,
    )


def main():
    if not TRACE_PATH.exists():
        print(f"Trace not found at {TRACE_PATH} -- download first (see EXPERIMENT_LOG.md).")
        sys.exit(1)

    with open(TRACE_PATH) as f:
        trace = json.load(f)
    print(f"Loaded {len(trace)} devices from FLASH state_traces.json")

    grid = pd.date_range(WINDOW_START, WINDOW_END, freq=f"{RESAMPLE_MINUTES}min")
    print(f"Common window: {WINDOW_START} to {WINDOW_END} "
          f"({len(grid)} points at {RESAMPLE_MINUTES}-minute resolution)")

    A, kept_ids = build_availability_matrix(trace, grid)
    print(f"Availability matrix: {A.shape[0]} time points x {A.shape[1]} devices "
          f"({len(trace) - A.shape[1]} devices dropped: missing signal type(s) or <50% grid coverage)")

    stats = compute_stats(A)
    print(json.dumps(stats, indent=2))

    with open(OUT_DIR / "e5_stats.json", "w") as f:
        json.dump(dict(
            source="FLASH (Yang et al., WWW'21), github.com/PKU-Chengxu/FLASH, "
                   "data/state_traces.json, BSD-2-Clause",
            resample_minutes=RESAMPLE_MINUTES,
            window=[WINDOW_START, WINDOW_END],
            **stats,
        ), f, indent=2)
    np.save(OUT_DIR / "availability_matrix.npy", A)
    print(f"\nWritten -> {OUT_DIR / 'e5_stats.json'}, availability_matrix.npy")


if __name__ == "__main__":
    main()
