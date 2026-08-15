"""
E6 -- feasibility nomograph: N*(d, eps_acc, eps_p; LD) across a grid
relevant to the compact models named across the FMCL series, computed
both under the manuscript's own basic (Mironov) conversion and under
E1's Path A (Canonne-Kamath-Steinke) tightening.

Design decision (see results/e6/E6_RESULTS.md for the full reasoning):
this table does NOT attempt to bake E2's inexactness term, E3's
clipping-bias factor, or E4's temporal-correlation term into a single
"corrected" N*, because each of those depends on model/deployment-
specific quantities (L_i, G_i, clip-vs-update-norm ratio, lambda) that
aren't functions of d alone -- doing so would manufacture false
precision. Instead, this script computes the two closed-form numbers
that ARE exact functions of (d, eps_acc, eps_p, LD) -- baseline and
Path-A-tightened N* -- and the accompanying write-up states the
empirically-found conservatism factors (E2, E3, E4, E5) as multiplicative
safety margins to apply on top, honestly kept separate rather than
conflated with the exact closed-form numbers.

Run: python3 src/e6_nomograph/run_e6.py
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import privacy as P  # noqa: E402

OUT_DIR = ROOT / "results" / "e6"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Grid: d extended beyond the original plan's 10-5000 up to 100000 to
# bracket realistic compact healthcare-model parameter counts (a small
# CNN/ResNet1D head is often tens of thousands to low hundreds of
# thousands of parameters, not just a few thousand) -- d here is the
# TOTAL model dimension the DP noise term is summed over (Eq. 17's d),
# i.e. literally the parameter count.
D_GRID = [10, 50, 100, 500, 1000, 5000, 20000, 100000]
EPS_ACC_GRID = [0.1, 0.5, 1.0]
EPS_P_GRID = [1, 2, 4, 8, 16]
LD_GRID = [1, 10, 50]
DELTA = 1e-5
P_AVAIL = 0.5
G = C = 1.0  # per resolved Q1


def sigma_max_sq(N, d, eps_acc, p=P_AVAIL, G_=G, C_=C):
    return max(0.0, (N * p * eps_acc / 2 - (1 - p) * G_ ** 2) / (d * C_ ** 2))


def feasible_basic(N, d, eps_acc, eps_p, LD):
    T_acc = 4 * LD / eps_acc
    s2max = sigma_max_sq(N, d, eps_acc)
    if s2max <= 0:
        return False
    sigma = math.sqrt(s2max)
    T_priv = P.T_priv_manuscript(sigma, eps_p, DELTA)
    return P_AVAIL * T_acc <= T_priv


def feasible_pathA(N, d, eps_acc, eps_p, LD, orders):
    T_acc = 4 * LD / eps_acc
    s2max = sigma_max_sq(N, d, eps_acc)
    if s2max <= 0:
        return False
    sigma = math.sqrt(s2max)
    T_priv = P.invert_T_priv(
        eps_p, sigma, DELTA,
        lambda T, s, dl: P.eps_from_orders(T, s, dl, orders, convert=P.convert_tight),
        T_lo=1.0, T_hi=1e12, max_iter=40,  # nomograph precision, not proof-grade --
                                             # see E6_RESULTS.md; the 70-outer x 200-inner
                                             # x 2500-order default settings used in E1/E3
                                             # (which need tight tolerances for correctness
                                             # gates) would take ~2.5 hours over this grid;
                                             # 40 bisection iterations already gives 2^-40
                                             # relative precision, far beyond what a lookup
                                             # table needs.
    )
    return P_AVAIL * T_acc <= T_priv


def bisect_N(feasible_fn, N_lo=1.0, N_hi=1e10, iters=40):
    if not feasible_fn(N_hi):
        return None
    if feasible_fn(N_lo):
        return N_lo
    lo, hi = N_lo, N_hi
    for _ in range(iters):
        mid = math.sqrt(lo * hi)
        if feasible_fn(mid):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    orders = P.default_order_grid(1.001, 2048.0, 300)
    rows = []
    for d in D_GRID:
        for eps_acc in EPS_ACC_GRID:
            for eps_p in EPS_P_GRID:
                for LD in LD_GRID:
                    n_basic = bisect_N(lambda N: feasible_basic(N, d, eps_acc, eps_p, LD))
                    n_pathA = bisect_N(lambda N: feasible_pathA(N, d, eps_acc, eps_p, LD, orders))
                    rows.append(dict(
                        d=d, eps_acc=eps_acc, eps_p=eps_p, LD=LD,
                        N_star_basic=n_basic, N_star_pathA=n_pathA,
                        tightening_ratio=(n_basic / n_pathA) if (n_basic and n_pathA) else None,
                    ))
        print(f"d={d} done")

    out_csv = OUT_DIR / "e6_nomograph_grid.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out_csv}")

    # Figure: heatmap of log10(N*_pathA) over (d, eps_p), at eps_acc=0.5, LD=10
    # (the manuscript's own §7.3 operating point), one panel -- kept to a
    # single, clear figure rather than a large grid of subplots.
    eps_acc_fixed, LD_fixed = 0.5, 10
    sub = [r for r in rows if r["eps_acc"] == eps_acc_fixed and r["LD"] == LD_fixed]
    d_vals = sorted(set(r["d"] for r in sub))
    epsp_vals = sorted(set(r["eps_p"] for r in sub))
    Z = np.full((len(epsp_vals), len(d_vals)), np.nan)
    for r in sub:
        i = epsp_vals.index(r["eps_p"])
        j = d_vals.index(r["d"])
        if r["N_star_pathA"]:
            Z[i, j] = np.log10(r["N_star_pathA"])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(Z, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(d_vals)))
    ax.set_xticklabels([str(d) for d in d_vals], rotation=45)
    ax.set_yticks(range(len(epsp_vals)))
    ax.set_yticklabels([str(e) for e in epsp_vals])
    ax.set_xlabel("model dimension d (parameter count)")
    ax.set_ylabel("privacy budget eps_p")
    ax.set_title(f"log10(N*) feasibility threshold, Path-A-tightened\n"
                 f"(eps_acc={eps_acc_fixed}, LD={LD_fixed}, delta={DELTA}, p={P_AVAIL})")
    for i in range(len(epsp_vals)):
        for j in range(len(d_vals)):
            if not np.isnan(Z[i, j]):
                ax.text(j, i, f"{Z[i,j]:.1f}", ha="center", va="center",
                         color="white" if Z[i, j] > np.nanmedian(Z) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="log10(N*)")
    fig.tight_layout()
    fig_path = OUT_DIR / "e6_nomograph_heatmap.png"
    fig.savefig(fig_path, dpi=150)
    print(f"Figure -> {fig_path}")

    with open(OUT_DIR / "e6_summary.json", "w") as f:
        json.dump(dict(n_rows=len(rows), d_grid=D_GRID, eps_acc_grid=EPS_ACC_GRID,
                        eps_p_grid=EPS_P_GRID, LD_grid=LD_GRID, delta=DELTA, p=P_AVAIL), f, indent=2)


if __name__ == "__main__":
    main()
