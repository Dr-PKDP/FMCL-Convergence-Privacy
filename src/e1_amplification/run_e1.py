"""
E1 — numerical companion to docs/derivations/e1_amplification.md.

Produces two CSVs:
  results/e1/path_a_tightening.csv
      Valid, unconditional tightening: manuscript's basic (Mironov 2017)
      RDP->DP conversion vs. the tight (Canonne-Kamath-Steinke 2020)
      conversion, same composition (no subsampling), across a grid of
      (T_i, sigma, delta). This IS proposed for adoption in Theorem 2'.

  results/e1/path_b_reference_only.csv
      NOT a valid privacy claim (see derivation doc, Section 3-4). Computed
      only to quantify, for the paper's discussion, how large an error a
      deployment would make if it applied device-level subsampling
      amplification against the aggregator despite the aggregator observing
      participation directly. Every row is labeled invalid=True.

Also reproduces the manuscript's own §7.3 numerical illustration
(d=100, eps_acc=0.5, eps_p=8, delta=1e-5, p=0.5, LD=10 -> N* ~ 1e5) using
Path A, to report the tightened N*' at that exact operating point.

Run: python3 src/e1_amplification/run_e1.py
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import privacy as P  # noqa: E402

OUT_DIR = ROOT / "results" / "e1"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def path_a_grid():
    """Valid tightening: basic vs. tight conversion, no subsampling."""
    rows = []
    T_values = [10, 50, 100, 500, 1000]
    sigma_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    delta = 1e-5
    orders = P.default_order_grid(1.001, 1024.0, 6000)
    for T_i in T_values:
        for sigma in sigma_values:
            eps_basic = P.eps_manuscript(T_i, sigma, delta)
            eps_tight = P.eps_from_orders(T_i, sigma, delta, orders, convert=P.convert_tight)
            rows.append(dict(
                T_i=T_i, sigma=sigma, delta=delta,
                eps_manuscript_basic=eps_basic,
                eps_pathA_tight=eps_tight,
                tightening_factor=eps_basic / eps_tight,
            ))
    return rows


def path_b_reference_only():
    """Invalid-but-quantified: subsampled vs non-subsampled, both via the
    library's tight-conversion accountant, isolating the (invalid) marginal
    contribution of subsampling per se from Path A's conversion-formula
    contribution, per Amendment 1's design."""
    rows = []
    if not P._HAVE_DP_ACCOUNTING:
        return rows
    T_values = [10, 50, 100, 500, 1000]
    sigma_values = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    q_values = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    delta = 1e-5
    orders = list(P.default_order_grid(1.001, 1024.0, 400))  # coarser: library call is per-order, keep runtime sane
    for T_i in T_values:
        for sigma in sigma_values:
            eps_nonsub = P.eps_nonsampled_library(T_i, sigma, delta, orders)
            for q in q_values:
                eps_sub = P.eps_subsampled(T_i, sigma, delta, q, orders)
                rows.append(dict(
                    T_i=T_i, sigma=sigma, delta=delta, q=q,
                    eps_nonsubsampled_library=eps_nonsub,
                    eps_subsampled_INVALID=eps_sub,
                    naive_amplification_factor=(eps_nonsub / eps_sub) if eps_sub > 0 else float("inf"),
                    invalid=True,
                    reason="aggregator observes participation A_i(t) directly; "
                           "subsampling-amplification hypothesis violated -- see "
                           "docs/derivations/e1_amplification.md",
                ))
    return rows


def manuscript_illustration_tightened():
    """Reproduce Section 7.3's own numerical illustration, then report the
    Path-A-tightened N* at the same operating point.

    Manuscript operating point: d=100, eps_acc=0.5, eps_p=8, delta=1e-5,
    p=0.5, LD=10 (initial optimality gap L*(F(w0)-F*) = 10).
    Manuscript's own claim: "feasibility requires a device population on
    the order of 10^5."

    We recompute N* by solving Theorem 2's condition (Eq. 19) directly:
    for the largest sigma admissible (sigma_max^2 from Eq. 17, which itself
    depends on N), find the smallest N such that
        p * T_acc <= T_priv(sigma_max(N))
    holds, using bisection on N. This is done twice: once with the
    manuscript's own T_priv (Eq. 16, basic conversion) to check we can
    reproduce their ~1e5 claim, and once with a Path-A T_priv (numeric
    inversion via the tight conversion) for the tightened N*'.
    """
    d, eps_acc, eps_p, delta, p, LD = 100, 0.5, 8.0, 1e-5, 0.5, 10.0
    G = C = 1.0  # normalized; only the ratio matters for this illustration,
                 # matching the manuscript's own presentation which does not
                 # separately vary G and C in Section 7.3.

    def T_acc_of(eps_acc_):
        return 4 * LD / eps_acc_  # Corollary 1's round requirement, L*(F0-F*) folded into LD

    def sigma_max_sq(N):
        return max(0.0, (N * p * eps_acc / 2 - (1 - p) * G ** 2) / (d * C ** 2))

    T_acc = T_acc_of(eps_acc)

    def feasible_basic(N):
        smax2 = sigma_max_sq(N)
        if smax2 <= 0:
            return False
        sigma_max = math.sqrt(smax2)
        T_priv = P.T_priv_manuscript(sigma_max, eps_p, delta)
        return p * T_acc <= T_priv

    def feasible_pathA(N):
        smax2 = sigma_max_sq(N)
        if smax2 <= 0:
            return False
        sigma_max = math.sqrt(smax2)
        orders = P.default_order_grid(1.001, 2048.0, 4000)
        # invert: largest T such that eps_from_orders(T, sigma_max, delta, tight) <= eps_p
        T_priv_A = P.invert_T_priv(
            eps_p, sigma_max, delta,
            lambda T, s, d_: P.eps_from_orders(T, s, d_, orders, convert=P.convert_tight),
            T_lo=1.0, T_hi=1e12,
        )
        return p * T_acc <= T_priv_A

    def bisect_N(feasible_fn, N_lo=1.0, N_hi=1e9, iters=60):
        if not feasible_fn(N_hi):
            return None  # not feasible even at N_hi
        if feasible_fn(N_lo):
            return N_lo
        lo, hi = N_lo, N_hi
        for _ in range(iters):
            mid = math.sqrt(lo * hi)  # geometric bisection, N spans orders of magnitude
            if feasible_fn(mid):
                hi = mid
            else:
                lo = mid
        return hi

    N_star_basic = bisect_N(feasible_basic)
    N_star_pathA = bisect_N(feasible_pathA)

    return dict(
        d=d, eps_acc=eps_acc, eps_p=eps_p, delta=delta, p=p, LD=LD,
        N_star_manuscript_basic=N_star_basic,
        N_star_pathA_tight=N_star_pathA,
        ratio=(N_star_basic / N_star_pathA) if N_star_pathA else None,
    )


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    t0 = time.time()
    print("Running Path A grid (valid, unconditional tightening)...")
    a_rows = path_a_grid()
    write_csv(OUT_DIR / "path_a_tightening.csv", a_rows)
    print(f"  {len(a_rows)} rows -> results/e1/path_a_tightening.csv")

    print("Running Path B reference-only grid (INVALID under current threat model)...")
    b_rows = path_b_reference_only()
    write_csv(OUT_DIR / "path_b_reference_only.csv", b_rows)
    print(f"  {len(b_rows)} rows -> results/e1/path_b_reference_only.csv")

    print("Reproducing manuscript Sec 7.3 illustration + Path-A-tightened N*...")
    illus = manuscript_illustration_tightened()
    with open(OUT_DIR / "manuscript_illustration_tightened.json", "w") as f:
        json.dump(illus, f, indent=2)
    print(json.dumps(illus, indent=2))

    # summary stats for the log
    tightening_factors = [r["tightening_factor"] for r in a_rows]
    summary = dict(
        n_pathA_configs=len(a_rows),
        pathA_tightening_factor_min=min(tightening_factors),
        pathA_tightening_factor_max=max(tightening_factors),
        pathA_tightening_factor_mean=sum(tightening_factors) / len(tightening_factors),
        n_pathB_configs=len(b_rows),
        manuscript_illustration=illus,
        wall_clock_seconds=time.time() - t0,
    )
    with open(OUT_DIR / "e1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
