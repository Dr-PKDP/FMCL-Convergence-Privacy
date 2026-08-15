"""
E2 -- numerical companion to docs/derivations/e2_inexactness.md.

For a fixed set of clients (Synthetic(1,1), matching manuscript's Scenario
C / Section 11 and this repo's E3 convention), measure the inexactness
delta_i(E) = ||w_E - w_i^*|| for E in {1,2,5,10,20,50} and mu in
{0.01,0.1,0.5,1.0}, against a near-exact ground-truth solve (many GD
steps, tight tolerance) -- exactly the grid specified in EXPERIMENT_PLAN.md.

Checks:
  1. log(delta) vs E slope matches the derived rate log(r(mu,L)),
     r = sqrt(L/(L+mu)), to within a factor of 2 (pre-committed tolerance).
  2. At the manuscript's own (mu=0.1, E=5) operating point, delta^2's size
     relative to V_ind at representative (N, sigma) from this repo's E3
     grid -- is Assumption 2's idealization negligible or material.

Run: python3 src/e2_inexactness/run_e2.py
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import synthetic_data as SD   # noqa: E402
from common import fedprox as FP          # noqa: E402

OUT_DIR = ROOT / "results" / "e2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CLIENTS = 20
D, K = 10, 4
SAMPLES_PER_CLIENT = 10
ALPHA_BETA = (1.0, 1.0)
LR = 0.5  # matches manuscript's Section 11.1 / this repo's local_fedprox_steps default
MU_GRID = [0.01, 0.1, 0.5, 1.0]
E_GRID = [1, 2, 5, 10, 20, 50]
SEEDS = [0, 1, 2, 3, 4]

# L used for the theoretical rate comparison: this repo's own calibrated
# L for d=10 (results/e3/calibration/calib_d10.json), reused here rather
# than re-estimated, since it is the same model family/data-generating
# process (Synthetic(1,1), d=10) E3 already calibrated it for.
def load_L():
    calib_path = ROOT / "results" / "e3" / "calibration" / "calib_d10.json"
    with open(calib_path) as f:
        return json.load(f)["L"]


def measure_inexactness(pop, mu, E, lr, seed):
    """For each client, compute ||w_E - w_exact|| where w_exact is a
    near-exact proximal solve (many steps, tight tolerance)."""
    rng = np.random.default_rng(seed)
    W0 = np.zeros((D, K))

    delta_E = FP.local_fedprox_steps(W0, pop.X, pop.y, K, mu=mu, local_steps=E, lr=lr)
    delta_exact, iters = FP.local_fedprox_exact(W0, pop.X, pop.y, K, mu=mu, lr=lr,
                                                  max_steps=3000, tol=1e-10)
    gap = np.sqrt(((delta_E - delta_exact) ** 2).reshape(pop.N, -1).sum(axis=1))
    grad_norm_at_w0 = np.sqrt(
        (FP.softmax_grad_and_loss(np.broadcast_to(W0, (pop.N, D, K)), pop.X, pop.y, K)[1]
         .reshape(pop.N, -1) ** 2).sum(axis=1)
    )
    return gap, grad_norm_at_w0, iters


def main():
    L = load_L()
    print(f"Using calibrated L (d=10) = {L:.4f}")

    rows = []
    for seed in SEEDS:
        pop = SD.make_synthetic_population(N=N_CLIENTS, d=D, K=K, alpha=ALPHA_BETA[0],
                                            beta=ALPHA_BETA[1], samples_per_client=SAMPLES_PER_CLIENT,
                                            seed=2000 + seed)
        for mu in MU_GRID:
            for E in E_GRID:
                gap, grad0, iters = measure_inexactness(pop, mu, E, LR, seed)
                for client_idx in range(N_CLIENTS):
                    rows.append(dict(
                        seed=seed, mu=mu, E=E, client=client_idx,
                        measured_delta=float(gap[client_idx]),
                        grad_norm_at_w0=float(grad0[client_idx]),
                        exact_solve_iters=iters,
                    ))
        print(f"  seed {seed} done ({len(MU_GRID)*len(E_GRID)} (mu,E) cells)")

    out_csv = OUT_DIR / "e2_inexactness_grid.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {out_csv}")

    # --- Check 1: rate comparison ---
    import numpy as np
    rate_results = []
    for mu in MU_GRID:
        Es, log_deltas = [], []
        for E in E_GRID:
            vals = [r["measured_delta"] for r in rows if r["mu"] == mu and r["E"] == E and r["measured_delta"] > 0]
            if vals:
                Es.append(E)
                log_deltas.append(np.log(np.mean(vals)))
        if len(Es) >= 2:
            slope, intercept = np.polyfit(Es, log_deltas, 1)
        else:
            slope, intercept = float("nan"), float("nan")
        theoretical_slope = 0.5 * np.log(L / (L + mu))
        rate_results.append(dict(
            mu=mu, empirical_slope=float(slope), theoretical_slope=float(theoretical_slope),
            ratio=float(slope / theoretical_slope) if theoretical_slope != 0 else None,
            within_factor_2=bool(theoretical_slope != 0 and 0.5 <= (slope / theoretical_slope) <= 2.0),
        ))

    # --- Check 2: magnitude at manuscript operating point (mu=0.1, E=5) ---
    op_rows = [r for r in rows if r["mu"] == 0.1 and r["E"] == 5]
    delta_op = np.array([r["measured_delta"] for r in op_rows])
    delta_sq_mean = float((delta_op ** 2).mean())

    # Compare against V_ind at representative (N, sigma) from E3's grid
    e3_traj_dir = ROOT / "results" / "e3" / "trajectories"
    v_ind_comparisons = []
    if e3_traj_dir.exists():
        seen = set()
        for traj_file in sorted(e3_traj_dir.glob("*.json")):
            if "ckpt" in traj_file.name:
                continue
            with open(traj_file) as f:
                rec = json.load(f)
            key = (rec["d"], rec["N"])
            if key in seen or rec["d"] != D:
                continue
            seen.add(key)
            N, sigma = rec["N"], rec["sigma"]
            p = 0.5
            V_ind = (1 - p) * 1.0 / (p * N) + D * sigma ** 2 * 1.0 / (N * p)
            v_ind_comparisons.append(dict(
                N=N, sigma=sigma, V_ind=V_ind, delta_sq=delta_sq_mean,
                delta_sq_over_V_ind=delta_sq_mean / V_ind if V_ind > 0 else None,
            ))

    summary = dict(
        L_used=L, rate_check=rate_results,
        operating_point=dict(mu=0.1, E=5, mean_delta=float(delta_op.mean()),
                              mean_delta_sq=delta_sq_mean, n=len(delta_op)),
        v_ind_comparisons=v_ind_comparisons,
    )
    with open(OUT_DIR / "e2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
