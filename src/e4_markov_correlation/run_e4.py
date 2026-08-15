"""
E4 -- numerical companion to docs/derivations/e4_markov_correlation.md.

Four checks, matching the derivation doc's Section 5:
  1. Lemma E4.1: E[b_i^t] = 0 over stationary history.
  2. Lemma E4.2: ||beta_t|| <= lambda*G/p bound holds.
  3. Single-round check (manuscript's own methodology): confirms it is
     blind to this effect when participation is drawn from the marginal
     without conditioning on history -- makes Section 2's point concrete.
  4. Multi-round training comparison (real fedprox.py harness): does the
     (lambda*G/p)^2 term predict observed extra floor inflation.

Run: python3 src/e4_markov_correlation/run_e4.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import synthetic_data as SD   # noqa: E402
from common import fedprox as FP          # noqa: E402
from common import participation as PT    # noqa: E402

OUT_DIR = ROOT / "results" / "e4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

P_GRID = [0.1, 0.2, 0.5, 1.0]
LAM_GRID = [0.0, 0.1, 0.2, 0.35, 0.5]
N_MC = 100_000  # matches manuscript's own Section 6 Monte Carlo sample size


def check1_2_bias_bound(G=1.0):
    """Checks 1 and 2: draw many (A_i(t-1), A_i(t)) pairs at stationarity,
    compute b_i^t = E[A_i(t)|A_i(t-1)] - p directly from the known
    transition probabilities (exact, not simulated -- these ARE the
    conditional means by construction), verify Lemma E4.1's zero-mean-
    over-history property, and verify Lemma E4.2's aggregate bound via a
    Monte Carlo aggregate-bias simulation at N=200 (matching the
    manuscript's own single-round check's population size)."""
    results = []
    rng = np.random.default_rng(10)
    for p in P_GRID:
        if p >= 1.0:
            continue  # lambda undefined / degenerate at p=1 (always available)
        for lam in LAM_GRID:
            tau, q = PT.markov_params(p, lam)
            b_given_1 = tau - p  # = lambda(1-p)
            b_given_0 = q - p    # = -lambda*p
            mean_b = p * b_given_1 + (1 - p) * b_given_0  # Lemma E4.1: should be ~0
            max_b = max(abs(b_given_1), abs(b_given_0))
            lemma_e4_1_bound = lam  # |b_i^t| <= lambda

            # Lemma E4.2 aggregate check: simulate N=200 devices' A(t-1) at
            # stationarity, compute realized aggregate bias beta_t for a
            # fixed synthetic g_i^t (unit-norm, random direction per
            # device, matching the G=1 clip-bound convention) and check
            # ||beta_t|| against the derived bound lambda*G/p.
            N = 200
            n_trials = 2000
            beta_norms = []
            for _ in range(n_trials):
                A_prev = rng.random(N) < p
                b_i = np.where(A_prev, b_given_1, b_given_0)
                g_i = rng.normal(size=(N, 5))
                g_i /= np.linalg.norm(g_i, axis=1, keepdims=True)  # ||g_i||=G=1
                beta_t = (b_i[:, None] * g_i).sum(axis=0) / (N * p)
                beta_norms.append(np.linalg.norm(beta_t))
            beta_norms = np.array(beta_norms)
            bound = lam * G / p

            results.append(dict(
                p=p, lam=lam, mean_b_over_history=mean_b,
                lemma_e4_1_holds=bool(abs(mean_b) < 1e-9),
                max_beta_observed=float(beta_norms.max()),
                mean_beta_observed=float(beta_norms.mean()),
                lemma_e4_2_bound=bound,
                lemma_e4_2_holds=bool(beta_norms.max() <= bound + 1e-9),
            ))
    return results


def check3_single_round_null(G=1.0):
    """The manuscript's own Section 6 methodology: draw participation
    directly from the STATIONARY MARGINAL (no history conditioning) and
    check aggregation-error variance against V_ind/V_corr. Demonstrates
    this is blind to lambda by construction -- results should be
    statistically indistinguishable across lambda when done this way,
    which is the point being made, not a bug."""
    results = []
    rng = np.random.default_rng(11)
    d, C = 10, 1.0
    for p in [0.2, 0.5]:
        for lam in [0.0, 0.3, 0.5]:  # lambda should NOT matter here by construction
            N = 200
            g = rng.normal(size=(N, d))
            g = g / np.linalg.norm(g, axis=1, keepdims=True) * C  # ||g_i||=C=G
            true_mean = g.mean(axis=0)

            ests = []
            for _ in range(N_MC // 100):  # scaled down from manuscript's 1e5 for wall-clock budget; still >> needed for a mean estimate at this tolerance
                A = rng.random(N) < p  # drawn from MARGINAL directly, no history
                est = (A[:, None] * g).sum(axis=0) / (N * p)
                ests.append(est)
            ests = np.array(ests)
            var_ind_theory = (1 - p) * (C ** 2) / (p * N)  # matches Lemma 1's per-device term (G=C)
            emp_var = ((ests - true_mean) ** 2).sum(axis=1).mean()

            results.append(dict(
                p=p, lam=lam, emp_var=float(emp_var), theory_var_ind=float(var_ind_theory),
                rel_err=float(abs(emp_var - var_ind_theory) / var_ind_theory),
            ))
    return results


def check4_training_comparison():
    """Real multi-round training: compare i.i.d. vs Markov-correlated
    participation (same marginal p) using fedprox.py, and check whether
    the extra floor inflation matches (lambda*G/p)^2."""
    d, K = 10, 4
    N = 2000
    samples_per_client = 10
    p = 0.5
    T = 150
    G = C = 1.0

    pop = SD.make_synthetic_population(N=N, d=d, K=K, alpha=1.0, beta=1.0,
                                        samples_per_client=samples_per_client, seed=3000)

    # calibrated L (reuse E3's d=10 calibration)
    calib_path = ROOT / "results" / "e3" / "calibration" / "calib_d10.json"
    with open(calib_path) as f:
        L = json.load(f)["L"]
    eta = 1.0 / L

    results = []
    for lam in [0.0, 0.2, 0.4]:
        for seed in [0, 1]:
            rng = np.random.default_rng(100 + seed)
            if lam == 0.0:
                A = PT.iid_participation(N, T, p, rng)
            else:
                A = PT.markov_participation(N, T, p, lam, rng)

            W = np.zeros((d, K))
            gaps = []
            for t in range(T):
                mask = A[t]
                delta = FP.local_fedprox_steps(W, pop.X, pop.y, K, mu=0.1, local_steps=5, lr=0.5,
                                                client_mask=mask)
                delta_active = delta[mask]
                clipped, _ = FP.clip_updates(delta_active, C)
                delta_full = np.zeros_like(delta)
                delta_full[mask] = clipped
                A_t = mask.astype(np.float64)
                W = W + eta * FP.debiased_aggregate(delta_full, A_t, N, p)
                loss, grad = FP.full_batch_loss_and_grad(W, pop.X, pop.y, K)
                gaps.append(float((grad ** 2).sum()))

            gaps = np.array(gaps)
            cum_mean_final = float(np.cumsum(gaps)[-1] / len(gaps))
            results.append(dict(lam=lam, seed=seed, final_cumavg_gap=cum_mean_final,
                                 last30_mean=float(gaps[-30:].mean())))

    predicted_extra = {lam: (lam * G / p) ** 2 for lam in [0.0, 0.2, 0.4]}
    return results, predicted_extra, L


def main():
    print("=== Checks 1-2: bias bound ===")
    r12 = check1_2_bias_bound()
    n_fail_1 = sum(1 for r in r12 if not r["lemma_e4_1_holds"])
    n_fail_2 = sum(1 for r in r12 if not r["lemma_e4_2_holds"])
    print(f"Lemma E4.1 (zero mean): {len(r12)-n_fail_1}/{len(r12)} pass")
    print(f"Lemma E4.2 (bound holds): {len(r12)-n_fail_2}/{len(r12)} pass")

    print("\n=== Check 3: single-round null result (manuscript's own methodology) ===")
    r3 = check3_single_round_null()
    for r in r3:
        print(f"  p={r['p']} lam={r['lam']}: emp_var={r['emp_var']:.6f} theory={r['theory_var_ind']:.6f} rel_err={r['rel_err']:.4f}")

    print("\n=== Check 4: multi-round training comparison ===")
    r4, predicted, L = check4_training_comparison()
    for r in r4:
        print(f"  lambda={r['lam']} seed={r['seed']}: final_cumavg_gap={r['final_cumavg_gap']:.4f} last30={r['last30_mean']:.4f}")
    print(f"  predicted extra floor (lambda*G/p)^2: {predicted}")

    out = dict(check1_2=r12, check3=r3, check4=dict(results=r4, predicted_extra=predicted, L=L))
    with open(OUT_DIR / "e4_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWritten -> {OUT_DIR / 'e4_results.json'}")


if __name__ == "__main__":
    main()
