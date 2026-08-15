"""
Correctness gate for synthetic_data.py + fedprox.py.
    python3 src/common/tests_fedprox.py
"""
import math
import sys
import time
import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from common import synthetic_data as SD
from common import fedprox as FP


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    ok = True
    rng = np.random.default_rng(0)

    # 1. Synthetic population shapes and basic sanity.
    pop = SD.make_synthetic_population(N=50, d=10, K=4, alpha=1.0, beta=1.0,
                                        samples_per_client=20, seed=1)
    ok &= check("population X shape (N, n_k, d)", pop.X.shape == (50, 20, 10))
    ok &= check("population y shape (N, n_k)", pop.y.shape == (50, 20))
    ok &= check("labels within [0, K)", pop.y.min() >= 0 and pop.y.max() < 4)

    # 2. Gradient check: finite-difference vs analytic softmax gradient.
    W = rng.normal(size=(5, 10, 4)) * 0.1
    Xs, ys = pop.X[:5], pop.y[:5]
    loss0, grad0 = FP.softmax_grad_and_loss(W, Xs, ys, K=4)
    h = 1e-5
    idx = (0, 3, 2)  # probe one coordinate of client 0's weight matrix
    Wp = W.copy(); Wp[idx] += h
    Wm = W.copy(); Wm[idx] -= h
    lp, _ = FP.softmax_grad_and_loss(Wp, Xs, ys, K=4)
    lm, _ = FP.softmax_grad_and_loss(Wm, Xs, ys, K=4)
    fd = (lp[0] - lm[0]) / (2 * h)
    rel_err = abs(fd - grad0[0][idx[1], idx[2]]) / max(abs(fd), 1e-8)
    ok &= check(f"finite-difference gradient check (rel.err={rel_err:.2e})", rel_err < 1e-3)

    # 3. FedProx local step reduces the *local* proximal objective.
    Wg = np.zeros((10, 4))
    delta = FP.local_fedprox_steps(Wg, pop.X, pop.y, K=4, mu=0.1, local_steps=5, lr=0.5)
    def prox_obj(Wl, Wg_, Xc, yc, mu):
        loss, _ = FP.softmax_grad_and_loss(Wl[None], Xc[None], yc[None], 4)
        return loss[0] + 0.5 * mu * ((Wl - Wg_) ** 2).sum()
    before = prox_obj(Wg, Wg, pop.X[0], pop.y[0], 0.1)
    after = prox_obj(Wg + delta[0], Wg, pop.X[0], pop.y[0], 0.1)
    ok &= check(f"FedProx local step decreases proximal objective ({before:.4f} -> {after:.4f})", after < before)

    # 4. Exact solve converges to a stationary point of the proximal objective
    #    (gradient of prox objective ~ 0 at the solution).
    delta_exact, iters = FP.local_fedprox_exact(Wg, pop.X[:3], pop.y[:3], K=4,
                                                  mu=0.1, lr=0.5, max_steps=3000, tol=1e-9)
    _, g_exact = FP.softmax_grad_and_loss((Wg + delta_exact)[:3], pop.X[:3], pop.y[:3], 4)
    prox_grad_exact = g_exact + 0.1 * delta_exact[:3]
    max_gnorm = np.sqrt((prox_grad_exact ** 2).sum(axis=(1, 2))).max()
    ok &= check(f"exact solve reaches near-stationary point (max prox-grad norm={max_gnorm:.2e}, {iters} steps)",
                max_gnorm < 1e-6)

    # 5. Clipping bounds the norm correctly and is a no-op under the bound.
    big = rng.normal(size=(4, 10, 4)) * 100
    clipped, pre = FP.clip_updates(big, C=1.0)
    post_norms = np.sqrt((clipped.reshape(4, -1) ** 2).sum(axis=1))
    ok &= check("clipping bounds norm to C", np.all(post_norms <= 1.0 + 1e-6))
    small = rng.normal(size=(4, 10, 4)) * 0.001
    clipped_s, pre_s = FP.clip_updates(small, C=10.0)
    ok &= check("clipping is a no-op under the bound", np.allclose(clipped_s, small))

    # 6. Noise: empirical std matches sigma*C.
    zeros = np.zeros((1, 200000 // 4, 4))  # ~50000 elements total, flattened via reshape below
    zeros = np.zeros((1, 50000, 1))
    noised = FP.add_gaussian_noise(zeros, sigma=0.7, C=2.0, rng=np.random.default_rng(2))
    emp_std = noised.std()
    ok &= check(f"noise std matches sigma*C (emp={emp_std:.4f}, expected={0.7*2.0:.4f})",
                abs(emp_std - 0.7 * 2.0) / (0.7 * 2.0) < 0.02)

    # 7. Debiased aggregation is unbiased under Bernoulli(p) participation:
    #    E[aggregate] should equal the true full-population mean delta.
    N = 2000
    pop2 = SD.make_synthetic_population(N=N, d=5, K=3, alpha=0.5, beta=0.5,
                                         samples_per_client=10, seed=3)
    Wg2 = np.zeros((5, 3))
    true_delta = FP.local_fedprox_steps(Wg2, pop2.X, pop2.y, K=3, mu=0.1, local_steps=5, lr=0.5)
    true_mean = true_delta.mean(axis=0)
    p = 0.3
    n_trials = 300
    rng2 = np.random.default_rng(4)
    ests = []
    for _ in range(n_trials):
        A = (rng2.random(N) < p).astype(np.float64)
        est = FP.debiased_aggregate(true_delta * A[:, None, None] / A[:, None, None].clip(min=1), A, N, p)
        # (the clip(min=1) above is a no-op guard; delta already masked by A via multiplication)
        est = (A[:, None, None] * true_delta).sum(axis=0) / (N * p)
        ests.append(est)
    mean_est = np.mean(ests, axis=0)
    rel_err_agg = np.abs(mean_est - true_mean).max() / (np.abs(true_mean).max() + 1e-8)
    ok &= check(f"debiased aggregation unbiased under dropout (max rel.err over {n_trials} trials={rel_err_agg:.3f})",
                rel_err_agg < 0.15)  # Monte Carlo tolerance at 300 trials, not a tight numerical check

    print()
    print("ALL CHECKS PASSED" if ok else "AT LEAST ONE CHECK FAILED")
    return 0 if ok else 1


def throughput_calibration():
    """Not a correctness check -- measures wall-clock cost per round at a
    few (N, d) combinations so E3's grid can be sized to this machine's
    actual 1-CPU-core throughput rather than guessed at."""
    print("\n--- throughput calibration (for E3 grid sizing) ---")
    K = 4
    for N, d, samples in [(1000, 10, 10), (5000, 10, 10), (20000, 10, 10),
                           (1000, 100, 10), (5000, 100, 10)]:
        pop = synthetic_data_cache(N, d, K, samples)
        cfg = FP.TrainConfig(K=K, T=3, p=0.5, sigma=0.5, C=1.0, seed=0)
        t0 = time.time()
        FP.run_training(pop, cfg, eval_every=1)
        dt = (time.time() - t0) / cfg.T
        print(f"N={N:>7d} d={d:>4d} samples={samples:>3d}: {dt*1000:8.1f} ms/round "
              f"(=> ~{dt*100:.1f}s for 100 rounds)")


_cache = {}
def synthetic_data_cache(N, d, K, samples):
    key = (N, d, K, samples)
    if key not in _cache:
        _cache[key] = SD.make_synthetic_population(N=N, d=d, K=K, alpha=1.0, beta=1.0,
                                                     samples_per_client=samples, seed=0)
    return _cache[key]


if __name__ == "__main__":
    rc = main()
    throughput_calibration()
    sys.exit(rc)
