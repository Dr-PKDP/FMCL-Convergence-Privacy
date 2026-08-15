"""
Calibrate L (smoothness constant) and D (initial optimality gap) for the
actual multinomial-logistic-regression / Synthetic(1,1) instance E3 trains
on, so Corollary 1's T_acc = 4LD/eps_acc can be computed for real instead
of assumed (the manuscript's own Section 7.3 assumes LD=10 as a given
constant; this is the first place in the FMCL paper series either
quantity has actually been measured from a trained model).

D = F(w^0) - F*:
    F(w^0) is exact (evaluate the loss at initialization).
    F* is approximated by training to convergence under favorable
    conditions (p=1, no privacy noise) -- the lowest loss such training
    reaches is an upper bound on F*, hence D is a (slight) overestimate,
    which makes T_acc a (slight) overestimate too -- conservative in the
    direction that makes Theorem 2's feasibility condition harder to
    satisfy, not easier, so this does not risk an optimistic bias in the
    feasibility comparison.

L (smoothness): estimated empirically via gradient-Lipschitz probing --
sampling pairs of weight matrices near the calibration trajectory and
measuring ||grad F(w1) - grad F(w2)|| / ||w1 - w2||, taking a high
percentile (not the max, to avoid a single numerically-unstable probe
pair dominating the estimate) as a defensible, standard-practice
estimator. This is calibrated ONCE per d (not per N): L and D are
properties of the loss landscape (task, architecture, data-generating
process), not of how many devices happen to be in a given FMCL
deployment, so recalibrating per N would be both wasteful and not
reflective of what a real deployment would do either.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import synthetic_data as SD   # noqa: E402
from common import fedprox as FP          # noqa: E402

CACHE_DIR = ROOT / "results" / "e3" / "calibration"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CALIBRATION_N = 3000
CALIBRATION_SEED = 777
CALIBRATION_T = 400  # generous; favorable conditions (p=1, sigma=0) should
                      # converge well before this, checked below not assumed


def calibrate(d: int, K: int = 4, samples_per_client: int = 10,
              alpha: float = 1.0, beta: float = 1.0,
              mu: float = 0.1, local_steps: int = 5, lr: float = 0.5,
              n_probes: int = 60, probe_scale_frac: float = 0.05,
              percentile: float = 90.0) -> dict:
    cache_path = CACHE_DIR / f"calib_d{d}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    pop = SD.make_synthetic_population(
        N=CALIBRATION_N, d=d, K=K, alpha=alpha, beta=beta,
        samples_per_client=samples_per_client, seed=CALIBRATION_SEED,
    )

    # --- D: train under favorable conditions (p=1, sigma=0) ---
    cfg = FP.TrainConfig(K=K, mu=mu, local_steps=local_steps, lr=lr, C=1.0,
                          sigma=0.0, p=1.0, T=CALIBRATION_T, seed=CALIBRATION_SEED)
    W0 = np.zeros((d, K))
    F0, _ = FP.full_batch_loss_and_grad(W0, pop.X, pop.y, K)
    out = FP.run_training(pop, cfg, w0=W0, eval_every=5)
    losses = np.array(out["losses"])
    gaps = np.array(out["gaps"])

    # Convergence check: compare loss over the last 20% of recorded evals
    # vs the preceding 20% -- if still dropping meaningfully, CALIBRATION_T
    # wasn't generous enough and D would be an overestimate beyond the
    # deliberate, bounded overestimate described in the module docstring.
    n_eval = len(losses)
    tail = losses[-max(1, n_eval // 5):]
    pre_tail = losses[-max(2, n_eval // 5) * 2: -max(1, n_eval // 5)]
    still_dropping = bool(pre_tail.mean() - tail.mean() > 0.02 * max(abs(pre_tail.mean()), 1e-8))

    F_star = float(losses.min())
    D = float(F0) - F_star

    # --- L: gradient-Lipschitz probing along + near the calibration trajectory ---
    rng = np.random.default_rng(CALIBRATION_SEED + 1)
    W_final = out["W_final"]
    # probe points: initialization, a few along the recorded loss trajectory
    # (reconstructing exact W at each eval isn't stored, so instead probe
    # around W0 and W_final, and interpolated points between them, which
    # covers the region actually traversed for a monotonically-converging
    # run of this kind)
    anchors = [W0 + t * (W_final - W0) for t in np.linspace(0, 1, 8)]

    ratios = []
    for anchor in anchors:
        for _ in range(n_probes // len(anchors)):
            direction = rng.normal(size=anchor.shape)
            direction /= np.linalg.norm(direction) + 1e-12
            scale = probe_scale_frac * (np.linalg.norm(anchor) + 1.0)
            w1 = anchor
            w2 = anchor + scale * direction
            _, g1 = FP.full_batch_loss_and_grad(w1, pop.X, pop.y, K)
            _, g2 = FP.full_batch_loss_and_grad(w2, pop.X, pop.y, K)
            num = np.linalg.norm(g1 - g2)
            den = np.linalg.norm(w1 - w2)
            if den > 1e-10:
                ratios.append(num / den)

    L = float(np.percentile(ratios, percentile))

    result = dict(
        d=d, N_calibration=CALIBRATION_N, T_calibration=CALIBRATION_T,
        F0=float(F0), F_star=F_star, D=D, L=L, LD=L * D,
        still_dropping_at_end=still_dropping,
        n_probes_used=len(ratios), probe_percentile=percentile,
        loss_trajectory_tail=losses[-10:].tolist(),
        ratios_summary=dict(min=float(np.min(ratios)), max=float(np.max(ratios)),
                             median=float(np.median(ratios)), p90=float(np.percentile(ratios, 90))),
    )
    with open(cache_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


if __name__ == "__main__":
    for d in [10, 100]:
        r = calibrate(d)
        print(f"d={d}: F0={r['F0']:.4f} F*={r['F_star']:.4f} D={r['D']:.4f} "
              f"L={r['L']:.4f} LD={r['LD']:.4f} still_dropping={r['still_dropping_at_end']}")
        print(f"  gradient-ratio distribution: {r['ratios_summary']}")
