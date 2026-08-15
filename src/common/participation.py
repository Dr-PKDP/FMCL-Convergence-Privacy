"""
Participation generators for FMCL Paper 2A.

iid_participation: matches Assumption 4's baseline (independent Bernoulli(p)
each round), already used inline in fedprox.run_training -- reproduced
here as a named function so E4 can swap it for the Markov generator
without duplicating logic.

markov_participation: 2-state Markov chain per device, parametrized by
(p, lambda) as derived in docs/derivations/e4_markov_correlation.md --
lambda=0 recovers iid_participation exactly (tau=q=p).
"""
from __future__ import annotations
import numpy as np


def iid_participation(N: int, T: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """(T, N) boolean array, A[t,i] = 1 if device i participates in round t."""
    return rng.random((T, N)) < p


def markov_params(p: float, lam: float) -> tuple[float, float]:
    """tau, q from (p, lambda) -- see e4_markov_correlation.md Section 1."""
    tau = p + lam * (1 - p)
    q = p * (1 - lam)
    return tau, q


def markov_participation(N: int, T: int, p: float, lam: float,
                          rng: np.random.Generator, burn_in: int = 50) -> np.ndarray:
    """(T, N) boolean array under the 2-state Markov model, independent
    across devices, each individually started from the stationary
    distribution (via burn-in) so the process is stationary from t=0 in
    the returned array."""
    tau, q = markov_params(p, lam)
    A = np.zeros((T + burn_in, N), dtype=bool)
    A[0] = rng.random(N) < p  # start at stationary marginal
    for t in range(1, T + burn_in):
        prev = A[t - 1]
        prob_stay_or_move = np.where(prev, tau, q)
        A[t] = rng.random(N) < prob_stay_or_move
    return A[burn_in:]


def empirical_lag1_autocorr(A: np.ndarray) -> float:
    """Sanity-check helper: empirical lag-1 autocorrelation of a (T,N)
    boolean participation array, averaged across devices."""
    A = A.astype(float)
    T = A.shape[0]
    x0 = A[:-1] - A[:-1].mean(axis=0, keepdims=True)
    x1 = A[1:] - A[1:].mean(axis=0, keepdims=True)
    num = (x0 * x1).sum(axis=0)
    den = np.sqrt((x0 ** 2).sum(axis=0) * (x1 ** 2).sum(axis=0))
    valid = den > 1e-9
    return float(np.mean(num[valid] / den[valid])) if valid.any() else 0.0
