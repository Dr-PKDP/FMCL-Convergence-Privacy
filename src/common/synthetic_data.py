"""
Synthetic(alpha, beta) federated benchmark generator.

Reproduces the generative process from Li, Sahu, Zaheer, Sanjabi, Talwalkar
& Smith, "Federated Optimization in Heterogeneous Networks" (MLSys 2020) --
the manuscript's own reference [8], and the exact generator its §11.1
already uses for Scenarios B and C ("the standard Synthetic(alpha, beta)
federated benchmark of [8]"). Reusing it here (rather than inventing a new
generator for E2/E3) keeps every new experiment in this repo consistent
with what the manuscript already validates in Section 11, and is why E3's
results can be compared to Section 11's existing Table 8/9 numbers on the
same footing.

Generative process, per client k, given (alpha, beta), feature dimension d,
class count K, and n_k local samples:
    v_k        ~ N(0, beta)                        (scalar)
    B_k        ~ N(v_k, 1)              in R^d      (feature-mean shift)
    u_k        ~ N(0, alpha)                        (scalar)
    W_k        ~ N(u_k, 1)              in R^{d x K} (client's true weights)
    b_k        ~ N(u_k, 1)              in R^K       (client's true bias)
    Sigma_jj   = j^{-1.2}  for j = 1..d              (fixed feature covariance,
                                                       same for every client)
    x_{k,i}    ~ N(B_k, Sigma)          in R^d, i = 1..n_k
    y_{k,i}    = argmax_c softmax(W_k^T x_{k,i} + b_k)[c]

alpha controls how much clients' TRUE MODELS differ from each other; beta
controls how much clients' INPUT FEATURE DISTRIBUTIONS differ. The
manuscript's Scenario C (primary, used throughout Sections 4-9) is
Synthetic(1, 1); Scenario B is Synthetic(0.5, 0.5). This module defaults to
(1, 1) to match the manuscript's own primary setting.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class FederatedPopulation:
    """N clients' local (X, y) data, stacked for vectorized training (every
    client has the same sample count by construction, so a dense array is
    valid and lets local FedProx updates for all N clients run as batched
    numpy operations instead of a Python loop over clients -- essential for
    N in the tens of thousands on a single CPU core).

    X: (N, samples_per_client, d)
    y: (N, samples_per_client) int in [0, K)
    W_true, b_true kept only for diagnostics -- training code must not use
    them, matching a real deployment's information constraints.
    """
    X: np.ndarray
    y: np.ndarray
    d: int
    K: int
    N: int
    samples_per_client: int
    W_true: np.ndarray  # (N, d, K) -- diagnostics only
    b_true: np.ndarray  # (N, K)    -- diagnostics only
    seed: int
    alpha: float
    beta: float


def make_synthetic_population(N: int, d: int, K: int, alpha: float, beta: float,
                               samples_per_client: int, seed: int) -> FederatedPopulation:
    rng = np.random.default_rng(seed)

    j = np.arange(1, d + 1)
    sigma_diag = j ** (-1.2)  # Sigma_jj = j^-1.2, fixed across clients

    v = rng.normal(0, np.sqrt(beta), size=N)
    B = rng.normal(v[:, None], 1.0, size=(N, d))
    u = rng.normal(0, np.sqrt(alpha), size=N)
    W = rng.normal(u[:, None, None], 1.0, size=(N, d, K))
    b = rng.normal(u[:, None], 1.0, size=(N, K))

    # Vectorized sample draw across all N clients at once: X ~ N(B_k, Sigma)
    # per client k, same diagonal Sigma for all clients.
    noise = rng.normal(0.0, 1.0, size=(N, samples_per_client, d))
    X = B[:, None, :] + noise * np.sqrt(sigma_diag)[None, None, :]  # (N, n_k, d)

    logits = np.einsum('nsd,ndk->nsk', X, W) + b[:, None, :]        # (N, n_k, K)
    logits -= logits.max(axis=2, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=2, keepdims=True)
    # Manuscript's own text: "y = argmax_c softmax(...)" -- deterministic
    # label from the client's true model, matched literally here (argmax,
    # not a sampled draw), for exact consistency with Section 11's stated
    # generator.
    y = probs.argmax(axis=2)  # (N, n_k)

    return FederatedPopulation(X=X, y=y, d=d, K=K, N=N,
                                samples_per_client=samples_per_client,
                                W_true=W, b_true=b, seed=seed, alpha=alpha, beta=beta)
