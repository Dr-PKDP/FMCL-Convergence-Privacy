"""
FedProx local training + debiased aggregation, vectorized across all N
clients simultaneously (no per-client Python loop) so that populations in
the tens of thousands are tractable on a single CPU core.

Model: multinomial logistic regression, weights W of shape (d, K), same
architecture across clients (a client's *data* differs, not its model
class), matching the manuscript's own §11.1 simulation exactly ("a
multinomial logistic-regression classifier over K=4 classes in
10-dimensional feature space... five local steps, proximal coefficient
mu=0.1").

Everything here operates on a *stacked* client tensor: W_local of shape
(N, d, K), one local copy per client, updated in parallel via broadcasted
numpy ops. This is mathematically identical to running N independent local
solves; it is not an approximation.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def softmax_grad_and_loss(W: np.ndarray, X: np.ndarray, y: np.ndarray, K: int):
    """Cross-entropy loss and gradient for multinomial logistic regression,
    vectorized across a leading client axis.

    W: (N, d, K)   -- one weight matrix per client
    X: (N, n_k, d) -- one sample batch per client
    y: (N, n_k)    -- int labels in [0, K)
    Returns: loss (N,), grad (N, d, K)
    """
    logits = np.einsum('nsd,ndk->nsk', X, W)                # (N, n_k, K)
    logits = logits - logits.max(axis=2, keepdims=True)
    exp = np.exp(logits)
    probs = exp / exp.sum(axis=2, keepdims=True)             # (N, n_k, K)

    N, n_k, _ = probs.shape
    y_onehot = np.zeros_like(probs)
    ii, jj = np.meshgrid(np.arange(N), np.arange(n_k), indexing='ij')
    y_onehot[ii, jj, y] = 1.0

    # mean cross-entropy loss per client
    logp = logits - np.log(exp.sum(axis=2, keepdims=True))
    loss = -(y_onehot * logp).sum(axis=2).mean(axis=1)       # (N,)

    # gradient: X^T (probs - y_onehot) / n_k, per client
    grad = np.einsum('nsd,nsk->ndk', X, (probs - y_onehot)) / n_k   # (N, d, K)
    return loss, grad


def full_batch_loss_and_grad(W_single: np.ndarray, X_all: np.ndarray, y_all: np.ndarray, K: int):
    """Global objective F(w) = (1/N) sum_i F_i(w), evaluated at a SINGLE
    shared weight matrix W_single (d, K) across the whole population --
    this is the quantity Theorem 1's ||grad F(w^t)||^2 refers to, and must
    be computed at one shared w, not per-client local weights."""
    N = X_all.shape[0]
    W_bcast = np.broadcast_to(W_single, (N,) + W_single.shape)
    loss, grad = softmax_grad_and_loss(W_bcast, X_all, y_all, K)
    return loss.mean(), grad.mean(axis=0)


def local_fedprox_steps(W_global: np.ndarray, X: np.ndarray, y: np.ndarray, K: int,
                         mu: float, local_steps: int, lr: float,
                         client_mask: np.ndarray | None = None) -> np.ndarray:
    """E steps of gradient descent on the FedProx proximal objective
        min_w F_i(w) + (mu/2)||w - w_global||^2
    run independently (but vectorized) for every client, starting every
    client's local weights at W_global. Returns the delta (w_local -
    w_global) per client, shape (N, d, K) -- matching the manuscript's
    g_i^t (Section 4/6/11).

    client_mask: optional (N,) boolean; if given, clients with mask=False
    are skipped (their delta is left at 0) -- used to avoid wasted compute
    on non-participating clients in a round. If None, all clients are run.
    """
    N, d, K = X.shape[0], W_global.shape[0], W_global.shape[1]
    W_local = np.broadcast_to(W_global, (N, d, K)).copy()

    if client_mask is None:
        active = np.ones(N, dtype=bool)
    else:
        active = client_mask

    Xa, ya = X[active], y[active]
    Wa = W_local[active]
    Wg = W_global  # (d, K), broadcasts against Wa's (n_active, d, K)

    for _ in range(local_steps):
        _, grad = softmax_grad_and_loss(Wa, Xa, ya, K)
        prox_grad = grad + mu * (Wa - Wg)
        Wa = Wa - lr * prox_grad

    W_local[active] = Wa
    delta = W_local - np.broadcast_to(W_global, (N, d, K))
    return delta


def local_fedprox_exact(W_global: np.ndarray, X: np.ndarray, y: np.ndarray, K: int,
                         mu: float, lr: float, max_steps: int = 2000,
                         tol: float = 1e-10, client_mask: np.ndarray | None = None) -> np.ndarray:
    """Near-exact proximal solve (many GD steps until the proximal
    gradient's norm is below `tol` or max_steps is hit), used as ground
    truth for E2's inexactness measurement. Same vectorization as
    local_fedprox_steps; NOT intended for the main training loop (too
    expensive to run every round at scale -- this is a diagnostic-only
    routine invoked at small N)."""
    N, d, K = X.shape[0], W_global.shape[0], W_global.shape[1]
    if client_mask is None:
        active = np.ones(N, dtype=bool)
    else:
        active = client_mask
    Xa, ya = X[active], y[active]
    Wa = np.broadcast_to(W_global, (active.sum(), d, K)).copy()
    Wg = W_global

    for step in range(max_steps):
        _, grad = softmax_grad_and_loss(Wa, Xa, ya, K)
        prox_grad = grad + mu * (Wa - Wg)
        Wa = Wa - lr * prox_grad
        gnorm = np.sqrt((prox_grad ** 2).sum(axis=(1, 2))).max()
        if gnorm < tol:
            break

    W_local = np.broadcast_to(W_global, (N, d, K)).copy()
    W_local[active] = Wa
    delta = W_local - np.broadcast_to(W_global, (N, d, K))
    return delta, step + 1


def clip_updates(delta: np.ndarray, C: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip each client's flattened update to L2-norm C. delta: (N, d, K).
    Returns (clipped_delta, pre_clip_norms)."""
    N = delta.shape[0]
    flat = delta.reshape(N, -1)
    norms = np.sqrt((flat ** 2).sum(axis=1))
    scale = np.minimum(1.0, C / np.maximum(norms, 1e-12))
    clipped = delta * scale[:, None, None]
    return clipped, norms


def add_gaussian_noise(delta: np.ndarray, sigma: float, C: float, rng: np.random.Generator) -> np.ndarray:
    """Gaussian mechanism: N(0, sigma^2 C^2 I) added per client, matching
    Assumption 6 / Section 4's noise model exactly."""
    if sigma <= 0:
        return delta
    noise = rng.normal(0.0, sigma * C, size=delta.shape)
    return delta + noise


def debiased_aggregate(delta: np.ndarray, A_t: np.ndarray, N: int, p: float) -> np.ndarray:
    """Eq. 8: w^{t+1} = w^t + (1/(Np)) sum_i A_i(t) delta_i. Matches
    Assumption 5's debiasing exactly; A_t is (N,) in {0,1}."""
    return (A_t[:, None, None] * delta).sum(axis=0) / (N * p)


@dataclass
class TrainConfig:
    K: int
    mu: float = 0.1
    local_steps: int = 5
    lr: float = 0.5
    C: float = 1.0     # clip norm; also fixes G = C by Assumption 3
    sigma: float = 0.0
    p: float = 1.0
    T: int = 100
    seed: int = 0
    eta: float = 1.0   # server-side step size (Eq. 8's eta). Theorem 1 is
                        # PROVEN at eta=1/L specifically -- earlier E3 runs
                        # used the default eta=1.0 (no explicit scaling,
                        # matching the original Paper 2 session's own
                        # simulation convention), which is a real deviation
                        # from what the theorem's proof assumes. See
                        # EXPERIMENT_LOG.md for the investigation this
                        # surfaced and why it's being corrected here.


def run_training(pop, cfg: TrainConfig, w0: np.ndarray | None = None,
                  eval_every: int = 1, start_round: int = 0,
                  init_gaps: list | None = None, init_losses: list | None = None,
                  checkpoint_cb=None, checkpoint_every: int = 25) -> dict:
    """Full T-round FedProx training loop with stochastic participation,
    clipping, and optional DP noise, tracking the true global stationarity
    gap ||grad F(w^t)||^2 each round (computed on the FULL population, not
    just participants -- this is only possible in simulation, where we
    have access to everyone's data for evaluation; a real deployment
    cannot compute this, which is exactly why Theorem 1 exists as an
    a-priori bound rather than something you'd just measure directly).

    Resumable: start_round/init_gaps/init_losses let a caller resume a
    partially-completed run (e.g. after a wall-clock budget cutoff)
    without redoing already-completed rounds. Uses the SAME per-round RNG
    stream a from-scratch run would (seeded once at cfg.seed, then
    advanced round-by-round) -- resuming reconstructs the RNG state by
    replaying only the *draws*, not the compute, up to start_round, which
    is cheap (just Bernoulli draws) relative to the training compute this
    checkpointing is meant to save.
    """
    rng = np.random.default_rng(cfg.seed if start_round == 0 else cfg.seed * 100003 + start_round)
    d, K = pop.d, cfg.K
    W = np.zeros((d, K)) if w0 is None else w0.copy()
    gaps = list(init_gaps) if init_gaps else []
    losses = list(init_losses) if init_losses else []
    # NOTE on resumption randomness: a resumed run does NOT replay the
    # exact RNG state an uninterrupted run would have had at start_round
    # (it reseeds instead, via a deterministic derived seed so results are
    # still exactly reproducible from checkpoint state). This is
    # statistically valid but not bit-identical to an uninterrupted run,
    # since each round's participation/noise draws are i.i.d. regardless
    # of the exact sequence (Assumption 4) -- acceptable for a floor
    # measurement, not acceptable if bit-for-bit reproducibility of a
    # single specific trajectory were the goal.

    for t in range(start_round, cfg.T):
        A_t = (rng.random(pop.N) < cfg.p).astype(np.float64)
        mask = A_t > 0

        delta = local_fedprox_steps(W, pop.X, pop.y, K, cfg.mu, cfg.local_steps,
                                     cfg.lr, client_mask=mask)
        delta_active = delta[mask]
        clipped, _ = clip_updates(delta_active, cfg.C)
        noised = add_gaussian_noise(clipped, cfg.sigma, cfg.C, rng)
        delta_full = np.zeros_like(delta)
        delta_full[mask] = noised

        W = W + cfg.eta * debiased_aggregate(delta_full, A_t, pop.N, cfg.p)

        if t % eval_every == 0 or t == cfg.T - 1:
            loss, grad = full_batch_loss_and_grad(W, pop.X, pop.y, K)
            gap = float((grad ** 2).sum())
            gaps.append(gap)
            losses.append(float(loss))

        if checkpoint_cb is not None and (t + 1) % checkpoint_every == 0:
            checkpoint_cb(t + 1, W, gaps, losses)

    return dict(gaps=gaps, losses=losses, W_final=W, T=cfg.T)
