"""
Privacy accounting for FMCL Paper 2A.

Implements, and keeps clearly separated:
  (1) the manuscript's own closed form (Proposition 1, Eq. 15-16), reproduced
      exactly for regression-testing against the paper as written;
  (2) Path A: the same Gaussian-mechanism RDP composition, but converted to
      (eps, delta)-DP via the tighter Canonne-Kamath-Steinke (2020) bound
      instead of the manuscript's Mironov (2017) basic bound;
  (3) Path B: Poisson-subsampled RDP composition (device-level participation
      as the sampling event), via Google's `dp_accounting` library, which
      implements a numerically-verified subsampled-Gaussian accountant.

Every function is unit-tested against the manuscript's own numbers where the
manuscript gives one (see `results/e1/` and `tests_privacy.py`).

Manuscript equation references (PaperA_ConvergencePrivacy_structure_1.docx):
  Eq. 15 : eps_i(T_i) = T_i/(2 sigma^2) + sqrt(2 T_i kappa) / sigma
  Eq. 16 : T_priv(sigma) = 2 sigma^2 (sqrt(kappa+eps_p) - sqrt(kappa))^2
  Eq. 17 : sigma_max^2 = (N p eps_acc/2 - (1-p) G^2) / (d C^2)
  Eq. 19 : joint feasibility condition (Theorem 2)
where kappa = log(1/delta).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from dp_accounting import rdp as _google_rdp
    from dp_accounting import dp_event as _google_event
    _HAVE_DP_ACCOUNTING = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAVE_DP_ACCOUNTING = False


# --------------------------------------------------------------------------
# (1) Manuscript closed form — Proposition 1 / Eq. 15-16, reproduced exactly.
# --------------------------------------------------------------------------

def kappa(delta: float) -> float:
    """kappa = log(1/delta), as defined in the manuscript's notation table."""
    return math.log(1.0 / delta)


def eps_manuscript(T_i: float, sigma: float, delta: float) -> float:
    """Eq. 15: per-device privacy loss after T_i participations.

    This is the *basic* Mironov (2017) RDP-to-DP conversion, optimized in
    closed form over the RDP order lambda (the manuscript derives the
    optimal lambda* analytically; we use that closed form directly here,
    not a numerical search, so this function is a direct transcription of
    the paper, not an independent check of it).
    """
    kap = kappa(delta)
    return T_i / (2 * sigma ** 2) + math.sqrt(2 * T_i * kap) / sigma


def T_priv_manuscript(sigma: float, eps_p: float, delta: float) -> float:
    """Eq. 16: privacy-permitted rounds at noise multiplier sigma."""
    kap = kappa(delta)
    return 2 * sigma ** 2 * (math.sqrt(kap + eps_p) - math.sqrt(kap)) ** 2


def lambda_star_manuscript(T_i: float, sigma: float, delta: float) -> float:
    """Closed-form optimal RDP order used inside Eq. 15's derivation."""
    kap = kappa(delta)
    return 1 + sigma * math.sqrt(2 * kap / T_i)


# --------------------------------------------------------------------------
# Generic RDP machinery, used by both Path A and the independent-check path.
# --------------------------------------------------------------------------

def rdp_gaussian(sigma: float, order: float) -> float:
    """RDP of the Gaussian mechanism at unit sensitivity (clip C absorbed
    into sigma, matching the manuscript's parameterisation: noise is
    N(0, sigma^2 C^2 I) after clipping to norm C, so normalized sensitivity
    is 1 and RDP(order) = order / (2 sigma^2)."""
    return order / (2 * sigma ** 2)


def convert_basic(rdp_val: float, order: float, delta: float) -> float:
    """Mironov (2017), Prop. 3: the conversion the manuscript uses."""
    return rdp_val + math.log(1.0 / delta) / (order - 1)


def convert_tight(rdp_val: float, order: float, delta: float) -> float:
    """Canonne, Kamath & Steinke (2020), arXiv:2004.00010 Prop. 12 (also
    Asoodeh et al. 2020 Eq. 20): a strictly tighter RDP-to-DP conversion
    than the manuscript's basic bound, for order > 1. This is Path A."""
    if order <= 1.0:
        raise ValueError("order must be > 1")
    return rdp_val + math.log1p(-1.0 / order) - math.log(delta * order) / (order - 1)


def eps_from_orders(T_i: float, sigma: float, delta: float,
                     orders: Sequence[float], convert=convert_basic) -> float:
    """Numerically optimize the RDP order over a supplied grid, using the
    given conversion function, for T_i-fold composition of the Gaussian
    mechanism. This is the general path used to compute Path A/B numbers;
    it is verified against the closed forms above before being trusted."""
    best = math.inf
    for order in orders:
        r = T_i * rdp_gaussian(sigma, order)
        e = convert(r, order, delta)
        if e < best:
            best = e
    return best


def default_order_grid(lo: float = 1.001, hi: float = 512.0, n: int = 4000) -> np.ndarray:
    return np.geomspace(lo, hi, n)


# --------------------------------------------------------------------------
# (3) Path B — Poisson-subsampled RDP via Google's dp_accounting library.
# --------------------------------------------------------------------------

def eps_subsampled(T_i_rounds: int, sigma: float, delta: float, q: float,
                    orders: Sequence[float] | None = None) -> float:
    """Per-device (eps, delta)-DP after T_i_rounds *opportunities* to
    participate, each independently included at rate q (the FMCL
    participation probability p), using Poisson-subsampled RDP composition.

    This treats "one device-round" as the privacy-relevant sampling unit:
    at each of T_i_rounds rounds the device is drawn i.i.d. Bernoulli(q)
    (matches Assumption 4's independent-participation model exactly; NOT
    assumed to transfer to Theorem 1b's correlated setting without further
    argument -- see docs/derivations/e1_amplification.md for the boundary
    discussion this function deliberately does not resolve on its own).

    Requires the `dp_accounting` package (Google, PyPI). Raises
    RuntimeError if unavailable so a missing dependency fails loudly rather
    than silently returning a wrong number.
    """
    if not _HAVE_DP_ACCOUNTING:
        raise RuntimeError("dp_accounting not installed; pip install dp-accounting")
    acct = _google_rdp.RdpAccountant(orders=list(orders) if orders is not None else None)
    base_event = _google_event.GaussianDpEvent(sigma)
    sampled_event = _google_event.PoissonSampledDpEvent(q, base_event)
    composed = _google_event.SelfComposedDpEvent(sampled_event, int(T_i_rounds))
    acct.compose(composed)
    return acct.get_epsilon(delta)


def eps_nonsampled_library(T_i: int, sigma: float, delta: float,
                            orders: Sequence[float] | None = None) -> float:
    """Same as eps_manuscript but computed via the library end-to-end (no
    subsampling), for use as the correctness gate against eps_manuscript,
    and as the like-for-like baseline against which eps_subsampled's
    amplification benefit is measured (both go through the library's own
    tight conversion, isolating the subsampling effect from the
    conversion-formula effect -- see Amendment 1 in EXPERIMENT_PLAN.md)."""
    if not _HAVE_DP_ACCOUNTING:
        raise RuntimeError("dp_accounting not installed; pip install dp-accounting")
    acct = _google_rdp.RdpAccountant(orders=list(orders) if orders is not None else None)
    event = _google_event.SelfComposedDpEvent(_google_event.GaussianDpEvent(sigma), int(T_i))
    acct.compose(event)
    return acct.get_epsilon(delta)


@dataclass
class TighteningResult:
    T_i: int
    sigma: float
    delta: float
    q: float | None  # participation rate used for subsampling; None if n/a
    eps_manuscript_basic: float          # paper as written (Eq. 15)
    eps_pathA_tight_conversion: float    # same composition, tight conversion
    eps_pathB_library_nonsampled: float  # library, no subsampling (gate)
    eps_pathB_library_subsampled: float  # library, with subsampling (Path B)

    def ratios(self) -> dict:
        return {
            "pathA_tightening_factor": self.eps_manuscript_basic / self.eps_pathA_tight_conversion,
            "gate_agreement_rel_err": abs(self.eps_pathA_tight_conversion - self.eps_pathB_library_nonsampled)
                                        / self.eps_pathA_tight_conversion,
            "pathB_amplification_factor": self.eps_pathB_library_nonsampled / self.eps_pathB_library_subsampled
                                            if self.eps_pathB_library_subsampled > 0 else math.inf,
            "combined_tightening_factor": self.eps_manuscript_basic / self.eps_pathB_library_subsampled
                                            if self.eps_pathB_library_subsampled > 0 else math.inf,
        }


def full_comparison(T_i: int, sigma: float, delta: float, q: float,
                     orders: Sequence[float] | None = None) -> TighteningResult:
    orders = list(orders) if orders is not None else list(default_order_grid())
    eps_paper = eps_manuscript(T_i, sigma, delta)
    eps_a = eps_from_orders(T_i, sigma, delta, orders, convert=convert_tight)
    eps_gate = eps_nonsampled_library(T_i, sigma, delta, orders)
    eps_b = eps_subsampled(T_i, sigma, delta, q, orders)
    return TighteningResult(T_i, sigma, delta, q, eps_paper, eps_a, eps_gate, eps_b)


# --------------------------------------------------------------------------
# Inversions needed downstream (T_priv under each path), by bisection since
# eps(.) is monotone increasing in T_i but the tight/subsampled forms have
# no simple closed-form inverse the way Eq. 16 does for the basic bound.
# --------------------------------------------------------------------------

def invert_T_priv(eps_target: float, sigma: float, delta: float,
                   eps_fn, T_lo: float = 1.0, T_hi: float = 1e9,
                   tol: float = 1e-6, max_iter: int = 200) -> float:
    """Largest T such that eps_fn(T, sigma, delta, ...) <= eps_target, by
    bisection. eps_fn must be monotone increasing in its first argument."""
    lo, hi = T_lo, T_hi
    if eps_fn(hi, sigma, delta) < eps_target:
        raise ValueError("T_hi too small: eps_fn(T_hi) still below target")
    if eps_fn(lo, sigma, delta) > eps_target:
        return 0.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if eps_fn(mid, sigma, delta) <= eps_target:
            lo = mid
        else:
            hi = mid
        if (hi - lo) / max(hi, 1.0) < tol:
            break
    return lo
