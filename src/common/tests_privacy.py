"""
Correctness gate for privacy.py. Run before trusting any E1/E3 result.

    python3 src/common/tests_privacy.py

Exits nonzero and prints which check failed if anything is wrong. This is
deliberately not pytest -- kept as a standalone script so it can be run
identically on this machine and, later, on the R770, without a test-runner
dependency.
"""
import math
import sys
sys.path.insert(0, __file__.rsplit("/", 2)[0])  # allow `from common import privacy`
from common import privacy as P


def check(name, cond):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}")
    return cond


def main():
    all_ok = True

    # 1. eps_manuscript reproduces the manuscript's own numerical illustration.
    #    Section 7.3: "for a model of dimension d=100 with eps_acc=0.5 and
    #    eps_p=8, feasibility requires a device population on the order of
    #    10^5 when delta=1e-5, availability p=0.5 and LD=10". We can't check
    #    N* here (that needs the full Theorem 2 pipeline, done in E1's run
    #    script), but we CAN check that eps_manuscript/T_priv_manuscript
    #    round-trip correctly, which is the first-order sanity check.
    sigma, delta, eps_p = 1.2, 1e-5, 8.0
    T = P.T_priv_manuscript(sigma, eps_p, delta)
    eps_back = P.eps_manuscript(T, sigma, delta)
    all_ok &= check(
        "Eq.16 -> Eq.15 round-trip (T_priv then eps) recovers eps_p",
        math.isclose(eps_back, eps_p, rel_tol=1e-6),
    )

    # 2. lambda_star_manuscript is the true unconstrained minimizer of the
    #    basic-bound eps(lambda); verify by finite-difference around it.
    T_i, sigma = 100.0, 1.0
    lam = P.lambda_star_manuscript(T_i, sigma, delta)
    h = 1e-6
    def eps_of_lambda(l):
        r = T_i * P.rdp_gaussian(sigma, l)
        return P.convert_basic(r, l, delta)
    slope = (eps_of_lambda(lam + h) - eps_of_lambda(lam - h)) / (2 * h)
    all_ok &= check(
        "lambda_star is a stationary point of the basic-bound eps(lambda)",
        abs(slope) < 1e-3,
    )
    all_ok &= check(
        "eps at lambda_star (numeric) matches eps_manuscript (closed form)",
        math.isclose(eps_of_lambda(lam), P.eps_manuscript(T_i, sigma, delta), rel_tol=1e-6),
    )

    # 3. eps_from_orders with convert_basic, on a dense grid bracketing
    #    lambda_star, must match eps_manuscript closely (this validates the
    #    generic numerical-optimization code path against the closed form
    #    before it's reused for Path A/B, which have no closed form).
    orders = P.default_order_grid(1.001, 50.0, 20000)
    eps_numeric_basic = P.eps_from_orders(T_i, sigma, delta, orders, convert=P.convert_basic)
    rel_err = abs(eps_numeric_basic - P.eps_manuscript(T_i, sigma, delta)) / P.eps_manuscript(T_i, sigma, delta)
    all_ok &= check(
        f"generic grid optimizer (basic conversion) matches closed form (rel.err={rel_err:.2e})",
        rel_err < 1e-3,
    )

    # 4. Path A (tight conversion) must be strictly <= Path basic at every
    #    matched order, hence the optimized eps must also be strictly lower
    #    (Canonne-Kamath-Steinke is a provably tighter bound).
    eps_tight = P.eps_from_orders(T_i, sigma, delta, orders, convert=P.convert_tight)
    all_ok &= check(
        f"Path A (tight conversion) eps ({eps_tight:.4f}) < manuscript basic eps ({P.eps_manuscript(T_i, sigma, delta):.4f})",
        eps_tight < P.eps_manuscript(T_i, sigma, delta),
    )

    # 5. Library non-subsampled path must closely match Path A's own
    #    from-scratch tight-conversion number (this is the two-independent-
    #    implementations cross-check promised in Amendment 1).
    if P._HAVE_DP_ACCOUNTING:
        eps_lib = P.eps_nonsampled_library(int(T_i), sigma, delta)
        rel_err2 = abs(eps_lib - eps_tight) / eps_tight
        all_ok &= check(
            f"library (no subsampling) matches from-scratch Path A within 2% (rel.err={rel_err2:.2e})",
            rel_err2 < 0.02,
        )

        # 6. Subsampling must only ever help (lower eps) relative to the
        #    non-subsampled library baseline, for q < 1, and must recover
        #    the non-subsampled number as q -> 1.
        q_low = 0.1
        eps_sub = P.eps_subsampled(int(T_i), sigma, delta, q_low)
        all_ok &= check(
            f"subsampled eps at q={q_low} ({eps_sub:.4f}) < non-subsampled library eps ({eps_lib:.4f})",
            eps_sub < eps_lib,
        )
        eps_sub_q1 = P.eps_subsampled(int(T_i), sigma, delta, 1.0)
        rel_err3 = abs(eps_sub_q1 - eps_lib) / eps_lib
        all_ok &= check(
            f"subsampled eps at q=1.0 recovers non-subsampled library eps (rel.err={rel_err3:.2e})",
            rel_err3 < 1e-6,
        )
    else:
        print("[SKIP] dp_accounting not installed -- Path B checks skipped")

    # 7. invert_T_priv bisection matches the closed-form Eq.16 on the basic
    #    (manuscript) path, where a closed form exists to check against.
    T_closed = P.T_priv_manuscript(sigma, eps_p, delta)
    T_bisect = P.invert_T_priv(eps_p, sigma, delta, P.eps_manuscript, T_lo=1.0, T_hi=1e8)
    rel_err4 = abs(T_bisect - T_closed) / T_closed
    all_ok &= check(
        f"bisection inversion matches closed-form T_priv (rel.err={rel_err4:.2e})",
        rel_err4 < 1e-3,
    )

    print()
    print("ALL CHECKS PASSED" if all_ok else "AT LEAST ONE CHECK FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
