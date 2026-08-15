"""
Correctness gate for dp_wrapper.py, run against a tiny toy model and
synthetic data -- validates the mechanics (sampling, clipping, noising,
aggregation, state_dict flatten/unflatten round-trip) before any of it
is pointed at a real model or real server time.

    python3 src/e7_neural_feasibility/test_dp_wrapper.py
"""
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e7_neural_feasibility import dp_wrapper as DW  # noqa: E402


def check(name, cond):
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")
    return cond


def main():
    ok = True
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    sd = model.state_dict()

    # 1. flatten/unflatten round-trip
    flat = DW.flatten_state_dict(sd)
    back = DW.unflatten_to_state_dict(flat, sd)
    ok &= check("flatten/unflatten round-trip is exact",
                all(torch.equal(sd[k], back[k]) for k in sd))

    # 2. clipping bounds the norm and is a no-op under the bound
    big = torch.randn(50) * 100
    clipped, raw = DW.clip_flat(big, C=1.0)
    ok &= check(f"clipping bounds norm to C (post-norm={float(torch.linalg.norm(clipped)):.4f})",
                float(torch.linalg.norm(clipped)) <= 1.0 + 1e-4)
    small = torch.randn(50) * 0.001
    clipped_s, _ = DW.clip_flat(small, C=10.0)
    ok &= check("clipping is a no-op under the bound", torch.allclose(clipped_s, small))

    # 3. noise scale matches sigma*C
    gen = torch.Generator().manual_seed(1)
    zeros = torch.zeros(200000)
    noised = DW.add_noise_flat(zeros, sigma=0.7, C=2.0, generator=gen)
    emp_std = float(noised.std())
    ok &= check(f"noise std matches sigma*C (emp={emp_std:.4f}, expected={0.7*2.0:.4f})",
                abs(emp_std - 1.4) / 1.4 < 0.02)

    # 4. run_one_round: fixed-K sampling, deterministic per-client "training"
    #    (each client's local_train_fn just nudges every param by a fixed,
    #    client-specific constant, so the aggregate is checkable by hand).
    client_ids = list(range(10))
    py_rng = random.Random(42)
    torch_rng = torch.Generator().manual_seed(42)

    def fake_local_train(global_sd, cid):
        # nudge every parameter by +cid*0.01 (deterministic, so the
        # resulting aggregate is exactly checkable)
        return {k: v + cid * 0.01 for k, v in global_sd.items()}

    new_sd, result = DW.run_one_round(
        global_state_dict=sd, client_ids=client_ids, K=4,
        local_train_fn=fake_local_train, C=100.0, sigma=0.0,  # C huge: no clipping; sigma=0: no noise
        rng=torch_rng, py_rng=py_rng,
    )
    ok &= check(f"run_one_round selects exactly K=4 clients (selected={result.selected_clients})",
                len(result.selected_clients) == 4)
    expected_bump = sum(result.selected_clients) / len(result.selected_clients) * 0.01
    actual_bump = float((DW.flatten_state_dict(new_sd) - DW.flatten_state_dict(sd)).mean())
    ok &= check(f"fixed-K debiasing matches hand-computed average (expected~{expected_bump:.5f}, "
                f"actual={actual_bump:.5f})", abs(expected_bump - actual_bump) < 1e-4)

    # 5. K larger than population is handled (clamped, not an error)
    new_sd2, result2 = DW.run_one_round(
        global_state_dict=sd, client_ids=client_ids, K=999,
        local_train_fn=fake_local_train, C=100.0, sigma=0.0,
        rng=torch_rng, py_rng=py_rng,
    )
    ok &= check(f"K > population is clamped to population size (selected {len(result2.selected_clients)})",
                len(result2.selected_clients) == len(client_ids))

    print()
    print("ALL CHECKS PASSED" if ok else "AT LEAST ONE CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
