#!/usr/bin/env python3
"""
E7 -- real-model feasibility sweep on MIT-BIH ResNet.

Tests whether Theorem 1's qualitative prediction (lower participation
K -> higher convergence floor; higher privacy noise sigma -> higher
floor) holds on a real, already-validated nonconvex model, rather than
only the synthetic multinomial-logistic setting used elsewhere in this
project. MIT-BIH ResNet was chosen as the target because it is the one
domain in the existing pipeline with a decisive, well-powered
convergence result under favorable conditions (8 seeds, statistically
significant) -- adding participation dropout and DP noise on top of an
already-validated baseline isolates what this experiment is actually
testing, rather than confounding "does dropout/privacy break it" with
"does this model/data combination work at all."

INTEGRATION REQUIRED before running -- search this file for "TODO(confirm)"
and fill in or verify each one against the actual codebase on this
machine. None of these were directly inspected while writing this script;
each is based on file/function/argument names referenced in prior work on
this model, and should be checked, not assumed.

Population note: MIT-BIH's total client pool is small (on the order of
44-47 patients, not thousands). K (clients selected per round) should be
set accordingly -- the existing pipeline's own guidance was K in roughly
4-10 for this domain, not the K=50 used for the larger PTB-XL/FEMNIST
pools. The sweep below reflects that.

Sampling model note: this script samples a fixed-size K clients per
round without replacement, matching the existing pipeline's own
sampling scheme. This is a different process from the independent
Bernoulli(p) participation this project's theoretical results (Theorem
1, Theorem 1b) are stated for. For a small pool, the two are not simply
interchangeable -- state this explicitly in any write-up rather than
describing K/pool_size as "the participation rate p."
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from pathlib import Path

# --- CPU thread cap: set before importing torch. Verified fix from prior
# work on this same codebase (unbounded PyTorch/OpenMP/MKL threading
# claimed 100+ cores per process on this shared, usage-tracked server).
# Also set at the shell level via OMP_NUM_THREADS/MKL_NUM_THREADS in the
# launch command below -- torch.set_num_threads alone does not bind
# every underlying math library.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import torch  # noqa: E402
torch.set_num_threads(4)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from e7_neural_feasibility import dp_wrapper as DW  # noqa: E402

# ---------------------------------------------------------------------
# TODO(confirm): exact import path for the MIT-BIH ResNet model class
# and the MIT-BIH data loader, as used in the existing pipeline. Prior
# reference points: a model class referred to as "ResNet" for MIT-BIH
# (347,973 parameters at one point in development -- confirm current
# figure), a data loader module referred to as data_mitbih.py, and a
# training harness referred to as fl_harness.py / run_mitbih_study.py.
# Replace the two lines below with the actual imports once confirmed.
# ---------------------------------------------------------------------
try:
    from models import ResNet1D_MITBIH as MITBIH_MODEL_CLASS  # TODO(confirm)
    from data_mitbih import load_mitbih_clients  # TODO(confirm)
    INTEGRATION_READY = True
except ImportError:
    MITBIH_MODEL_CLASS = None
    load_mitbih_clients = None
    INTEGRATION_READY = False


OUT_DIR = Path(__file__).resolve().parents[2] / "results" / "e7"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_model():
    """TODO(confirm): constructor arguments for the MIT-BIH ResNet class
    (input length, number of classes, channel count). Placeholder args
    below are guesses and MUST be checked against the actual class
    signature before running."""
    if not INTEGRATION_READY:
        raise RuntimeError(
            "Model/data imports not resolved -- see TODO(confirm) markers "
            "at the top of this file. Do not run until these are fixed."
        )
    return MITBIH_MODEL_CLASS()  # TODO(confirm): constructor args


def local_train_step(global_state_dict, client_id, local_epochs, lr, device):
    """TODO(confirm): this reimplements a local-training step matching
    the existing pipeline's own per-client training loop (loss function,
    optimizer, number of local epochs/batches, data loading for a single
    client). The version below is a reasonable default (cross-entropy,
    SGD, `local_epochs` full passes over the client's data) but should
    be checked against fl_harness.py's actual local-update logic before
    trusting results -- if the existing pipeline uses a different loss,
    optimizer, or per-client batching scheme, results here will not be
    comparable to the established 8-seed baseline."""
    model = build_model()
    model.load_state_dict(global_state_dict)
    model.to(device)
    model.train()

    client_data = load_mitbih_clients()[client_id]  # TODO(confirm): exact accessor
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for xb, yb in client_data:  # TODO(confirm): iterable of (x,y) batches
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


def evaluate_global_model(state_dict, held_out_data, device):
    """TODO(confirm): evaluation matching the existing pipeline's own
    metric (likely accuracy on a held-out split per fl_harness.py /
    diagnose_trajectory.py conventions)."""
    model = build_model()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in held_out_data:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(dim=1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(total, 1)


def estimate_L_and_C(global_state_dict, client_ids, local_epochs, lr, device, n_probe=5):
    """Empirical smoothness (L) and clip-norm (C) calibration, matching
    this project's own calibration methodology (results/e3/calibration/)
    rather than assuming values. C is set from the observed distribution
    of raw (pre-clip) local update norms across a probe set of clients
    (median, not max -- matches this project's own finding that setting C
    near the median leaves most updates clipped, which is itself worth
    checking for this model rather than assumed away). L is estimated via
    gradient-Lipschitz probing near the current global state.

    Run this once before the main sweep, not per (K, sigma) cell.
    """
    raw_norms = []
    for cid in random.sample(list(client_ids), k=min(n_probe, len(client_ids))):
        trained = local_train_step(global_state_dict, cid, local_epochs, lr, device)
        delta = DW.flatten_state_dict(trained) - DW.flatten_state_dict(global_state_dict)
        raw_norms.append(float(torch.linalg.norm(delta)))
    raw_norms.sort()
    C_estimate = raw_norms[len(raw_norms) // 2]  # median
    return dict(C_estimate=C_estimate, raw_norms_observed=raw_norms)


def sigma_max_sq(N, d, eps_acc, p_eff, G=1.0, C=1.0):
    """Lemma 4's noise ceiling, using p_eff = K/N as the fixed-K analogue
    of the Bernoulli participation rate -- an approximation, not an exact
    match (see sampling-model note in the module docstring)."""
    return max(0.0, (N * p_eff * eps_acc / 2 - (1 - p_eff) * G ** 2) / (d * C ** 2))


def run_sweep(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not INTEGRATION_READY:
        print("INTEGRATION NOT READY -- resolve TODO(confirm) markers before running.")
        sys.exit(1)

    clients = load_mitbih_clients()  # TODO(confirm)
    client_ids = list(clients.keys()) if hasattr(clients, "keys") else list(range(len(clients)))
    N = len(client_ids)
    print(f"Client pool size N={N}")

    model0 = build_model()
    global_sd0 = {k: v.detach().clone() for k, v in model0.state_dict().items()}
    d = sum(v.numel() for v in global_sd0.values())
    print(f"Model dimension d={d} parameters")

    calib = estimate_L_and_C(global_sd0, client_ids, args.local_epochs, args.lr, device)
    C = calib["C_estimate"]
    print(f"Calibrated C (median raw update norm) = {C:.4f}; "
          f"raw norms observed: {calib['raw_norms_observed']}")

    K_grid = args.K_grid
    sigma_fracs = [0.0, 0.5, 1.0]  # 0, sigma_max/2, sigma_max

    results_csv = OUT_DIR / "e7_mitbih_results.csv"
    existing_keys = set()
    if results_csv.exists():
        with open(results_csv) as f:
            for row in csv.DictReader(f):
                existing_keys.add((row["K"], row["sigma_frac"], row["seed"]))

    fieldnames = ["K", "sigma_frac", "sigma", "seed", "round", "accuracy",
                  "wall_clock_seconds", "N", "d", "C"]
    write_header = not results_csv.exists()
    with open(results_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for K in K_grid:
            p_eff = K / N
            s2max = sigma_max_sq(N, d, args.eps_acc, p_eff)
            sigma_max = s2max ** 0.5 if s2max > 0 else 0.0

            for sigma_frac in sigma_fracs:
                sigma = sigma_max * sigma_frac
                for seed in range(args.n_seeds):
                    key = (str(K), str(sigma_frac), str(seed))
                    if key in existing_keys:
                        print(f"[skip] K={K} sigma_frac={sigma_frac} seed={seed} (already done)")
                        continue

                    ckpt_path = OUT_DIR / f"ckpt_K{K}_sf{sigma_frac}_seed{seed}.pt"
                    start_round = 0
                    global_sd = {k: v.clone() for k, v in global_sd0.items()}
                    if ckpt_path.exists():
                        ckpt = torch.load(ckpt_path)
                        global_sd = ckpt["state_dict"]
                        start_round = ckpt["round"]
                        print(f"[resume] K={K} sigma_frac={sigma_frac} seed={seed} "
                              f"from round {start_round}")

                    random.seed(1000 * seed + K)
                    torch_rng = torch.Generator().manual_seed(2000 * seed + K)
                    py_rng = random.Random(3000 * seed + K)

                    t0 = time.time()
                    for r in range(start_round, args.rounds):
                        global_sd, round_info = DW.run_one_round(
                            global_state_dict=global_sd, client_ids=client_ids, K=K,
                            local_train_fn=lambda sd, cid: local_train_step(
                                sd, cid, args.local_epochs, args.lr, device),
                            C=C, sigma=sigma, rng=torch_rng, py_rng=py_rng,
                        )
                        if (r + 1) % args.checkpoint_every == 0:
                            torch.save({"state_dict": global_sd, "round": r + 1}, ckpt_path)

                    wall = time.time() - t0
                    acc = evaluate_global_model(global_sd, clients, device)  # TODO(confirm): held-out split

                    writer.writerow(dict(K=K, sigma_frac=sigma_frac, sigma=sigma, seed=seed,
                                          round=args.rounds, accuracy=acc,
                                          wall_clock_seconds=wall, N=N, d=d, C=C))
                    f.flush()
                    if ckpt_path.exists():
                        ckpt_path.unlink()
                    print(f"[done] K={K} sigma_frac={sigma_frac} seed={seed} "
                          f"acc={acc:.4f} wall={wall:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K_grid", type=int, nargs="+", default=[5, 10, 20],
                     help="Clients selected per round -- sized for MIT-BIH's small pool "
                          "(~44-47 total); do not reuse the K=50 convention from PTB-XL/FEMNIST.")
    ap.add_argument("--n_seeds", type=int, default=2,
                     help="Reduced from the established 8-seed MIT-BIH standard for budget "
                          "reasons -- stated explicitly, not silently under-powered.")
    ap.add_argument("--rounds", type=int, default=150,
                     help="Starting point matching this project's own synthetic-experiment "
                          "convention (T_MAX=150, found sufficient there); NOT validated for "
                          "this real model -- run a short pilot first to check floor "
                          "stabilization before trusting a full sweep at this round count.")
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=0.001,
                     help="Matches the learning rate found stable for MIT-BIH ResNet in prior "
                          "work on this codebase -- confirm this is still current.")
    ap.add_argument("--eps_acc", type=float, default=0.5)
    ap.add_argument("--checkpoint_every", type=int, default=25)
    args = ap.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
