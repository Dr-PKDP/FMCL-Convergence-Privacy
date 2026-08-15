"""
Differential-privacy and federated-aggregation mechanics for the E7
real-model feasibility experiment. Model- and dataset-independent: this
module wraps any per-client local-training step with clipping, Gaussian
noise, and debiased aggregation, matching the mechanism this project's
synthetic experiments (fedprox.py) already use and validate, translated
to operate on real PyTorch model parameters instead of numpy arrays.

Sampling convention note: the existing MIT-BIH training harness selects
a fixed-size set of K clients per round (sampling without replacement
from the available pool), not independent Bernoulli(p) participation per
client. These are different processes. For a large client population the
two are close; for MIT-BIH's small pool (on the order of 44-47 total
patients), they are not necessarily interchangeable. This module
implements the debiasing appropriate to fixed-K sampling (a plain average
over the K selected clients), matching the existing harness's actual
sampling scheme, rather than importing the (1/Np) debiasing that assumes
Bernoulli(p) participation. This distinction should be stated in any
write-up drawing on this experiment; the two participation models are not
the same and the theoretical results this project derives elsewhere
assume the Bernoulli(p) model specifically.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Sequence

import torch


def flatten_state_dict(sd: dict) -> torch.Tensor:
    return torch.cat([v.reshape(-1) for v in sd.values()])


def unflatten_to_state_dict(flat: torch.Tensor, template: dict) -> dict:
    out = {}
    idx = 0
    for k, v in template.items():
        n = v.numel()
        out[k] = flat[idx:idx + n].reshape(v.shape).clone()
        idx += n
    return out


def clip_flat(delta: torch.Tensor, C: float) -> tuple[torch.Tensor, float]:
    norm = float(torch.linalg.norm(delta))
    scale = min(1.0, C / max(norm, 1e-12))
    return delta * scale, norm


def add_noise_flat(delta: torch.Tensor, sigma: float, C: float,
                    generator: torch.Generator) -> torch.Tensor:
    if sigma <= 0:
        return delta
    noise = torch.normal(0.0, sigma * C, size=delta.shape, generator=generator)
    return delta + noise


@dataclass
class RoundResult:
    round_idx: int
    selected_clients: list
    raw_delta_norms: list
    aggregate_applied: bool


def run_one_round(
    global_state_dict: dict,
    client_ids: Sequence,
    K: int,
    local_train_fn: Callable[[dict, object], dict],
    # local_train_fn(global_state_dict, client_id) -> trained local
    # state_dict. The caller owns model instantiation entirely (loads a
    # fresh model of whatever class, copies in global_state_dict, trains
    # on client_id's data, returns the resulting state_dict) -- this is
    # the integration point run_e7_mitbih.py's TODO markers wire up.
    C: float,
    sigma: float,
    rng: torch.Generator,
    py_rng,
) -> tuple[dict, RoundResult]:
    """One federated round: sample K of client_ids without replacement,
    run local_train_fn on each, clip+noise each client's update, average
    (fixed-K debiasing), return the aggregate delta as a state_dict plus
    round diagnostics."""
    selected = py_rng.sample(list(client_ids), k=min(K, len(client_ids)))
    global_sd = {k: v.detach().clone() for k, v in global_state_dict.items()}
    global_flat = flatten_state_dict(global_sd)

    deltas = []
    raw_norms = []
    for cid in selected:
        trained_sd = local_train_fn(global_sd, cid)
        trained_flat = flatten_state_dict(trained_sd)
        delta = trained_flat - global_flat
        clipped, raw_norm = clip_flat(delta, C)
        noised = add_noise_flat(clipped, sigma, C, rng)
        deltas.append(noised)
        raw_norms.append(raw_norm)

    agg_flat = torch.stack(deltas).mean(dim=0)  # fixed-K debiasing: plain average
    new_flat = global_flat + agg_flat
    new_sd = unflatten_to_state_dict(new_flat, global_sd)

    result = RoundResult(round_idx=-1, selected_clients=selected,
                          raw_delta_norms=raw_norms, aggregate_applied=True)
    return new_sd, result
