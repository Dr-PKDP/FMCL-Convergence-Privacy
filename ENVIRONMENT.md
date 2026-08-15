# ENVIRONMENT.md

## Track A (local execution environment)

Recorded 2026-08-14, at repo init, before any experiment ran.

- OS: Ubuntu (container), 1 vCPU, 3.9 GB RAM, ~10 GB free disk at init.
- Python: 3.12.3
- Key packages (exact versions pinned in `requirements.txt`):
  - numpy 2.4.4
  - scipy 1.17.1
  - pandas 3.0.2
  - matplotlib 3.10.8
  - torch 2.13.0 (CPU-only wheel; CUDA sub-packages were pulled in as
    declared dependencies of the PyPI wheel but there is no GPU in this
    environment — torch here is used only to write and logic-check E7's
    script against toy data, never for real training)
- Network: egress restricted to an allow-list (PyPI, npm, GitHub,
  crates.io, Ubuntu archives). Confirmed NOT reachable:
  129.106.31.39 (R770 jump server), 129.106.31.17 (R770).
- Determinism: every script fixes `numpy.random.default_rng(seed)` /
  `torch.manual_seed(seed)` explicitly per run; seeds are recorded in each
  result's metadata, never left to global RNG state.

## Track B (R770 — to be filled in on first actual run)

Per the established configuration used for the R770 (recorded here for
reference, to be confirmed current before E7 is run, not assumed):
- Host: Dell PowerEdge R770, 2× NVIDIA H200 NVL (143.8 GB VRAM each),
  1 TB RAM, 44 TB storage.
- Access: SSH via jump server 129.106.31.39 → 129.106.31.17,
  user pduttapramanik.
- Working directory convention: `/data/pduttapramanik/fmcl_paper2a/`
  (new — parallel to the existing `/data/pduttapramanik/fmcl_paper3/`,
  kept separate so the two projects' experiment logs and checkpoints
  never collide).
- Known environment fixes to carry over (established previously, must
  be re-applied here, not re-discovered):
  - `torch.set_num_threads(4)` plus OMP/MKL thread-count environment
    variables, to prevent CPU thread explosion on the shared host.
  - Heredocs for remote script launch use `<<'PYEOF'` (quoted) to prevent
    shell variable expansion; use `os.environ` inside the Python block
    instead of shell interpolation.
  - tmux for all unattended runs; no interactive back-and-forth on this
    government system.
- **PENDING**: exact torch/CUDA driver version on the R770, exact Python
  version, and confirmation that `torch.manual_seed` + cuDNN
  determinism flags are set, must be recorded here from the first
  actual run's output before any E7 number is treated as final. This
  section will be updated (not silently overwritten — see amendment
  discipline in EXPERIMENT_PLAN.md) once that happens.
