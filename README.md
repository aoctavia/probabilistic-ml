# Structured Probabilistic Factor Models

> **This README will be rewritten in Phase 6 with actual findings.**
> See `BRANCHING.md` for how to customise this for a specific PhD application.
> Fill in `BRIEF.md` Section 0 before writing the introduction.

---

## Research question

Under what conditions can a structured Bayesian factor model recover interpretable
latent structure from high-dimensional data, and what does imposing structure cost
in terms of identifiability, calibration, and computation?

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold: tooling, CI, simulators, docs skeleton | ✅ complete |
| 1 | M0 baseline + CAVI inference + E1 recovery | ⬜ |
| 2 | MCMC reference + E3 identifiability | ⬜ |
| 3 | M1/M2 sparse/ARD + E4 calibration | ⬜ |
| 4 | M3 spatial + SVI + E5 scaling | ⬜ |
| 5 | M4 multi-view + E2 misspecification | ⬜ |
| 6 | E6 real data + final writing | ⬜ |

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd probabilistic_ml
uv sync --all-extras
```

Run tests:

```bash
make test
```

Lint and type-check:

```bash
make check
```

Build docs:

```bash
make docs
make serve-docs  # serve at http://127.0.0.1:8000
```

---

## Repository structure

```
src/sfm/
├── models/         M0–M4 model definitions
├── inference/      CAVI, SVI, MCMC
├── diagnostics/    alignment, SBC, coverage
├── simulate.py     generative simulators
└── viz.py          shared plotting utilities

experiments/
├── e1_recovery/    phase diagram: recovery vs. N, D, K, tau
├── e2_misspecification/
├── e3_identifiability/
├── e4_calibration/
├── e5_scaling/
└── e6_real_data/

docs/
├── methods.md      full derivations
├── identifiability.md
├── inference.md
├── results.md      findings (Phase 6)
└── limitations.md
```

---

## Reproduce all figures

```bash
make reproduce
```

This runs all experiments in order (E1–E5) and regenerates all figures from
the stored results. Each experiment is driven by a config file and a seed.

---

## Topic branches

This scaffold is designed to be adapted for specific PhD applications. See
[BRANCHING.md](BRANCHING.md) for the strategy and available topic branches.

---

## What this reimplements

- **M0** — probabilistic PCA (Tipping & Bishop, 1999)
- **M1/M2** — horseshoe-regularised factor model, ARD prior
- **M3** — spatial factor model with GP prior on scores
- **M4** — multi-view factor model (MOFA-style, Argelaguet et al., 2018)

The inference implementations (CAVI, JAX SVI) are original.
The MCMC reference uses NumPyro as an off-the-shelf tool.
