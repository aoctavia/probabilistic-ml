# Structured Probabilistic Factor Models for Spatial Transcriptomics

> **Branch:** `topic/spatial-transcriptomics`
> **Findings section** will be written in Phase 6, once experiments are complete.

---

## Research question

Under what conditions can a spatially-structured Bayesian factor model recover
interpretable latent organisation from high-dimensional transcriptomic data,
and what does imposing spatial structure cost in terms of identifiability,
calibration, and computation?

**Sub-questions:**

- **Q1 — Recovery.** Does placing a GP prior on factor scores improve latent
  factor recovery when data has genuine spatial organisation, and when does it
  hurt when spatial assumptions are violated?
- **Q2 — Identifiability.** How do the rotational and permutation invariances
  of factor models interact with a spatial prior, and what constraints are
  needed for the posterior to be interpretable?
- **Q3 — Inference trade-off.** Mean-field CAVI, stochastic VI, and MCMC on the
  same model: how do accuracy, calibration, and wall-clock time scale as the
  number of spatial locations grows?

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold: tooling, CI, simulators, docs skeleton | ✅ |
| 1 | M0 baseline (probabilistic PCA) + CAVI + E1 recovery | ✅ |
| 2 | MCMC reference + E3 identifiability analysis | 🔄 in progress |
| 3 | M1/M2 sparse/ARD | ⬜ |
| 4 | **M3 spatial GP factor model + SVI** | ⬜ |
| 5 | M4 multi-view | ⬜ |
| 6 | E6 spatial transcriptomics (Visium) + final writing | ⬜ |

---

## Key methodological contributions

- **M3 — Spatial factor model.** A GP prior on each column of the factor score
  matrix \(Z\), indexed by spatial coordinates \(s_i\). Factor scores vary
  smoothly in space; the kernel lengthscale controls the spatial scale of
  the inferred programmes.
- **Identifiability analysis.** Empirical demonstration of how rotational
  invariance manifests in MCMC output and how the spatial prior partially
  resolves it — a question that arises directly in spatial transcriptomics,
  where factors are interpreted as gene expression programmes localised in tissue.
- **Scalable SVI.** JAX-based stochastic VI with reparameterised gradients and
  minibatching over spatial locations, making the spatial model feasible for
  tissue-scale datasets (N ~ 10^4 spots).

---

## Phase 1 results (M0 baseline)

E1 sweep over N, D, K, tau — key findings:

- Factor recovery (matched correlation) improves monotonically with D and with
  moderate N; subspace distance < 0.2 at N=1000, D=50, K=4.
- Recovery degrades gracefully as K increases — no phase transition observed
  in this regime.
- At high SNR (tau=10), CAVI does not declare convergence within 500 iterations
  despite good subspace distance (0.09). Slow mixing near the optimum at
  high precision — documented as a limitation of CAVI for the M0 model.

---

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone <repo-url>
cd probabilistic_ml
git checkout topic/spatial-transcriptomics
uv sync --all-extras
```

```bash
make test     # 28 pass, 4 skip (stubs for later phases)
make check    # lint + typecheck + test
make docs     # build MkDocs site
```

---

## Reproduce all figures

```bash
make reproduce
```

Reruns all experiments from config files and seeds, then regenerates figures.

---

## What this reimplements vs. what is original

| Component | Status |
|-----------|--------|
| M0 probabilistic PCA | Reimplementation of Tipping & Bishop (1999) |
| CAVI for M0 | Original derivation and implementation |
| M3 spatial GP factor model | Original; related to Svensson et al. (2020), Townes & Engelhardt (2023) |
| SVI for M3 | Original JAX implementation |
| Identifiability analysis | Original empirical study |
| MCMC reference | NumPyro (off-the-shelf, not a contribution) |

---

## Repository structure

```
src/sfm/
├── models/         M0–M4 model definitions
├── inference/      cavi.py, svi.py (Phase 4), mcmc.py (Phase 2)
├── diagnostics/    alignment.py, sbc.py (Phase 3), coverage.py (Phase 3)
├── simulate.py     generative simulators (M0 done; M3 spatial in Phase 4)
└── viz.py

experiments/
├── e1_recovery/    ✅ sweep N/D/K/tau, figures generated
├── e3_identifiability/   Phase 2
├── e4_calibration/       Phase 3
├── e5_scaling/           Phase 4
└── e6_real_data/         Phase 6 — 10x Genomics Visium

docs/
├── methods.md      M0 derivation complete; M3 in Phase 4
├── identifiability.md
├── inference.md
├── results.md      Phase 6
└── limitations.md  Phase 6
```
