# Project Brief — Structured Probabilistic Factor Models with Scalable Inference

---

## 0. Fill this in before starting a topic branch

> **My master's thesis (replace this block):**
> - Topic: _[fill in]_
> - Model / method used: _[fill in]_
> - Inference or estimation approach: _[fill in]_
> - What was probabilistic or Bayesian about it: _[fill in]_
> - What I found unsatisfying or unresolved in it: _[fill in]_
>
> **The link I want this project to make:** _one sentence — how this project
> extends, generalises, or interrogates something from the thesis._

Fill this in on your topic branch (`git checkout -b topic/<name>`), not on `main`.
See `BRANCHING.md` for the full workflow.

---

## 1. Context and audience

Research-grade portfolio artifact for a PhD application in probabilistic machine
learning and statistics (latent-variable models for structured, high-dimensional
data; applications in genomics, spatial transcriptomics, imaging).

**Methodological reasoning is the product.** Code quality is the supporting evidence.

---

## 2. Research question

Under what conditions can a structured Bayesian factor model recover interpretable
latent structure from high-dimensional data, and what does imposing structure cost
in terms of identifiability, calibration, and computation?

Sub-questions:
- **Q1 — Recovery.** When do structural priors help, and when do they hurt?
- **Q2 — Identifiability.** How do invariances manifest, and how does structure
  partially resolve them?
- **Q3 — Inference trade-off.** CAVI vs. SVI vs. MCMC as dimensionality grows.

---

## 3. The model

**M0** — Gaussian priors (probabilistic PCA baseline)
**M1** — Sparse loadings (horseshoe or spike-and-slab)
**M2** — ARD prior (automatic factor selection)
**M3** — Spatial structure (GP or CAR prior on factor scores) ← core contribution
**M4** — Multi-view (shared and view-specific factors)

---

## 4. Inference

- **CAVI** — analytic updates, conjugate parts (M0–M2)
- **SVI** — JAX + reparameterised gradients + minibatching (M3–M4)
- **Structured variational family** — at least one non-mean-field departure
- **NUTS** — NumPyro reference on small problems

Correctness gates (each is a test, not a claim):
- ELBO monotonicity for CAVI
- Gradient check for SVI objective
- VI means converge to NUTS means as model simplifies

---

## 5. Experiments

| ID | Name | Phase |
|----|------|-------|
| E1 | Recovery under known truth | 1 |
| E2 | Misspecification | 5 |
| E3 | Identifiability | 2 |
| E4 | Calibration (SBC + coverage) | 3 |
| E5 | Scaling (wall-clock, memory) | 4 |
| E6 | One real dataset | 6 |

---

## 6–11. See original brief

Full details in the original project brief (Section 6 onwards covers
repository structure, documentation standard, engineering standard,
phases, working agreement, and definition of done).
