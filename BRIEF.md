# Project Brief — Structured Probabilistic Factor Models with Scalable Inference

---

## 0. My thesis and the connection to this project

> **My master's thesis:**
> - Topic: _[fill in — your thesis topic]_
> - Model / method used: _[fill in]_
> - Inference or estimation approach: _[fill in — e.g., variational inference,
>   MCMC, MAP estimation]_
> - What was probabilistic or Bayesian about it: _[fill in — e.g., posterior
>   over parameters, uncertainty on predictions, prior on model structure]_
> - What I found unsatisfying or unresolved: my model treated observations as
>   exchangeable — it had no way to represent the spatial organisation of the
>   data. In biological data (cells in tissue, pixels in an image), *where*
>   an observation comes from is as informative as *what* it measures.
>
> **The link this project makes:** this project extends [thesis model] to
> structured high-dimensional data by placing a GP prior on the factor scores,
> so the latent representation inherits the spatial geometry of the tissue —
> the methodological gap that spatial transcriptomics makes urgent.

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
