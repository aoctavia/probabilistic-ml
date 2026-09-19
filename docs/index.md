# Structured Probabilistic Factor Models

> **This page will be rewritten in Phase 6 when findings are known.**
> Fill in Section 0 of `BRIEF.md` before writing this page.

## Research question

Under what conditions can a structured Bayesian factor model recover interpretable
latent structure from high-dimensional data, and what does imposing structure cost
in terms of identifiability, calibration, and computation?

## Sub-questions

- **Q1 — Recovery.** When do structural priors help, and when do they hurt?
- **Q2 — Identifiability.** How do rotational and other invariances manifest in the
  posterior, and how does structure partially resolve them?
- **Q3 — Inference trade-off.** CAVI vs. SVI vs. MCMC: accuracy, calibration,
  and wall-clock time as dimensionality grows.

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Scaffold | ✅ complete |
| 1 | M0 + CAVI | ⬜ pending |
| 2 | MCMC reference + E3 | ⬜ pending |
| 3 | M1/M2 + E4 | ⬜ pending |
| 4 | M3 + SVI | ⬜ pending |
| 5 | M4 + E2 | ⬜ pending |
| 6 | E6 + writing | ⬜ pending |

## Navigation

- [Methods](methods.md) — full generative model and inference derivations
- [Identifiability](identifiability.md) — invariance analysis
- [Inference](inference.md) — what each method assumes and costs
- [Results](results.md) — findings, including negative results
- [Limitations](limitations.md) — what this work does not show
