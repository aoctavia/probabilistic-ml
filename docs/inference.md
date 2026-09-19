# Inference

> **Status:** skeleton — to be completed across Phases 1–4.

---

## CAVI — Coordinate-ascent variational inference

**Assumptions:** mean-field variational family; conjugate priors on all parameters.
**Cost:** \(\mathcal{O}(N D K)\) per iteration.
**Implemented for:** M0, M1, M2.

Coordinate ascent maximises the ELBO by iterating analytic updates for each
variational factor while holding the others fixed. Each update has a closed form
when the prior and likelihood are conjugate.

**Correctness gate:** ELBO is monotonically non-decreasing across iterations
(verified in `tests/test_cavi.py`).

> *Full derivation in [Methods](methods.md).*

---

## SVI — Stochastic variational inference

**Assumptions:** reparameterisable variational family; minibatching over \(N\).
**Cost:** \(\mathcal{O}(B D K)\) per gradient step, where \(B \ll N\) is batch size.
**Implemented for:** M3, M4.

Uses JAX's `jit` + `grad` throughout. PRNG key is threaded explicitly — no global
random state.

**Correctness gate:** gradient check against finite differences on a small instance
(verified in `tests/test_svi.py`).

---

## Structured variational family

> *To be specified and implemented in Phase 4.*

At least one departure from mean-field: e.g., a low-rank-plus-diagonal Gaussian
over the factor scores, retaining within-factor correlations. The cost of the
mean-field independence assumption is measured directly by comparing this family
to the mean-field family on the same model.

---

## MCMC — NUTS reference

**Tool:** NumPyro.
**Role:** ground truth for VI on small problems. This is the one place a PPL is
the right tool — it is the reference implementation, not the contribution.

**Correctness gate:** VI posterior means converge toward NUTS posterior means as
the model is simplified and \(N\) grows (verified in `tests/test_convergence.py`).

---

## Comparison summary

| Method | Scales to large \(N\)? | Analytic? | Uncertainty quality |
|--------|----------------------|-----------|---------------------|
| CAVI | No (\(\mathcal{O}(N)\) passes) | Yes (conjugate) | Mean-field: typically overconfident |
| SVI | Yes (minibatching) | No (gradient) | Mean-field: typically overconfident |
| Structured VI | Yes | No | Better; cost measured empirically |
| NUTS | No (\(N \lesssim 10^3\)) | No | Gold standard (within model) |
