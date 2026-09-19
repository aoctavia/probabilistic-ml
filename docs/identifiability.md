# Identifiability

> **Status:** skeleton — to be completed in Phase 2.

---

## The three invariances in factor models

A factor model \(X \approx Z W^\top\) is invariant to any invertible matrix
\(R \in \mathbb{R}^{K \times K}\):

\[
Z W^\top = (Z R^{-1})(W R^\top)^\top
\]

This creates three classes of non-identifiability:

1. **Rotational invariance.** Any orthogonal \(R\) leaves the likelihood unchanged.
2. **Sign flips.** Each column of \(Z\) can be negated if the corresponding column
   of \(W\) is also negated.
3. **Column permutation.** Factors can be reordered.

### What this means for posteriors

Under an uninformative prior, the posterior over \((Z, W)\) is a union of equivalent
modes related by these transformations. Naive posterior means over MCMC samples
destroy the signal — each sample may represent a different rotation.

### How structural priors partially resolve identifiability

> *To be empirically verified in E3 and documented here in Phase 2.*

- **Sparsity priors** on \(W\) break rotational invariance: there is no rotation
  that simultaneously keeps all columns sparse.
- **Ordered factors** (by decreasing variance) break permutation invariance.
- **Sign constraints** (e.g., positive loading for the first feature in each factor)
  break sign-flip invariance.

None of these fully resolve the posterior geometry — they narrow the invariance
group but do not eliminate it.

---

## Post-hoc alignment

When working with MCMC samples, alignment is required before computing posterior
summaries. The standard approach:

1. Choose a reference sample (e.g., the posterior mode).
2. For each other sample, find the permutation + sign assignment minimising
   Frobenius distance to the reference (solved by the Hungarian algorithm).
3. Optionally: orthogonal Procrustes alignment for the continuous rotational
   component.

> *Implementation: `src/sfm/diagnostics/alignment.py` (Phase 2).*

---

## Empirical demonstration

> *Figures from E3 will be embedded here in Phase 2.*
