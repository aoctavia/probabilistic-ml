# Methods

> **Status:** skeleton — to be completed in Phase 1 (M0 + CAVI).
> Derivations are written here *before* the corresponding code is written.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| \(N\) | number of observations |
| \(D\) | number of features (single-view case) |
| \(K\) | number of latent factors |
| \(Z \in \mathbb{R}^{N \times K}\) | factor scores |
| \(W \in \mathbb{R}^{D \times K}\) | factor loadings |
| \(\tau\) | observation noise precision |

---

## M0 — Baseline (Probabilistic PCA)

### Generative model

\[
z_i \sim \mathcal{N}(0, I_K), \quad i = 1, \dots, N
\]

\[
w_j \sim \mathcal{N}(0, I_K), \quad j = 1, \dots, D
\]

\[
\tau \sim \text{Gamma}(a_0, b_0)
\]

\[
x_i \mid z_i, W, \tau \sim \mathcal{N}(W z_i,\; \tau^{-1} I_D)
\]

This is the standard probabilistic PCA model of Tipping & Bishop (1999), with an
additional conjugate prior on \(\tau\).

### CAVI updates

> *To be derived and written here in Phase 1.*

The variational family factorises as:

\[
q(Z, W, \tau) = q(Z)\, q(W)\, q(\tau)
\]

where each factor is in the same exponential family as the corresponding prior.

**Update for \(q(z_i)\):** ...

**Update for \(q(w_j)\):** ...

**Update for \(q(\tau)\):** ...

### ELBO

> *To be written in Phase 1.*

---

## M1 — Sparse loadings

> *To be written in Phase 3.*

---

## M2 — Automatic relevance determination

> *To be written in Phase 3.*

---

## M3 — Spatial structure

> *To be written in Phase 4.*

---

## M4 — Multi-view

> *To be written in Phase 5.*

---

## References

- Tipping, M. E., & Bishop, C. M. (1999). Probabilistic principal component analysis.
  *Journal of the Royal Statistical Society: Series B*, 61(3), 611–622.
