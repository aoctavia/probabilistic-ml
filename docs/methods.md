# Methods

---

## Notation

| Symbol | Meaning |
|--------|---------|
| \(N\) | number of observations |
| \(D\) | number of features |
| \(K\) | number of latent factors |
| \(Z \in \mathbb{R}^{N \times K}\) | factor scores (rows \(z_i\)) |
| \(W \in \mathbb{R}^{D \times K}\) | factor loadings (rows \(w_j\)) |
| \(\tau\) | observation noise precision |
| \(M_Z \in \mathbb{R}^{N \times K}\) | variational means for \(Z\) |
| \(S_Z \in \mathbb{R}^{K \times K}\) | shared variational covariance for each \(z_i\) |
| \(M_W \in \mathbb{R}^{D \times K}\) | variational means for \(W\) |
| \(S_W \in \mathbb{R}^{K \times K}\) | shared variational covariance for each \(w_j\) |
| \(a_N, b_N\) | variational shape/rate for \(\tau\) |

---

## M0 — Baseline (Probabilistic PCA)

### Generative model

The model is probabilistic PCA (Tipping & Bishop, 1999) with a conjugate
Gamma prior on noise precision:

\[
z_i \sim \mathcal{N}(0,\, I_K), \quad i = 1, \dots, N
\]

\[
w_j \sim \mathcal{N}(0,\, I_K), \quad j = 1, \dots, D
\]

\[
\tau \sim \text{Gamma}(a_0, b_0) \quad \text{(shape/rate parameterisation)}
\]

\[
x_i \mid z_i, W, \tau \sim \mathcal{N}(W z_i,\; \tau^{-1} I_D)
\]

The log joint is:

\[
\log p(X, Z, W, \tau)
= \frac{ND}{2}\log\tau
- \frac{\tau}{2}\|X - ZW^\top\|_F^2
- \frac{1}{2}\|Z\|_F^2
- \frac{1}{2}\|W\|_F^2
+ (a_0 - 1)\log\tau - b_0\tau
+ \text{const}
\]

---

### CAVI — coordinate-ascent variational inference

#### Variational family

We use a fully factorised (mean-field) family:

\[
q(Z, W, \tau) = q(Z)\,q(W)\,q(\tau)
= \left[\prod_i q(z_i)\right]\!\left[\prod_j q(w_j)\right]\! q(\tau)
\]

Because the prior–likelihood pairs are conjugate, the optimal factor for each
variable is in the same exponential family as its prior:

\[
q(z_i) = \mathcal{N}(m_i,\, S_Z), \qquad
q(w_j) = \mathcal{N}(\mu_j,\, S_W), \qquad
q(\tau) = \text{Gamma}(a_N, b_N)
\]

**Shared covariance.** Because the prior on every \(z_i\) is the same isotropic
Gaussian and the likelihood is symmetric in \(i\), the optimal covariance \(S_Z\)
is identical for all \(i\) — it depends on \(i\) only through the mean \(m_i\).
The same holds for \(S_W\). This reduces storage from \(\mathcal{O}(NK^2)\) to
\(\mathcal{O}(NK + K^2)\).

#### Update for \(q(z_i)\)

Taking the expectation of the log joint over all variables except \(z_i\):

\[
\log q^*(z_i) \propto
\mathbb{E}_{-z_i}\!\left[\log p(x_i \mid z_i, W, \tau)\right]
+ \log p(z_i)
\]

Expanding and collecting terms quadratic and linear in \(z_i\):

\[
= -\frac{1}{2}\,z_i^\top
\underbrace{\!\left(\mathbb{E}[\tau]\,\mathbb{E}[W^\top W] + I_K\right)\!}_{\Lambda_Z}
z_i
+ z_i^\top
\underbrace{\mathbb{E}[\tau]\,M_W^\top x_i}_{h_i}
+ \text{const}
\]

This is Gaussian with precision \(\Lambda_Z\) and natural mean \(h_i\):

\[
\boxed{S_Z = \Lambda_Z^{-1}, \qquad m_i = S_Z\,\mathbb{E}[\tau]\,M_W^\top x_i}
\]

In matrix form (all \(N\) means in one step):

\[
M_Z = \mathbb{E}[\tau]\;X\,M_W\,S_Z
\]

The second moment of \(W\) required here is:

\[
\mathbb{E}[W^\top W]
= \sum_j \mathbb{E}[w_j w_j^\top]
= \sum_j (S_W + \mu_j\mu_j^\top)
= D\,S_W + M_W^\top M_W
\in \mathbb{R}^{K\times K}
\]

#### Update for \(q(w_j)\)

By symmetry (swapping the roles of \(Z\) and \(W\)):

\[
\boxed{S_W = \left(\mathbb{E}[\tau]\,\mathbb{E}[Z^\top Z] + I_K\right)^{-1},
\qquad
\mu_j = S_W\,\mathbb{E}[\tau]\,M_Z^\top x_j}
\]

where \(x_j \in \mathbb{R}^N\) is the \(j\)-th column of \(X\), and:

\[
M_W = \mathbb{E}[\tau]\;X^\top M_Z\,S_W
\]

\[
\mathbb{E}[Z^\top Z] = N\,S_Z + M_Z^\top M_Z \in \mathbb{R}^{K\times K}
\]

#### Update for \(q(\tau)\)

\[
\log q^*(\tau) \propto
\left(a_0 - 1 + \tfrac{ND}{2}\right)\log\tau
- \left(b_0 + \tfrac{1}{2}\,\mathcal{R}\right)\tau
\]

This is Gamma with:

\[
\boxed{a_N = a_0 + \tfrac{ND}{2}, \qquad b_N = b_0 + \tfrac{1}{2}\,\mathcal{R}}
\]

where \(\mathcal{R} = \mathbb{E}\!\left[\|X - ZW^\top\|_F^2\right]\) is the
expected reconstruction error:

\[
\mathcal{R}
= \|X\|_F^2
- 2\,\langle X,\, M_Z M_W^\top\rangle_F
+ \operatorname{tr}\!\left(\mathbb{E}[W^\top W]\;\mathbb{E}[Z^\top Z]\right)
\]

Note: \(a_N\) is constant across iterations (set once at initialisation).

The sufficient statistics of \(q(\tau)\) used in other updates are:

\[
\mathbb{E}[\tau] = \frac{a_N}{b_N}, \qquad
\mathbb{E}[\log\tau] = \psi(a_N) - \log b_N
\]

where \(\psi\) denotes the digamma function.

---

### ELBO

The evidence lower bound decomposes as:

\[
\mathcal{L}
= \underbrace{\mathbb{E}[\log p(X\mid Z,W,\tau)]}_{T_{\text{lik}}}
- \underbrace{\operatorname{KL}(q(Z)\,\|\,p(Z))}_{T_Z}
- \underbrace{\operatorname{KL}(q(W)\,\|\,p(W))}_{T_W}
- \underbrace{\operatorname{KL}(q(\tau)\,\|\,p(\tau))}_{T_\tau}
\]

**Expected log-likelihood \(T_{\text{lik}}\):**

\[
T_{\text{lik}}
= \frac{ND}{2}\!\left(\mathbb{E}[\log\tau] - \log 2\pi\right)
- \frac{\mathbb{E}[\tau]}{2}\,\mathcal{R}
\]

**KL for Gaussian factors \(T_Z\), \(T_W\):**

For \(\mathcal{N}(m, S)\) against \(\mathcal{N}(0, I_K)\):

\[
\operatorname{KL}(\mathcal{N}(m,S)\,\|\,\mathcal{N}(0,I_K))
= \tfrac{1}{2}\!\left(\operatorname{tr}(S) + m^\top m - K - \log\det S\right)
\]

Summing over all \(N\) latent score vectors:

\[
T_Z = -\frac{1}{2}\!\left(
N\operatorname{tr}(S_Z) + \|M_Z\|_F^2 - NK - N\log\det S_Z
\right)
\]

\[
T_W = -\frac{1}{2}\!\left(
D\operatorname{tr}(S_W) + \|M_W\|_F^2 - DK - D\log\det S_W
\right)
\]

**KL for noise precision \(T_\tau\):**

For \(\text{Gamma}(a_N, b_N)\) against \(\text{Gamma}(a_0, b_0)\):

\[
T_\tau = -\!\left[
(a_N - a_0)\psi(a_N)
- \log\Gamma(a_N) + \log\Gamma(a_0)
+ a_0\left(\log b_N - \log b_0\right)
+ (b_0 - b_N)\frac{a_N}{b_N}
\right]
\]

The ELBO is monotonically non-decreasing across CAVI iterations by construction:
each coordinate update maximises \(\mathcal{L}\) with respect to that factor
while holding all others fixed. This is verified in `tests/test_cavi.py`.

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
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning*. Springer.
  Chapters 10 (variational inference) and 12 (probabilistic PCA).
