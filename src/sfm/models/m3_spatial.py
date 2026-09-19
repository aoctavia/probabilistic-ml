"""M3 — Spatial factor model with GP prior on factor scores.

Each observation i carries a 2-D spatial coordinate s_i. Factor scores
Z are drawn from independent GPs (one per factor), so spatially nearby
observations have correlated latent representations.

Generative model
----------------
    z_{.k} | theta_k  ~ GP(0, kappa_{theta_k})   k = 1..K
    w_j               ~ N(0, I_K)
    tau               ~ Gamma(a0, b0)
    x_i | z_i, W, tau ~ N(W z_i, tau^{-1} I_D)

where kappa is the squared-exponential (RBF) kernel with per-factor
amplitude alpha_k and lengthscale ell_k.

Inference is handled by SVI (src/sfm/inference/svi.py): the GP prior is
non-conjugate with the Gaussian likelihood for Z once W is integrated, so
analytic CAVI updates are not available for q(Z).

See docs/methods.md §M3 for the full derivation and ELBO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp

from sfm.types import FloatArray, PRNGKey


# ── Kernel ────────────────────────────────────────────────────────────────────


def rbf_kernel(
    coords: FloatArray,
    amplitude: FloatArray,
    lengthscale: FloatArray,
    *,
    jitter: float = 1e-6,
) -> FloatArray:
    """Squared-exponential (RBF) kernel matrix.

    K_ij = amplitude^2 * exp(-||s_i - s_j||^2 / (2 * lengthscale^2))

    Parameters
    ----------
    coords:
        Spatial coordinates, shape ``(N, 2)``.
    amplitude:
        Kernel amplitude (scalar > 0).
    lengthscale:
        Kernel lengthscale (scalar > 0).
    jitter:
        Small diagonal added for numerical stability.

    Returns
    -------
    FloatArray
        Gram matrix, shape ``(N, N)``.
    """
    # Squared pairwise distances: (N, N)
    diff = coords[:, None, :] - coords[None, :, :]  # (N, N, 2)
    sq_dist = jnp.sum(diff**2, axis=-1)             # (N, N)

    K = amplitude**2 * jnp.exp(-0.5 * sq_dist / lengthscale**2)
    K = K + jitter * jnp.eye(K.shape[0])
    return K


# ── Hyperparameters ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class M3Hyperparams:
    """Prior hyperparameters for M3.

    Attributes
    ----------
    a0, b0:
        Gamma prior on tau (shape / rate).
    w_precision:
        Isotropic prior precision on loadings W.
    init_amplitude:
        Initial value for GP amplitude (one per factor at init).
    init_lengthscale:
        Initial value for GP lengthscale (one per factor at init).
        Expressed in the same units as the spatial coordinates.
    """

    a0: float = 1e-3
    b0: float = 1e-3
    w_precision: float = 1.0
    init_amplitude: float = 1.0
    init_lengthscale: float = 1.0


# ── Variational state ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SVIStateM3:
    """Variational parameters for M3 mean-field SVI.

    Attributes
    ----------
    m_Z:
        Variational means for Z, shape ``(N, K)``.
        m_Z[:, k] is the mean for factor k's score vector.
    log_v_Z:
        Log of variational variances for Z, shape ``(N, K)``.
        Using log parameterisation to enforce positivity.
    M_W:
        Variational means for W, shape ``(D, K)``.
    log_S_W_diag:
        Log of variational variances for w_j (diagonal), shape ``(K,)``.
        Shared across j (same structure as M0).
    log_a_N:
        Log of variational shape for tau (scalar).
    log_b_N:
        Log of variational rate for tau (scalar).
    log_amplitudes:
        Log GP amplitudes, shape ``(K,)``.
    log_lengthscales:
        Log GP lengthscales, shape ``(K,)``.
    """

    m_Z: FloatArray          # (N, K)
    log_v_Z: FloatArray      # (N, K)
    M_W: FloatArray          # (D, K)
    log_S_W_diag: FloatArray # (K,)
    log_a_N: float
    log_b_N: float
    log_amplitudes: FloatArray    # (K,)
    log_lengthscales: FloatArray  # (K,)


def init_svi_m3(
    X: FloatArray,
    coords: FloatArray,
    K: int,
    hp: M3Hyperparams,
    key: PRNGKey,
    *,
    init_scale: float = 0.1,
) -> SVIStateM3:
    """Initialise variational parameters for M3.

    Parameters
    ----------
    X:
        Observed data, shape ``(N, D)``.
    coords:
        Spatial coordinates, shape ``(N, 2)``.
    K:
        Number of latent factors.
    hp:
        Prior hyperparameters.
    key:
        JAX PRNG key.
    init_scale:
        Scale for random initialisation of means.
    """
    N, D = X.shape
    key_z, key_w = jax.random.split(key)

    m_Z = jax.random.normal(key_z, (N, K)) * init_scale
    log_v_Z = jnp.zeros((N, K))  # init variance = 1

    M_W = jax.random.normal(key_w, (D, K)) * init_scale
    log_S_W_diag = jnp.zeros((K,))  # init variance = 1

    log_a_N = jnp.log(jnp.array(hp.a0 + 0.5 * N * D))
    log_b_N = jnp.log(jnp.array(hp.b0 + 1.0))

    log_amplitudes = jnp.full((K,), jnp.log(jnp.array(hp.init_amplitude)))
    log_lengthscales = jnp.full((K,), jnp.log(jnp.array(hp.init_lengthscale)))

    return SVIStateM3(
        m_Z=m_Z,
        log_v_Z=log_v_Z,
        M_W=M_W,
        log_S_W_diag=log_S_W_diag,
        log_a_N=float(log_a_N),
        log_b_N=float(log_b_N),
        log_amplitudes=log_amplitudes,
        log_lengthscales=log_lengthscales,
    )


# ── ELBO ──────────────────────────────────────────────────────────────────────


def elbo_m3(
    X: FloatArray,
    coords: FloatArray,
    state: SVIStateM3,
    hp: M3Hyperparams,
) -> FloatArray:
    """Compute the M3 ELBO (analytic, no Monte Carlo).

    Decomposition: E[log p(X|Z,W,tau)] - KL_Z_GP - KL_W - KL_tau

    KL_Z_GP: sum over K factors of KL(N(m_k, diag(v_k)) || N(0, K_k))
    All terms are analytic given the GP Gram matrices.

    Parameters
    ----------
    X:
        Observed data, shape ``(N, D)``.
    coords:
        Spatial coordinates, shape ``(N, 2)``.
    state:
        Current variational state.
    hp:
        Prior hyperparameters.

    Returns
    -------
    FloatArray
        Scalar ELBO value.
    """
    N, D = X.shape
    K = state.m_Z.shape[1]

    amplitudes = jnp.exp(state.log_amplitudes)        # (K,)
    lengthscales = jnp.exp(state.log_lengthscales)    # (K,)
    v_Z = jnp.exp(state.log_v_Z)                      # (N, K)
    S_W_diag = jnp.exp(state.log_S_W_diag)            # (K,)
    a_N = jnp.exp(jnp.array(state.log_a_N))
    b_N = jnp.exp(jnp.array(state.log_b_N))

    e_tau = a_N / b_N
    e_log_tau = jax.scipy.special.digamma(a_N) - jnp.log(b_N)

    # ── Expected log-likelihood ───────────────────────────────────────────────
    # E[||X - ZW^T||_F^2] with diagonal variational covariances
    #
    # E[Z^T Z]_kk' = m_Z[:,k]^T m_Z[:,k'] + delta_{kk'} sum_i v_Z[i,k]
    # E[W^T W]_kk' = M_W[:,k]^T M_W[:,k'] + delta_{kk'} D * S_W_diag[k]

    EZtZ = state.m_Z.T @ state.m_Z + jnp.diag(jnp.sum(v_Z, axis=0))  # (K,K)
    EWtW = state.M_W.T @ state.M_W + jnp.diag(D * S_W_diag)           # (K,K)

    norm_X = jnp.sum(X**2)
    cross = 2.0 * jnp.sum(X * (state.m_Z @ state.M_W.T))
    trace_term = jnp.trace(EWtW @ EZtZ)
    R = norm_X - cross + trace_term  # expected reconstruction error

    t_lik = 0.5 * N * D * (e_log_tau - jnp.log(2.0 * jnp.pi)) - 0.5 * e_tau * R

    # ── KL for Z: sum over K GP factors ──────────────────────────────────────
    # KL(N(m_k, diag(v_k)) || N(0, K_k))
    # = 0.5 * (tr(K_k^{-1} diag(v_k)) + m_k^T K_k^{-1} m_k
    #          - N + log det K_k - sum log v_k[i])
    kl_z = jnp.array(0.0)
    for k in range(K):
        K_k = rbf_kernel(coords, amplitudes[k], lengthscales[k])
        L_k = jnp.linalg.cholesky(K_k)  # (N, N) lower triangular

        # tr(K_k^{-1} diag(v_k)) = sum_i v_k[i] * (K_k^{-1})_{ii}
        # Efficient: solve L_k @ alpha = I, then diag(alpha^T alpha)
        K_inv_diag = jnp.sum(jax.scipy.linalg.solve_triangular(
            L_k, jnp.eye(N), lower=True
        )**2, axis=0)  # (N,)

        # m_k^T K_k^{-1} m_k via Cholesky solve
        alpha_k = jax.scipy.linalg.solve_triangular(L_k, state.m_Z[:, k], lower=True)
        quad = jnp.dot(alpha_k, alpha_k)

        log_det_K = 2.0 * jnp.sum(jnp.log(jnp.diag(L_k)))
        log_det_q = jnp.sum(state.log_v_Z[:, k])

        kl_k = 0.5 * (
            jnp.dot(K_inv_diag, v_Z[:, k])
            + quad
            - N
            + log_det_K
            - log_det_q
        )
        kl_z = kl_z + kl_k

    # ── KL for W (diagonal Gaussian vs N(0, I_K)) ────────────────────────────
    # Each w_j ~ N(mu_j, diag(S_W_diag)); prior N(0, I_K)
    # KL summed over D loadings:
    kl_w = 0.5 * D * (
        jnp.sum(S_W_diag)
        + jnp.sum(state.M_W**2) / D  # sum_j mu_j^T mu_j / D * D
        - K
        - jnp.sum(jnp.log(S_W_diag))
    )
    # Correct form: KL summed over j
    kl_w = 0.5 * (
        D * jnp.sum(S_W_diag)
        + jnp.sum(state.M_W**2)
        - float(D * K)
        - D * jnp.sum(jnp.log(S_W_diag))
    )

    # ── KL for tau ────────────────────────────────────────────────────────────
    a0 = jnp.array(hp.a0)
    b0 = jnp.array(hp.b0)
    psi_a_N = jax.scipy.special.digamma(a_N)
    kl_tau = (
        (a_N - a0) * psi_a_N
        - jax.scipy.special.gammaln(a_N) + jax.scipy.special.gammaln(a0)
        + a0 * (jnp.log(b_N) - jnp.log(b0))
        + (b0 - b_N) * a_N / b_N
    )

    return t_lik - kl_z - kl_w - kl_tau
