"""CAVI — coordinate-ascent variational inference for the M0 baseline model.

Implements the analytic updates derived in docs/methods.md.

The variational family is fully factorised (mean-field):

    q(Z, W, tau) = q(Z) q(W) q(tau)

where
    q(z_i)  = N(m_i, S_Z)   -- shared covariance S_Z, individual means
    q(w_j)  = N(mu_j, S_W)  -- shared covariance S_W, individual means
    q(tau)  = Gamma(a_N, b_N)

The ELBO is tracked each iteration; monotonic increase is guaranteed by
construction and verified in tests/test_cavi.py.

All functions are pure (no side effects) and thread PRNG keys explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import NamedTuple

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import digamma, gammaln

from sfm.models.m0_baseline import M0Hyperparams
from sfm.types import FloatArray


# ── Variational state ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CAVIState:
    """All variational parameters for the M0 mean-field family.

    Attributes
    ----------
    M_Z:
        Posterior means for Z, shape ``(N, K)``.
    S_Z:
        Shared posterior covariance for each z_i, shape ``(K, K)``.
    M_W:
        Posterior means for W, shape ``(D, K)``.
    S_W:
        Shared posterior covariance for each w_j, shape ``(K, K)``.
    a_N:
        Variational shape for tau (scalar, constant across iterations).
    b_N:
        Variational rate for tau (scalar, updated each iteration).
    """

    M_Z: FloatArray  # (N, K)
    S_Z: FloatArray  # (K, K)
    M_W: FloatArray  # (D, K)
    S_W: FloatArray  # (K, K)
    a_N: float
    b_N: float


class CAVIResult(NamedTuple):
    """Output of ``run_cavi``."""

    state: CAVIState
    elbo_history: list[float]
    converged: bool
    n_iter: int


# ── Initialisation ────────────────────────────────────────────────────────────


def init_cavi(
    X: FloatArray,
    K: int,
    hp: M0Hyperparams,
    *,
    init_scale: float = 0.1,
    key: Array,
) -> CAVIState:
    """Initialise variational parameters.

    Parameters
    ----------
    X:
        Observed data, shape ``(N, D)``.
    K:
        Number of latent factors.
    hp:
        Prior hyperparameters.
    init_scale:
        Scale of random initialisation for M_Z and M_W.
    key:
        JAX PRNG key.

    Returns
    -------
    CAVIState
        Initial variational state.
    """
    import jax

    N, D = X.shape
    key_z, key_w = jax.random.split(key)

    M_Z = jax.random.normal(key_z, (N, K)) * init_scale
    M_W = jax.random.normal(key_w, (D, K)) * init_scale
    S_Z = jnp.eye(K)
    S_W = jnp.eye(K)

    a_N = float(hp.a0 + 0.5 * N * D)
    b_N = float(hp.b0 + 1.0)  # placeholder; will be updated in first iteration

    return CAVIState(M_Z=M_Z, S_Z=S_Z, M_W=M_W, S_W=S_W, a_N=a_N, b_N=b_N)


# ── Second-moment helpers ─────────────────────────────────────────────────────


def _second_moment_W(state: CAVIState) -> FloatArray:
    """E[W^T W] = D * S_W + M_W^T M_W,  shape (K, K)."""
    D = state.M_W.shape[0]
    return float(D) * state.S_W + state.M_W.T @ state.M_W


def _second_moment_Z(state: CAVIState) -> FloatArray:
    """E[Z^T Z] = N * S_Z + M_Z^T M_Z,  shape (K, K)."""
    N = state.M_Z.shape[0]
    return float(N) * state.S_Z + state.M_Z.T @ state.M_Z


def _expected_tau(state: CAVIState) -> float:
    """E[tau] = a_N / b_N."""
    return state.a_N / state.b_N


def _expected_log_tau(state: CAVIState) -> float:
    """E[log tau] = psi(a_N) - log(b_N)."""
    return float(digamma(jnp.array(state.a_N)) - jnp.log(jnp.array(state.b_N)))


# ── Coordinate updates ────────────────────────────────────────────────────────


def update_q_z(X: FloatArray, state: CAVIState) -> CAVIState:
    """Update q(Z): analytic Gaussian update.

    Derivation: docs/methods.md §"Update for q(z_i)".

    Returns updated state with new M_Z and S_Z.
    """
    e_tau = _expected_tau(state)
    EWtW = _second_moment_W(state)  # (K, K)

    Lambda_Z = e_tau * EWtW + jnp.eye(state.M_Z.shape[1])  # (K, K)
    S_Z = jnp.linalg.inv(Lambda_Z)  # (K, K)

    # M_Z[i,:] = e_tau * S_Z @ M_W^T @ x_i
    # Matrix form: M_Z = e_tau * X @ M_W @ S_Z
    M_Z = e_tau * X @ state.M_W @ S_Z  # (N, K)

    return replace(state, M_Z=M_Z, S_Z=S_Z)


def update_q_w(X: FloatArray, state: CAVIState) -> CAVIState:
    """Update q(W): analytic Gaussian update (symmetric to q(Z) update).

    Derivation: docs/methods.md §"Update for q(w_j)".

    Returns updated state with new M_W and S_W.
    """
    e_tau = _expected_tau(state)
    EZtZ = _second_moment_Z(state)  # (K, K)

    Lambda_W = e_tau * EZtZ + jnp.eye(state.M_W.shape[1])  # (K, K)
    S_W = jnp.linalg.inv(Lambda_W)  # (K, K)

    # M_W = e_tau * X^T @ M_Z @ S_W
    M_W = e_tau * X.T @ state.M_Z @ S_W  # (D, K)

    return replace(state, M_W=M_W, S_W=S_W)


def update_q_tau(X: FloatArray, state: CAVIState, hp: M0Hyperparams) -> CAVIState:
    """Update q(tau): analytic Gamma update.

    Derivation: docs/methods.md §"Update for q(tau)".

    Only b_N changes (a_N is constant across iterations).
    """
    R = _expected_reconstruction_error(X, state)
    b_N = float(hp.b0 + 0.5 * R)
    return replace(state, b_N=b_N)


# ── Expected reconstruction error ─────────────────────────────────────────────


def _expected_reconstruction_error(X: FloatArray, state: CAVIState) -> FloatArray:
    """E[||X - ZW^T||_F^2].

    = ||X||_F^2 - 2 <X, M_Z M_W^T>_F + tr(E[W^T W] E[Z^T Z])

    Derivation: docs/methods.md §"Update for q(tau)".
    """
    EWtW = _second_moment_W(state)  # (K, K)
    EZtZ = _second_moment_Z(state)  # (K, K)

    norm_X = jnp.sum(X**2)
    cross = 2.0 * jnp.sum(X * (state.M_Z @ state.M_W.T))
    trace_term = jnp.trace(EWtW @ EZtZ)

    return norm_X - cross + trace_term


# ── ELBO ──────────────────────────────────────────────────────────────────────


def compute_elbo(X: FloatArray, state: CAVIState, hp: M0Hyperparams) -> float:
    """Compute the ELBO.

    Decomposition:  L = T_lik - KL_Z - KL_W - KL_tau

    All terms are analytic. Derivation: docs/methods.md §"ELBO".
    """
    N, D = X.shape
    K = state.M_Z.shape[1]

    e_tau = _expected_tau(state)
    e_log_tau = _expected_log_tau(state)
    R = _expected_reconstruction_error(X, state)

    # T_lik: expected log-likelihood
    t_lik = 0.5 * N * D * (e_log_tau - jnp.log(2.0 * jnp.pi)) - 0.5 * e_tau * R

    # KL_Z: sum over N observations of KL(N(m_i, S_Z) || N(0, I_K))
    # KL(N(m,S)||N(0,I)) = 0.5*(tr(S) + m^T m - K - log det S)
    _, logdet_S_Z = jnp.linalg.slogdet(state.S_Z)
    kl_z = 0.5 * (
        N * jnp.trace(state.S_Z)
        + jnp.sum(state.M_Z**2)
        - float(N * K)
        - N * logdet_S_Z
    )

    # KL_W: sum over D loadings
    _, logdet_S_W = jnp.linalg.slogdet(state.S_W)
    kl_w = 0.5 * (
        D * jnp.trace(state.S_W)
        + jnp.sum(state.M_W**2)
        - float(D * K)
        - D * logdet_S_W
    )

    # KL_tau: KL(Gamma(a_N, b_N) || Gamma(a0, b0))
    a_N = jnp.array(state.a_N)
    b_N = jnp.array(state.b_N)
    a0 = jnp.array(hp.a0)
    b0 = jnp.array(hp.b0)

    psi_a_N = digamma(a_N)
    kl_tau = (
        (a_N - a0) * psi_a_N
        - gammaln(a_N) + gammaln(a0)
        + a0 * (jnp.log(b_N) - jnp.log(b0))
        + (b0 - b_N) * a_N / b_N
    )

    elbo = float(t_lik - kl_z - kl_w - kl_tau)
    return elbo


# ── Main loop ─────────────────────────────────────────────────────────────────


def run_cavi(
    X: FloatArray,
    K: int,
    hp: M0Hyperparams,
    *,
    key: Array,
    max_iter: int = 500,
    tol: float = 1e-4,
    init_scale: float = 0.1,
) -> CAVIResult:
    """Run coordinate-ascent VI on the M0 model.

    Parameters
    ----------
    X:
        Observed data, shape ``(N, D)``.
    K:
        Number of latent factors.
    hp:
        Prior hyperparameters.
    key:
        JAX PRNG key for initialisation.
    max_iter:
        Maximum number of full-pass iterations.
    tol:
        Convergence tolerance on the absolute ELBO change.
    init_scale:
        Scale of random initialisation.

    Returns
    -------
    CAVIResult
        Final state, ELBO history, convergence flag, and iteration count.
    """
    state = init_cavi(X, K, hp, init_scale=init_scale, key=key)
    elbo_history: list[float] = []

    for i in range(max_iter):
        state = update_q_z(X, state)
        state = update_q_w(X, state)
        state = update_q_tau(X, state, hp)

        elbo = compute_elbo(X, state, hp)
        elbo_history.append(elbo)

        if i > 0 and abs(elbo_history[-1] - elbo_history[-2]) < tol:
            return CAVIResult(
                state=state,
                elbo_history=elbo_history,
                converged=True,
                n_iter=i + 1,
            )

    return CAVIResult(
        state=state,
        elbo_history=elbo_history,
        converged=False,
        n_iter=max_iter,
    )
