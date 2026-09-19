"""M0 — Baseline Gaussian factor model (probabilistic PCA).

This module defines the generative model only. Inference (CAVI) lives in
``sfm.inference.cavi``. See ``docs/methods.md`` for the full derivation.

Model
-----
    z_i  ~ N(0, I_K)              factor scores
    w_j  ~ N(0, I_K)              factor loadings
    tau  ~ Gamma(a0, b0)          noise precision
    x_i  | z_i, W, tau  ~ N(W z_i, tau^{-1} I_D)

This is the model of Tipping & Bishop (1999) with a conjugate Gamma prior
on the noise precision.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from sfm.types import FloatArray


@dataclass(frozen=True)
class M0Hyperparams:
    """Prior hyperparameters for the M0 model.

    Attributes
    ----------
    a0:
        Shape parameter of the Gamma prior on ``tau``. Default 1e-3
        (weakly informative).
    b0:
        Rate parameter of the Gamma prior on ``tau``. Default 1e-3.
    z_precision:
        Prior precision on each element of Z (isotropic). Default 1.0.
    w_precision:
        Prior precision on each element of W (isotropic). Default 1.0.
    """

    a0: float = 1e-3
    b0: float = 1e-3
    z_precision: float = 1.0
    w_precision: float = 1.0


def log_joint(
    X: FloatArray,
    Z: FloatArray,
    W: FloatArray,
    log_tau: float | FloatArray,
    hp: M0Hyperparams,
) -> FloatArray:
    """Unnormalised log joint log p(X, Z, W, tau).

    Useful for debugging and for MCMC reference implementations.
    Does **not** include the normalising constant of the Gaussian.

    Parameters
    ----------
    X:
        Observed data, shape ``(N, D)``.
    Z:
        Factor scores, shape ``(N, K)``.
    W:
        Factor loadings, shape ``(D, K)``.
    log_tau:
        Log noise precision (scalar).
    hp:
        Prior hyperparameters.

    Returns
    -------
    Scalar log-joint value.
    """
    N, D = X.shape
    tau = jnp.exp(log_tau)

    # log p(Z)
    lp_z = -0.5 * hp.z_precision * jnp.sum(Z**2)

    # log p(W)
    lp_w = -0.5 * hp.w_precision * jnp.sum(W**2)

    # log p(tau) — Gamma(a0, b0), parameterised as shape/rate
    lp_tau = (hp.a0 - 1.0) * log_tau - hp.b0 * tau

    # log p(X | Z, W, tau)
    residual = X - Z @ W.T  # (N, D)
    lp_x = 0.5 * N * D * log_tau - 0.5 * tau * jnp.sum(residual**2)

    return lp_z + lp_w + lp_tau + lp_x
