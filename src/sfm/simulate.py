"""Generative simulators for experiments E1–E4.

Each simulator takes an explicit JAX PRNG key and returns a dataclass
containing the data and the ground-truth parameters used to generate it.
No hidden global random state.

Implemented so far:
    simulate_m0  — Gaussian factor model (probabilistic PCA)

Spatial and multi-view simulators will be added in Phases 4–5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp

from sfm.types import FloatArray, PRNGKey


# ── Data containers ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class M0Data:
    """Ground-truth and observed data from the M0 (baseline) simulator.

    Attributes
    ----------
    X:
        Observed data matrix, shape ``(N, D)``.
    Z_true:
        True factor scores, shape ``(N, K)``.
    W_true:
        True factor loadings, shape ``(D, K)``.
    tau_true:
        True observation noise precision (scalar).
    N, D, K:
        Dimensions for convenience.
    """

    X: FloatArray
    Z_true: FloatArray
    W_true: FloatArray
    tau_true: float
    N: int
    D: int
    K: int


# ── M0 simulator ─────────────────────────────────────────────────────────────


def simulate_m0(
    key: PRNGKey,
    *,
    N: int = 200,
    D: int = 50,
    K: int = 4,
    tau: float = 2.0,
    z_scale: float = 1.0,
    w_scale: float = 1.0,
) -> M0Data:
    """Simulate data from the M0 Gaussian factor model.

    Generative process
    ------------------
    .. math::

        z_i \\sim \\mathcal{N}(0, z_{\\text{scale}}^2 \\, I_K)

        w_j \\sim \\mathcal{N}(0, w_{\\text{scale}}^2 \\, I_K)

        x_i \\mid z_i, W, \\tau \\sim \\mathcal{N}(W z_i,\\; \\tau^{-1} I_D)

    Parameters
    ----------
    key:
        JAX PRNG key (shape ``(2,)``). Split internally; caller must manage.
    N:
        Number of observations.
    D:
        Number of features.
    K:
        Number of latent factors.
    tau:
        Observation noise precision. Higher = less noise.
    z_scale:
        Standard deviation of the factor score prior.
    w_scale:
        Standard deviation of the loading prior.

    Returns
    -------
    M0Data
        Frozen dataclass with ``X``, ``Z_true``, ``W_true``, ``tau_true``.
    """
    key_z, key_w, key_eps = jax.random.split(key, 3)

    Z = jax.random.normal(key_z, shape=(N, K)) * z_scale  # (N, K)
    W = jax.random.normal(key_w, shape=(D, K)) * w_scale  # (D, K)

    mean = Z @ W.T  # (N, D)
    eps = jax.random.normal(key_eps, shape=(N, D)) / jnp.sqrt(tau)  # (N, D)
    X = mean + eps

    return M0Data(
        X=X,
        Z_true=Z,
        W_true=W,
        tau_true=float(tau),
        N=N,
        D=D,
        K=K,
    )


# ── Convenience sweep ─────────────────────────────────────────────────────────


class SweepSpec(NamedTuple):
    """A single point in a parameter sweep for E1."""

    N: int
    D: int
    K: int
    tau: float
    seed: int


def simulate_sweep(
    specs: list[SweepSpec],
) -> list[M0Data]:
    """Simulate M0 data for every point in a parameter sweep.

    Parameters
    ----------
    specs:
        List of ``SweepSpec`` named tuples, one per sweep point.

    Returns
    -------
    list[M0Data]
        One ``M0Data`` per spec, in the same order.
    """
    results = []
    for spec in specs:
        key = jax.random.PRNGKey(spec.seed)
        data = simulate_m0(
            key,
            N=spec.N,
            D=spec.D,
            K=spec.K,
            tau=spec.tau,
        )
        results.append(data)
    return results
