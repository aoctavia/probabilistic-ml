"""Generative simulators for experiments E1–E4.

Each simulator takes an explicit JAX PRNG key and returns a dataclass
containing the data and the ground-truth parameters used to generate it.
No hidden global random state.

Implemented:
    simulate_m0  — Gaussian factor model (probabilistic PCA)
    simulate_m3  — Spatial GP factor model
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


# ── M3 simulator ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class M3Data:
    """Ground-truth and observed data from the M3 (spatial) simulator.

    Attributes
    ----------
    X:       Observed data matrix, shape ``(N, D)``.
    Z_true:  True factor scores, shape ``(N, K)``. Each column drawn from GP.
    W_true:  True factor loadings, shape ``(D, K)``.
    tau_true: True noise precision.
    coords:  Spatial coordinates, shape ``(N, 2)``.
    amplitudes: True GP amplitudes per factor, shape ``(K,)``.
    lengthscales: True GP lengthscales per factor, shape ``(K,)``.
    N, D, K: Dimensions.
    """

    X: FloatArray
    Z_true: FloatArray
    W_true: FloatArray
    tau_true: float
    coords: FloatArray
    amplitudes: FloatArray
    lengthscales: FloatArray
    N: int
    D: int
    K: int


def simulate_m3(
    key: PRNGKey,
    *,
    N: int = 200,
    D: int = 50,
    K: int = 3,
    tau: float = 2.0,
    w_scale: float = 1.0,
    amplitudes: tuple[float, ...] | None = None,
    lengthscales: tuple[float, ...] | None = None,
    grid_side: int | None = None,
) -> M3Data:
    """Simulate data from the M3 spatial GP factor model.

    Spatial coordinates are placed on a regular 2-D grid (or randomly in
    [0, 1]^2 if N is not a perfect square and grid_side is not given).

    Parameters
    ----------
    key:
        JAX PRNG key.
    N:
        Number of spatial locations.
    D:
        Number of features.
    K:
        Number of latent factors.
    tau:
        Observation noise precision.
    w_scale:
        Standard deviation of loading prior.
    amplitudes:
        GP amplitude per factor (length K). Default: all 1.0.
    lengthscales:
        GP lengthscale per factor (length K). Default: all 0.3.
    grid_side:
        If given, places N = grid_side^2 points on a regular grid.
    """
    from sfm.models.m3_spatial import rbf_kernel

    if amplitudes is None:
        amplitudes = tuple(1.0 for _ in range(K))
    if lengthscales is None:
        lengthscales = tuple(0.3 for _ in range(K))

    key_coords, key_w, key_eps, *keys_z = jax.random.split(key, K + 3)

    # Spatial coordinates
    if grid_side is not None:
        N = grid_side * grid_side
        lin = jnp.linspace(0.0, 1.0, grid_side)
        gx, gy = jnp.meshgrid(lin, lin)
        coords = jnp.stack([gx.ravel(), gy.ravel()], axis=-1)  # (N, 2)
    else:
        coords = jax.random.uniform(key_coords, (N, 2))

    # Factor scores: each column drawn from its GP
    Z_cols = []
    for k in range(K):
        K_k = rbf_kernel(coords, jnp.array(amplitudes[k]), jnp.array(lengthscales[k]))
        L_k = jnp.linalg.cholesky(K_k)
        eps_k = jax.random.normal(keys_z[k], (N,))
        z_k = L_k @ eps_k  # (N,)
        Z_cols.append(z_k)
    Z = jnp.stack(Z_cols, axis=1)  # (N, K)

    W = jax.random.normal(key_w, (D, K)) * w_scale
    eps = jax.random.normal(key_eps, (N, D)) / jnp.sqrt(tau)
    X = Z @ W.T + eps

    return M3Data(
        X=X,
        Z_true=Z,
        W_true=W,
        tau_true=float(tau),
        coords=coords,
        amplitudes=jnp.array(amplitudes),
        lengthscales=jnp.array(lengthscales),
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
