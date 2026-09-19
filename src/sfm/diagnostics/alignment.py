"""Post-hoc alignment for factor model posteriors.

Factor models are invariant to:
  1. Rotation / orthogonal transformation
  2. Column permutation
  3. Sign flips per column

This module provides rotation-invariant metrics for comparing estimated
factor subspaces to a ground-truth subspace, plus matching utilities
for MCMC posterior alignment (Phase 2).

Reference: docs/identifiability.md
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment

from sfm.types import FloatArray


# ── Subspace distance ─────────────────────────────────────────────────────────


def subspace_distance(A: FloatArray, B: FloatArray) -> float:
    """Procrustes-aligned subspace distance between two factor matrices.

    Measures how well the column spaces of A and B agree, invariant to
    rotation, permutation, and sign flips.

    Algorithm
    ---------
    1. Orthonormalise each matrix via QR decomposition: A -> Q_A, B -> Q_B.
    2. Compute the singular values of Q_A^T Q_B.
    3. The subspace distance is sqrt(K - sum(sigma_k^2)), which is 0 when
       the subspaces are identical and sqrt(K) when they are orthogonal.

    This is the standard principal-angle distance (also called the
    Grassmann distance).

    Parameters
    ----------
    A:
        First factor matrix, shape ``(n, K)``.
    B:
        Second factor matrix, shape ``(n, K)`` with same ``n`` and ``K``.

    Returns
    -------
    float
        Subspace distance in [0, sqrt(K)].
    """
    A_np = np.asarray(A)
    B_np = np.asarray(B)

    Q_A, _ = np.linalg.qr(A_np)
    Q_B, _ = np.linalg.qr(B_np)

    # Singular values of Q_A^T Q_B — these are cosines of principal angles
    sv = np.linalg.svd(Q_A.T @ Q_B, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)  # numerical safety

    K = A_np.shape[1]
    dist = float(np.sqrt(max(0.0, K - float(np.sum(sv**2)))))
    return dist


# ── Mean correlation after optimal matching ───────────────────────────────────


def matched_correlation(
    Z_true: FloatArray,
    Z_est: FloatArray,
) -> float:
    """Mean absolute Pearson correlation after optimal column matching.

    Finds the permutation + sign assignment of Z_est columns that maximises
    mean absolute correlation with Z_true, then reports that mean.

    Parameters
    ----------
    Z_true:
        True factor scores, shape ``(N, K)``.
    Z_est:
        Estimated factor scores, shape ``(N, K)``.

    Returns
    -------
    float
        Mean absolute correlation in [0, 1].  1.0 is perfect recovery.
    """
    Z_t = np.asarray(Z_true)
    Z_e = np.asarray(Z_est)

    # Normalise columns
    Z_t = Z_t / (np.std(Z_t, axis=0, keepdims=True) + 1e-12)
    Z_e = Z_e / (np.std(Z_e, axis=0, keepdims=True) + 1e-12)

    K = Z_t.shape[1]

    # Absolute correlation matrix: cost[i, j] = |corr(Z_true[:,i], Z_est[:,j])|
    corr = np.clip(np.abs(Z_t.T @ Z_e) / Z_t.shape[0], 0.0, 1.0)

    # Hungarian algorithm: maximise total correlation = minimise negative
    row_ind, col_ind = linear_sum_assignment(-corr)

    mean_corr = float(np.mean(corr[row_ind, col_ind]))
    return mean_corr


def procrustes_align(
    Z_ref: FloatArray,
    Z_est: FloatArray,
) -> FloatArray:
    """Orthogonal Procrustes alignment of Z_est onto Z_ref.

    Finds the orthogonal matrix R that minimises ||Z_ref - Z_est R||_F
    and returns Z_est @ R.

    Parameters
    ----------
    Z_ref:
        Reference matrix, shape ``(N, K)``.
    Z_est:
        Matrix to align, shape ``(N, K)``.

    Returns
    -------
    FloatArray
        Aligned version of Z_est, shape ``(N, K)``.
    """
    Z_r = np.asarray(Z_ref)
    Z_e = np.asarray(Z_est)

    M = Z_r.T @ Z_e  # (K, K)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt  # optimal orthogonal matrix

    return jnp.array(Z_e @ R.T)
