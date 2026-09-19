"""Tests for sfm.diagnostics.alignment."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfm.diagnostics.alignment import (
    matched_correlation,
    procrustes_align,
    subspace_distance,
)


class TestSubspaceDistance:
    def test_identical_subspaces(self) -> None:
        rng = np.random.default_rng(0)
        A = rng.standard_normal((100, 3))
        assert subspace_distance(A, A) == pytest.approx(0.0, abs=1e-6)

    def test_orthogonal_subspaces(self) -> None:
        # Two orthogonal subspaces of R^6 have distance sqrt(K) = sqrt(3)
        A = np.eye(6)[:, :3]
        B = np.eye(6)[:, 3:]
        dist = subspace_distance(A, B)
        assert dist == pytest.approx(np.sqrt(3), abs=1e-5)

    def test_rotation_invariant(self) -> None:
        """Rotating A should not change its subspace distance to B."""
        rng = np.random.default_rng(1)
        A = rng.standard_normal((50, 3))
        B = rng.standard_normal((50, 3))
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))

        dist_original = subspace_distance(A, B)
        dist_rotated = subspace_distance(A @ Q, B)
        assert dist_original == pytest.approx(dist_rotated, abs=1e-5)

    def test_range(self) -> None:
        rng = np.random.default_rng(2)
        A = rng.standard_normal((50, 4))
        B = rng.standard_normal((50, 4))
        dist = subspace_distance(A, B)
        assert 0.0 <= dist <= np.sqrt(4) + 1e-6


class TestMatchedCorrelation:
    def test_perfect_recovery(self) -> None:
        rng = np.random.default_rng(3)
        Z = rng.standard_normal((100, 3))
        # Same columns but permuted and sign-flipped
        Z_est = Z[:, [2, 0, 1]] * np.array([-1, 1, -1])
        corr = matched_correlation(Z, Z_est)
        assert corr == pytest.approx(1.0, abs=1e-6)

    def test_noise_floor(self) -> None:
        rng = np.random.default_rng(4)
        Z_true = rng.standard_normal((200, 3))
        Z_noise = rng.standard_normal((200, 3))
        corr = matched_correlation(Z_true, Z_noise)
        assert corr < 0.3  # noise should have low correlation

    def test_range(self) -> None:
        rng = np.random.default_rng(5)
        A = rng.standard_normal((50, 3))
        B = rng.standard_normal((50, 3))
        corr = matched_correlation(A, B)
        assert 0.0 <= corr <= 1.0


class TestProcrustesAlign:
    def test_aligns_rotation(self) -> None:
        """After alignment, Z_est should be close to Z_ref."""
        rng = np.random.default_rng(6)
        Z_ref = rng.standard_normal((100, 3))
        Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        Z_est = Z_ref @ Q  # rotated version

        Z_aligned = procrustes_align(Z_ref, Z_est)
        residual = np.linalg.norm(np.asarray(Z_aligned) - Z_ref, "fro")
        assert residual < 1e-4
