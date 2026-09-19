"""Tests for M3 spatial model and simulator."""

import jax
import jax.numpy as jnp
import pytest

from sfm.models.m3_spatial import M3Hyperparams, elbo_m3, init_svi_m3, rbf_kernel
from sfm.simulate import simulate_m3


class TestRBFKernel:
    def test_shape(self) -> None:
        coords = jnp.zeros((10, 2))
        K = rbf_kernel(coords, jnp.array(1.0), jnp.array(0.5))
        assert K.shape == (10, 10)

    def test_symmetric(self) -> None:
        key = jax.random.PRNGKey(0)
        coords = jax.random.uniform(key, (15, 2))
        K = rbf_kernel(coords, jnp.array(1.0), jnp.array(0.3))
        assert jnp.allclose(K, K.T, atol=1e-6)

    def test_positive_definite(self) -> None:
        key = jax.random.PRNGKey(1)
        coords = jax.random.uniform(key, (20, 2))
        K = rbf_kernel(coords, jnp.array(1.0), jnp.array(0.3))
        eigvals = jnp.linalg.eigvalsh(K)
        assert jnp.all(eigvals > 0), f"Not PD: min eigenvalue = {float(eigvals.min()):.6f}"

    def test_diagonal_is_amplitude_squared(self) -> None:
        coords = jnp.zeros((5, 2))  # all same location
        K = rbf_kernel(coords, jnp.array(2.0), jnp.array(1.0), jitter=0.0)
        assert jnp.allclose(jnp.diag(K), jnp.full(5, 4.0), atol=1e-5)


class TestSimulateM3:
    def test_output_shapes(self) -> None:
        key = jax.random.PRNGKey(0)
        data = simulate_m3(key, N=50, D=10, K=3, tau=2.0)
        assert data.X.shape == (50, 10)
        assert data.Z_true.shape == (50, 3)
        assert data.W_true.shape == (10, 3)
        assert data.coords.shape == (50, 2)

    def test_grid_side(self) -> None:
        key = jax.random.PRNGKey(1)
        data = simulate_m3(key, grid_side=5, D=10, K=2, tau=2.0)
        assert data.N == 25
        assert data.coords.shape == (25, 2)

    def test_reproducibility(self) -> None:
        key = jax.random.PRNGKey(2)
        d1 = simulate_m3(key, N=30, D=10, K=2)
        d2 = simulate_m3(key, N=30, D=10, K=2)
        assert jnp.allclose(d1.X, d2.X)

    def test_spatial_correlation(self) -> None:
        """Factor scores should be spatially correlated: nearby spots have
        more similar scores than distant spots on average."""
        key = jax.random.PRNGKey(3)
        data = simulate_m3(key, grid_side=10, D=5, K=2, lengthscales=(0.2, 0.2))

        # Compute pairwise distances and score differences for factor 0
        coords = data.coords
        z0 = data.Z_true[:, 0]
        dists = jnp.sqrt(jnp.sum((coords[:, None] - coords[None, :])**2, axis=-1))
        diffs = jnp.abs(z0[:, None] - z0[None, :])

        # Grid spacing is 1/(side-1) = 1/9 ≈ 0.111; use 0.15 as near threshold
        near_mask = (dists < 0.15) & (dists > 0)
        far_mask = dists > 0.5

        near_diff = float(jnp.mean(diffs[near_mask]))
        far_diff = float(jnp.mean(diffs[far_mask]))

        assert near_diff < far_diff, (
            f"Spatial correlation not detected: near_diff={near_diff:.3f}, "
            f"far_diff={far_diff:.3f}"
        )


class TestELBOM3:
    def test_elbo_finite(self) -> None:
        key = jax.random.PRNGKey(0)
        data = simulate_m3(key, N=20, D=8, K=2, tau=2.0)
        hp = M3Hyperparams()
        state = init_svi_m3(data.X, data.coords, 2, hp, jax.random.PRNGKey(1))
        elbo = elbo_m3(data.X, data.coords, state, hp)
        assert jnp.isfinite(elbo), f"ELBO is not finite: {float(elbo)}"

    def test_elbo_scalar(self) -> None:
        key = jax.random.PRNGKey(2)
        data = simulate_m3(key, N=15, D=5, K=2, tau=2.0)
        hp = M3Hyperparams()
        state = init_svi_m3(data.X, data.coords, 2, hp, jax.random.PRNGKey(3))
        elbo = elbo_m3(data.X, data.coords, state, hp)
        assert elbo.shape == ()
