"""Tests for the M0 simulator (Phase 0 correctness gate)."""

import jax
import jax.numpy as jnp
import pytest

from sfm.simulate import M0Data, SweepSpec, simulate_m0, simulate_sweep


class TestSimulateM0:
    def test_output_shapes(self) -> None:
        key = jax.random.PRNGKey(0)
        data = simulate_m0(key, N=100, D=30, K=4, tau=2.0)

        assert data.X.shape == (100, 30)
        assert data.Z_true.shape == (100, 4)
        assert data.W_true.shape == (30, 4)
        assert data.tau_true == pytest.approx(2.0)
        assert data.N == 100
        assert data.D == 30
        assert data.K == 4

    def test_signal_to_noise(self) -> None:
        """With high tau (low noise), X should be close to Z W^T."""
        key = jax.random.PRNGKey(42)
        data = simulate_m0(key, N=500, D=50, K=3, tau=1000.0)

        mean = data.Z_true @ data.W_true.T
        residual_var = float(jnp.var(data.X - mean))
        signal_var = float(jnp.var(mean))

        # residual variance should be much smaller than signal variance
        assert residual_var < 0.01 * signal_var, (
            f"Expected low residual variance with high tau, "
            f"got signal_var={signal_var:.3f}, residual_var={residual_var:.3f}"
        )

    def test_reproducibility(self) -> None:
        """Same key must produce identical output."""
        key = jax.random.PRNGKey(7)
        d1 = simulate_m0(key, N=50, D=20, K=3)
        d2 = simulate_m0(key, N=50, D=20, K=3)

        assert jnp.allclose(d1.X, d2.X)
        assert jnp.allclose(d1.Z_true, d2.Z_true)
        assert jnp.allclose(d1.W_true, d2.W_true)

    def test_different_keys_differ(self) -> None:
        """Different keys must (almost certainly) produce different output."""
        d1 = simulate_m0(jax.random.PRNGKey(0), N=50, D=20, K=3)
        d2 = simulate_m0(jax.random.PRNGKey(1), N=50, D=20, K=3)

        assert not jnp.allclose(d1.X, d2.X)

    def test_frozen_dataclass(self) -> None:
        """M0Data must be immutable (frozen dataclass)."""
        key = jax.random.PRNGKey(0)
        data = simulate_m0(key, N=10, D=5, K=2)

        with pytest.raises((AttributeError, TypeError)):
            data.N = 999  # type: ignore[misc]


class TestSimulateSweep:
    def test_sweep_length(self) -> None:
        specs = [
            SweepSpec(N=100, D=20, K=3, tau=1.0, seed=i)
            for i in range(5)
        ]
        results = simulate_sweep(specs)
        assert len(results) == 5

    def test_sweep_shapes(self) -> None:
        specs = [
            SweepSpec(N=50, D=10, K=2, tau=2.0, seed=0),
            SweepSpec(N=200, D=30, K=5, tau=0.5, seed=1),
        ]
        results = simulate_sweep(specs)

        assert results[0].X.shape == (50, 10)
        assert results[1].X.shape == (200, 30)
