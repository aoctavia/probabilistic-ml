"""Correctness gates for CAVI on M0.

Gate 1: ELBO is monotonically non-decreasing across all iterations.
Gate 2: On a small synthetic problem, CAVI recovers the true factor
        subspace (subspace distance < 0.3 and matched correlation > 0.7).
"""

import jax
import jax.numpy as jnp
import pytest

from sfm.diagnostics.alignment import matched_correlation, subspace_distance
from sfm.inference.cavi import compute_elbo, run_cavi
from sfm.models.m0_baseline import M0Hyperparams
from sfm.simulate import simulate_m0


class TestELBOMonotonicity:
    """Gate 1: ELBO must be non-decreasing."""

    def test_elbo_non_decreasing_small(self) -> None:
        key = jax.random.PRNGKey(0)
        data = simulate_m0(key, N=80, D=20, K=3, tau=2.0)
        hp = M0Hyperparams()

        result = run_cavi(
            data.X, K=3, hp=hp,
            key=jax.random.PRNGKey(1),
            max_iter=100, tol=1e-8,
        )

        diffs = [
            result.elbo_history[i + 1] - result.elbo_history[i]
            for i in range(len(result.elbo_history) - 1)
        ]
        violations = [d for d in diffs if d < -1e-6]  # allow tiny numerical noise
        assert len(violations) == 0, (
            f"ELBO decreased {len(violations)} times. "
            f"Worst drop: {min(violations):.6f}"
        )

    def test_elbo_non_decreasing_larger(self) -> None:
        key = jax.random.PRNGKey(42)
        data = simulate_m0(key, N=300, D=60, K=5, tau=3.0)
        hp = M0Hyperparams()

        result = run_cavi(
            data.X, K=5, hp=hp,
            key=jax.random.PRNGKey(7),
            max_iter=200, tol=1e-8,
        )

        diffs = [
            result.elbo_history[i + 1] - result.elbo_history[i]
            for i in range(len(result.elbo_history) - 1)
        ]
        violations = [d for d in diffs if d < -1e-6]
        assert len(violations) == 0, (
            f"ELBO decreased {len(violations)} times. "
            f"Worst drop: {min(violations):.6f}"
        )

    def test_elbo_increases_meaningfully(self) -> None:
        """ELBO at convergence should be substantially higher than at init."""
        key = jax.random.PRNGKey(5)
        data = simulate_m0(key, N=200, D=40, K=4, tau=2.0)
        hp = M0Hyperparams()

        result = run_cavi(
            data.X, K=4, hp=hp,
            key=jax.random.PRNGKey(6),
            max_iter=300, tol=1e-6,
        )

        elbo_gain = result.elbo_history[-1] - result.elbo_history[0]
        assert elbo_gain > 0, f"ELBO did not increase: gain = {elbo_gain:.2f}"


class TestFactorRecovery:
    """Gate 2: CAVI should recover the true factor subspace."""

    def test_subspace_distance_small_problem(self) -> None:
        """On N=200, D=40, K=3 with moderate SNR, subspace distance < 0.3."""
        key = jax.random.PRNGKey(10)
        data = simulate_m0(key, N=200, D=40, K=3, tau=3.0)
        hp = M0Hyperparams()

        result = run_cavi(
            data.X, K=3, hp=hp,
            key=jax.random.PRNGKey(11),
            max_iter=500, tol=1e-6,
        )

        dist = subspace_distance(data.Z_true, result.state.M_Z)
        assert dist < 0.3, (
            f"Subspace distance too large: {dist:.4f}. "
            "CAVI may not be converging to the right solution."
        )

    def test_matched_correlation_small_problem(self) -> None:
        """Matched correlation of factor scores > 0.7 on an easy problem."""
        key = jax.random.PRNGKey(20)
        data = simulate_m0(key, N=300, D=50, K=3, tau=5.0)
        hp = M0Hyperparams()

        result = run_cavi(
            data.X, K=3, hp=hp,
            key=jax.random.PRNGKey(21),
            max_iter=500, tol=1e-6,
        )

        corr = matched_correlation(data.Z_true, result.state.M_Z)
        assert corr > 0.7, (
            f"Matched correlation too low: {corr:.4f}. "
            "Factor recovery failed."
        )

    def test_noise_precision_recovered(self) -> None:
        """Posterior mean E[tau] should be within 2x of true tau."""
        key = jax.random.PRNGKey(30)
        true_tau = 4.0
        data = simulate_m0(key, N=300, D=50, K=3, tau=true_tau)
        hp = M0Hyperparams()

        result = run_cavi(
            data.X, K=3, hp=hp,
            key=jax.random.PRNGKey(31),
            max_iter=500, tol=1e-6,
        )

        e_tau = result.state.a_N / result.state.b_N
        ratio = e_tau / true_tau
        assert 0.5 < ratio < 2.0, (
            f"E[tau]={e_tau:.2f} far from true tau={true_tau:.2f} "
            f"(ratio={ratio:.2f})"
        )


class TestCAVIIntegrity:
    """Sanity checks on the CAVI implementation."""

    def test_elbo_finite(self) -> None:
        key = jax.random.PRNGKey(99)
        data = simulate_m0(key, N=50, D=15, K=2, tau=2.0)
        hp = M0Hyperparams()

        result = run_cavi(data.X, K=2, hp=hp, key=jax.random.PRNGKey(100), max_iter=50)
        assert all(
            jnp.isfinite(jnp.array(e)) for e in result.elbo_history
        ), "ELBO contained NaN or Inf"

    def test_convergence_flag(self) -> None:
        """With many iterations and a loose tolerance, should converge."""
        key = jax.random.PRNGKey(50)
        data = simulate_m0(key, N=100, D=20, K=3, tau=2.0)
        hp = M0Hyperparams()

        result = run_cavi(
            data.X, K=3, hp=hp,
            key=jax.random.PRNGKey(51),
            max_iter=1000, tol=1e-3,
        )
        assert result.converged, (
            f"CAVI did not converge in {result.n_iter} iterations "
            f"(final ELBO change too large)"
        )

    def test_output_shapes(self) -> None:
        key = jax.random.PRNGKey(60)
        data = simulate_m0(key, N=50, D=15, K=4, tau=2.0)
        hp = M0Hyperparams()

        result = run_cavi(data.X, K=4, hp=hp, key=jax.random.PRNGKey(61), max_iter=10)

        assert result.state.M_Z.shape == (50, 4)
        assert result.state.S_Z.shape == (4, 4)
        assert result.state.M_W.shape == (15, 4)
        assert result.state.S_W.shape == (4, 4)
        assert len(result.elbo_history) == 10

    def test_compute_elbo_matches_run_elbo(self) -> None:
        """compute_elbo called externally should match the last stored value."""
        key = jax.random.PRNGKey(70)
        data = simulate_m0(key, N=60, D=20, K=3, tau=2.0)
        hp = M0Hyperparams()

        result = run_cavi(data.X, K=3, hp=hp, key=jax.random.PRNGKey(71), max_iter=50)
        elbo_recomputed = compute_elbo(data.X, result.state, hp)

        assert abs(elbo_recomputed - result.elbo_history[-1]) < 1e-5
