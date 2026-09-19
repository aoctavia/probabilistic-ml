"""Tests for M0 model definition."""

import jax
import jax.numpy as jnp

from sfm.models.m0_baseline import M0Hyperparams, log_joint
from sfm.simulate import simulate_m0


class TestLogJoint:
    def test_returns_scalar(self) -> None:
        key = jax.random.PRNGKey(0)
        data = simulate_m0(key, N=20, D=10, K=3)
        hp = M0Hyperparams()

        lj = log_joint(data.X, data.Z_true, data.W_true, jnp.log(jnp.array(data.tau_true)), hp)
        assert lj.shape == ()

    def test_higher_tau_increases_log_joint_at_truth(self) -> None:
        """True parameters + correct tau should give higher log joint than
        true parameters + wrong tau, for a model with little noise."""
        key = jax.random.PRNGKey(1)
        data = simulate_m0(key, N=100, D=20, K=3, tau=5.0)
        hp = M0Hyperparams()

        lj_true = log_joint(
            data.X, data.Z_true, data.W_true,
            jnp.log(jnp.array(5.0)), hp,
        )
        lj_wrong = log_joint(
            data.X, data.Z_true, data.W_true,
            jnp.log(jnp.array(0.1)), hp,
        )
        assert float(lj_true) > float(lj_wrong)

    def test_finite(self) -> None:
        key = jax.random.PRNGKey(2)
        data = simulate_m0(key, N=50, D=15, K=4)
        hp = M0Hyperparams()

        lj = log_joint(data.X, data.Z_true, data.W_true, jnp.log(jnp.array(data.tau_true)), hp)
        assert jnp.isfinite(lj)
