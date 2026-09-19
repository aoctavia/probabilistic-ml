"""SVI — stochastic variational inference for M3 (spatial factor model).

Optimises the M3 ELBO via gradient ascent using JAX's automatic
differentiation. All parameters are in unconstrained space (log-space
for positive quantities); we optimise jointly.

The state is represented as a flat parameter vector to interface cleanly
with optax optimisers. A pack/unpack utility converts between the flat
vector and SVIStateM3.

Correctness gate (Phase 4): gradient of the ELBO against finite differences
on a small instance — verified in tests/test_svi.py.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from sfm.models.m3_spatial import (
    M3Hyperparams,
    SVIStateM3,
    elbo_m3,
    init_svi_m3,
)
from sfm.types import FloatArray, PRNGKey


# ── Parameter pack / unpack ───────────────────────────────────────────────────


def pack_params(state: SVIStateM3) -> FloatArray:
    """Flatten SVIStateM3 into a single 1-D parameter vector."""
    return jnp.concatenate([
        state.m_Z.ravel(),
        state.log_v_Z.ravel(),
        state.M_W.ravel(),
        state.log_S_W_diag.ravel(),
        jnp.array([state.log_a_N, state.log_b_N]),
        state.log_amplitudes.ravel(),
        state.log_lengthscales.ravel(),
    ])


def unpack_params(params: FloatArray, N: int, D: int, K: int) -> SVIStateM3:
    """Reconstruct SVIStateM3 from a flat parameter vector."""
    idx = 0

    def _take(size: int) -> FloatArray:
        nonlocal idx
        out = params[idx: idx + size]
        idx += size
        return out

    m_Z = _take(N * K).reshape(N, K)
    log_v_Z = _take(N * K).reshape(N, K)
    M_W = _take(D * K).reshape(D, K)
    log_S_W_diag = _take(K)
    scalars = _take(2)
    log_amplitudes = _take(K)
    log_lengthscales = _take(K)

    return SVIStateM3(
        m_Z=m_Z,
        log_v_Z=log_v_Z,
        M_W=M_W,
        log_S_W_diag=log_S_W_diag,
        log_a_N=float(scalars[0]),
        log_b_N=float(scalars[1]),
        log_amplitudes=log_amplitudes,
        log_lengthscales=log_lengthscales,
    )


# ── Objective (negative ELBO for minimisation) ───────────────────────────────


def _neg_elbo(
    params: FloatArray,
    X: FloatArray,
    coords: FloatArray,
    hp: M3Hyperparams,
    N: int,
    D: int,
    K: int,
) -> FloatArray:
    state = unpack_params(params, N, D, K)
    return -elbo_m3(X, coords, state, hp)


# ── Result container ──────────────────────────────────────────────────────────


class SVIResult(NamedTuple):
    state: SVIStateM3
    elbo_history: list[float]
    n_steps: int


# ── Main loop ─────────────────────────────────────────────────────────────────


def run_svi_m3(
    X: FloatArray,
    coords: FloatArray,
    K: int,
    hp: M3Hyperparams,
    key: PRNGKey,
    *,
    n_steps: int = 500,
    learning_rate: float = 1e-2,
    init_scale: float = 0.1,
    log_every: int = 50,
) -> SVIResult:
    """Run SVI on the M3 spatial factor model.

    Parameters
    ----------
    X:
        Observed data, shape ``(N, D)``.
    coords:
        Spatial coordinates, shape ``(N, 2)``.
    K:
        Number of latent factors.
    hp:
        Prior hyperparameters.
    key:
        JAX PRNG key for initialisation.
    n_steps:
        Number of gradient steps.
    learning_rate:
        Adam learning rate.
    init_scale:
        Scale of random parameter initialisation.
    log_every:
        Print ELBO every this many steps.

    Returns
    -------
    SVIResult
        Final state, ELBO history (recorded every ``log_every`` steps).
    """
    N, D = X.shape

    state0 = init_svi_m3(X, coords, K, hp, key, init_scale=init_scale)
    params = pack_params(state0)

    optimiser = optax.adam(learning_rate)
    opt_state = optimiser.init(params)

    grad_fn = jax.jit(jax.value_and_grad(
        lambda p: _neg_elbo(p, X, coords, hp, N, D, K)
    ))

    elbo_history: list[float] = []

    for step in range(n_steps):
        neg_elbo_val, grads = grad_fn(params)
        updates, opt_state = optimiser.update(grads, opt_state)
        params = optax.apply_updates(params, updates)

        if step % log_every == 0 or step == n_steps - 1:
            elbo_val = float(-neg_elbo_val)
            elbo_history.append(elbo_val)
            print(f"  step {step:4d}  ELBO={elbo_val:.2f}")

    final_state = unpack_params(params, N, D, K)
    return SVIResult(state=final_state, elbo_history=elbo_history, n_steps=n_steps)
