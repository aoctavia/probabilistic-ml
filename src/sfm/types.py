"""Shared type aliases used across the package."""

from typing import TypeAlias

import jax.numpy as jnp
from jax import Array

# A JAX array (used everywhere for data, parameters, samples).
FloatArray: TypeAlias = Array

# A JAX PRNG key (shape (2,), dtype uint32).
PRNGKey: TypeAlias = Array
