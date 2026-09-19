"""Correctness gate: ELBO monotonicity for CAVI.

This test is a STUB — it will be filled in Phase 1 when CAVI is implemented.
It is present now so it fails loudly rather than being absent.
"""

import pytest


@pytest.mark.skip(reason="CAVI not yet implemented — Phase 1 task")
def test_elbo_monotonicity() -> None:
    """ELBO must be non-decreasing across all CAVI iterations.

    This is the primary correctness gate for the CAVI implementation.
    See docs/methods.md for the derivation.
    """
    raise NotImplementedError


@pytest.mark.skip(reason="CAVI not yet implemented — Phase 1 task")
def test_cavi_recovers_factors_small() -> None:
    """On a small problem (N=100, D=20, K=3), CAVI should recover
    the true subspace to within a Procrustes-aligned subspace distance < 0.2."""
    raise NotImplementedError
