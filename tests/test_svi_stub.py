"""Correctness gate: SVI gradient check.

Stub for Phase 4. Present now to make missing implementation visible.
"""

import pytest


@pytest.mark.skip(reason="SVI not yet implemented — Phase 4 task")
def test_svi_gradient_finite_differences() -> None:
    """Gradient of the SVI objective must match finite differences
    to within 1e-4 relative tolerance on a small instance."""
    raise NotImplementedError


@pytest.mark.skip(reason="SVI not yet implemented — Phase 4 task")
def test_svi_vi_converges_to_mcmc() -> None:
    """VI posterior means should converge toward NUTS means as the
    model is simplified and N grows."""
    raise NotImplementedError
