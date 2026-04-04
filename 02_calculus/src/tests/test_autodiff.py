"""Tests for the autodiff engine."""

import numpy as np
import pytest

from math_for_ml.autodiff import Value, gradient_check


def test_backward_matches_polynomial_derivative() -> None:
    """The engine should recover a simple analytical derivative."""

    x = Value(2.0)
    y = x * x + 3.0 * x
    y.backward()
    assert x.grad == pytest.approx(7.0)


def test_gradient_check_matches_finite_difference() -> None:
    """Analytic and numerical gradients should align."""

    analytic, numeric = gradient_check(
        lambda nodes: ((nodes[0] * nodes[1]) + nodes[0].tanh() + nodes[1].exp()).log(),
        [1.5, 0.4],
    )
    assert np.allclose(analytic, numeric, atol=1e-5)


def test_log_rejects_non_positive_inputs() -> None:
    """Log should validate its domain."""

    with pytest.raises(ValueError):
        Value(0.0).log()
