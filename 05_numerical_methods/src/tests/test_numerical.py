"""Tests for numerical helpers."""

import numpy as np
import pytest

from math_for_ml.numerical import (
    finite_difference_derivative,
    logsumexp,
    safe_softmax,
    simpsons_rule,
    stable_sigmoid,
    trapezoidal_rule,
)


def test_safe_softmax_matches_probability_axioms() -> None:
    """Softmax outputs should sum to one."""

    logits = np.array([[1000.0, 1001.0, 1002.0]])
    probabilities = safe_softmax(logits, axis=1)
    assert probabilities.sum() == pytest.approx(1.0)
    assert np.all(probabilities > 0.0)


def test_logsumexp_matches_shifted_manual_computation() -> None:
    """Stable log-sum-exp should agree with a manually stabilized computation."""

    values = np.array([1000.0, 1001.0, 1002.0])
    shifted = values - values.max()
    manual = np.log(np.sum(np.exp(shifted))) + values.max()
    assert logsumexp(values) == pytest.approx(manual)


def test_integration_rules_and_derivative_are_accurate() -> None:
    """Basic numerical methods should be accurate on smooth functions."""

    x = np.linspace(0.0, np.pi, 101)
    y = np.sin(x)
    assert trapezoidal_rule(x, y) == pytest.approx(2.0, rel=1e-4)
    assert simpsons_rule(x, y) == pytest.approx(2.0, rel=1e-6)
    assert finite_difference_derivative(lambda t: t**2, 3.0) == pytest.approx(6.0, rel=1e-5)


def test_stable_sigmoid_handles_extreme_values() -> None:
    """Stable sigmoid should avoid overflow and stay in (0, 1)."""

    result = stable_sigmoid(np.array([-1000.0, 0.0, 1000.0]))
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)
