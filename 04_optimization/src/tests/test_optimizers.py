"""Tests for optimization helpers."""

import numpy as np

from math_for_ml.optimizers import Adam, GradientDescent, Momentum, RMSProp, minimize, newton_step


def quadratic_objective(parameters: np.ndarray) -> float:
    """Quadratic bowl used in optimizer tests."""

    return float(np.sum((parameters - 3.0) ** 2))


def quadratic_gradient(parameters: np.ndarray) -> np.ndarray:
    """Gradient of the quadratic bowl."""

    return 2.0 * (parameters - 3.0)


def test_gradient_descent_converges_on_quadratic() -> None:
    """Vanilla gradient descent should reach the minimizer."""

    result = minimize(
        quadratic_objective,
        quadratic_gradient,
        np.array([8.0]),
        GradientDescent(learning_rate=0.1),
        iterations=50,
    )
    assert np.allclose(result.parameters, np.array([3.0]), atol=1e-3)
    assert result.losses[-1] < result.losses[0]


def test_adaptive_optimizers_reduce_loss() -> None:
    """Momentum, RMSProp, and Adam should all make progress."""

    for optimizer in (Momentum(learning_rate=0.1), RMSProp(learning_rate=0.1), Adam(learning_rate=0.2)):
        result = minimize(
            quadratic_objective,
            quadratic_gradient,
            np.array([8.0]),
            optimizer,
            iterations=30,
        )
        assert result.losses[-1] < result.losses[0]


def test_newton_step_solves_quadratic_in_one_update() -> None:
    """Newton's method should jump to the minimizer on a quadratic."""

    updated = newton_step(np.array([5.0]), np.array([4.0]), np.array([[2.0]]))
    assert np.allclose(updated, np.array([3.0]))
