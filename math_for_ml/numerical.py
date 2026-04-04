"""Numerical methods and stability helpers."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
__all__ = [
    "condition_number",
    "finite_difference_derivative",
    "logsumexp",
    "machine_epsilon",
    "relative_error",
    "safe_softmax",
    "simpsons_rule",
    "solve_linear_system_numerically",
    "stable_sigmoid",
    "trapezoidal_rule",
]


def machine_epsilon(dtype: type[np.floating] = np.float64) -> float:
    """Return machine epsilon for a floating-point dtype.

    Examples:
        >>> machine_epsilon() > 0
        True
    """

    return float(np.finfo(dtype).eps)


def logsumexp(values: ArrayLike, axis: int | None = None) -> FloatArray:
    """Compute log-sum-exp in a numerically stable way."""

    array = np.asarray(values, dtype=np.float64)
    max_values = np.max(array, axis=axis, keepdims=True)
    shifted = array - max_values
    summed = np.sum(np.exp(shifted), axis=axis, keepdims=True)
    result = np.log(summed) + max_values
    if axis is None:
        return np.asarray(result.squeeze(), dtype=np.float64)
    return np.asarray(np.squeeze(result, axis=axis), dtype=np.float64)


def safe_softmax(logits: ArrayLike, axis: int = -1) -> FloatArray:
    """Compute a numerically stable softmax."""

    array = np.asarray(logits, dtype=np.float64)
    shifted = array - np.max(array, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def stable_sigmoid(values: ArrayLike) -> FloatArray:
    """Compute the logistic sigmoid without overflow."""

    array = np.asarray(values, dtype=np.float64)
    positive = array >= 0.0
    negative = ~positive
    result = np.empty_like(array)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exp_values = np.exp(array[negative])
    result[negative] = exp_values / (1.0 + exp_values)
    return result


def condition_number(matrix: ArrayLike, ord: int | float = 2) -> float:
    """Compute a matrix condition number."""

    array = np.asarray(matrix, dtype=np.float64)
    return float(np.linalg.cond(array, p=ord))


def solve_linear_system_numerically(matrix: ArrayLike, rhs: ArrayLike) -> FloatArray:
    """Solve a linear system using NumPy's dense solver."""

    array = np.asarray(matrix, dtype=np.float64)
    rhs_array = np.asarray(rhs, dtype=np.float64)
    return np.linalg.solve(array, rhs_array)


def trapezoidal_rule(x: ArrayLike, y: ArrayLike) -> float:
    """Approximate an integral using the composite trapezoidal rule."""

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must share the same shape.")
    return float(np.trapezoid(y_array, x_array))


def simpsons_rule(x: ArrayLike, y: ArrayLike) -> float:
    """Approximate an integral using composite Simpson's rule.

    Raises:
        ValueError: If the grid does not contain an odd number of equally
            spaced points.
    """

    x_array = np.asarray(x, dtype=np.float64)
    y_array = np.asarray(y, dtype=np.float64)
    if x_array.shape != y_array.shape:
        raise ValueError("x and y must share the same shape.")
    if len(x_array) < 3 or len(x_array) % 2 == 0:
        raise ValueError("Simpson's rule requires an odd number of grid points.")
    step_sizes = np.diff(x_array)
    if not np.allclose(step_sizes, step_sizes[0]):
        raise ValueError("Simpson's rule expects equally spaced points.")

    h = step_sizes[0]
    odd_sum = np.sum(y_array[1:-1:2])
    even_sum = np.sum(y_array[2:-1:2])
    return float((h / 3.0) * (y_array[0] + y_array[-1] + 4.0 * odd_sum + 2.0 * even_sum))


def finite_difference_derivative(
    function: Callable[[float], float],
    x: float,
    *,
    eps: float | None = None,
) -> float:
    """Approximate a derivative with a scale-aware central difference."""

    step = eps if eps is not None else np.sqrt(machine_epsilon()) * max(1.0, abs(x))
    return float((function(x + step) - function(x - step)) / (2.0 * step))


def relative_error(estimate: ArrayLike, truth: ArrayLike) -> float:
    """Compute relative error with a small denominator safeguard."""

    estimate_array = np.asarray(estimate, dtype=np.float64)
    truth_array = np.asarray(truth, dtype=np.float64)
    denominator = np.linalg.norm(truth_array) + 1e-12
    return float(np.linalg.norm(estimate_array - truth_array) / denominator)
