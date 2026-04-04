"""Optimization algorithms implemented with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
__all__ = [
    "Adam",
    "GradientDescent",
    "Momentum",
    "OptimizeResult",
    "Optimizer",
    "RMSProp",
    "SGD",
    "minimize",
    "newton_step",
]


class ObjectiveGradient(Protocol):
    """Protocol for gradient callables used by optimization helpers."""

    def __call__(self, parameters: FloatArray) -> FloatArray:
        """Return the gradient at the supplied parameters."""


class ObjectiveValue(Protocol):
    """Protocol for objective functions used by optimization helpers."""

    def __call__(self, parameters: FloatArray) -> float:
        """Return the scalar objective value."""


@dataclass(frozen=True)
class OptimizeResult:
    """Summary of an optimization run.

    Attributes:
        parameters: Final parameters.
        losses: Objective history.
        iterates: Parameter trajectory, including the initial point.
    """

    parameters: FloatArray
    losses: FloatArray
    iterates: FloatArray


class Optimizer:
    """Base class for iterative first-order optimizers."""

    def step(self, parameters: FloatArray, gradients: FloatArray) -> FloatArray:
        """Update parameters by one optimization step."""

        raise NotImplementedError


class GradientDescent(Optimizer):
    """Vanilla gradient descent."""

    def __init__(self, learning_rate: float = 1e-2) -> None:
        """Store the step size."""

        self.learning_rate = learning_rate

    def step(self, parameters: FloatArray, gradients: FloatArray) -> FloatArray:
        """Take one gradient descent step."""

        return parameters - self.learning_rate * gradients


class SGD(Optimizer):
    """Stochastic gradient descent without momentum."""

    def __init__(self, learning_rate: float = 1e-2) -> None:
        """Store the step size."""

        self.learning_rate = learning_rate

    def step(self, parameters: FloatArray, gradients: FloatArray) -> FloatArray:
        """Take one stochastic gradient descent step."""

        return parameters - self.learning_rate * gradients


class Momentum(Optimizer):
    """Momentum SGD."""

    def __init__(self, learning_rate: float = 1e-2, momentum: float = 0.9) -> None:
        """Initialize optimizer state."""

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity: FloatArray | None = None

    def step(self, parameters: FloatArray, gradients: FloatArray) -> FloatArray:
        """Take one momentum step."""

        if self.velocity is None:
            self.velocity = np.zeros_like(parameters)
        self.velocity = self.momentum * self.velocity - self.learning_rate * gradients
        return parameters + self.velocity


class RMSProp(Optimizer):
    """RMSProp optimizer."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        decay_rate: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize optimizer state."""

        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.square_average: FloatArray | None = None

    def step(self, parameters: FloatArray, gradients: FloatArray) -> FloatArray:
        """Take one RMSProp step."""

        if self.square_average is None:
            self.square_average = np.zeros_like(parameters)
        self.square_average = (
            self.decay_rate * self.square_average
            + (1.0 - self.decay_rate) * gradients**2
        )
        adjusted = gradients / (np.sqrt(self.square_average) + self.epsilon)
        return parameters - self.learning_rate * adjusted


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize optimizer state."""

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.first_moment: FloatArray | None = None
        self.second_moment: FloatArray | None = None
        self.timestep = 0

    def step(self, parameters: FloatArray, gradients: FloatArray) -> FloatArray:
        """Take one Adam step."""

        if self.first_moment is None:
            self.first_moment = np.zeros_like(parameters)
            self.second_moment = np.zeros_like(parameters)

        self.timestep += 1
        self.first_moment = self.beta1 * self.first_moment + (1.0 - self.beta1) * gradients
        self.second_moment = self.beta2 * self.second_moment + (1.0 - self.beta2) * (
            gradients**2
        )

        first_hat = self.first_moment / (1.0 - self.beta1**self.timestep)
        second_hat = self.second_moment / (1.0 - self.beta2**self.timestep)
        return parameters - self.learning_rate * first_hat / (
            np.sqrt(second_hat) + self.epsilon
        )


def minimize(
    objective: ObjectiveValue,
    gradient: ObjectiveGradient,
    initial_parameters: ArrayLike,
    optimizer: Optimizer,
    *,
    iterations: int = 100,
) -> OptimizeResult:
    """Minimize an objective with a supplied optimizer.

    Args:
        objective: Scalar objective function.
        gradient: Gradient of the objective.
        initial_parameters: Starting point.
        optimizer: Optimizer instance.
        iterations: Number of optimization steps.

    Returns:
        Trajectory, loss history, and final parameters.
    """

    parameters = np.asarray(initial_parameters, dtype=np.float64).copy()
    iterates = [parameters.copy()]
    losses = [objective(parameters)]

    for _ in range(iterations):
        gradients = np.asarray(gradient(parameters), dtype=np.float64)
        parameters = optimizer.step(parameters, gradients)
        iterates.append(parameters.copy())
        losses.append(objective(parameters))

    return OptimizeResult(
        parameters=parameters,
        losses=np.asarray(losses, dtype=np.float64),
        iterates=np.asarray(iterates, dtype=np.float64),
    )


def newton_step(
    parameters: ArrayLike,
    gradient: ArrayLike,
    hessian: ArrayLike,
    *,
    damping: float = 1e-8,
) -> FloatArray:
    """Take a damped Newton step.

    Args:
        parameters: Current parameters.
        gradient: Gradient at ``parameters``.
        hessian: Hessian at ``parameters``.
        damping: Small diagonal shift for numerical stability.

    Returns:
        Updated parameters after one Newton step.
    """

    parameter_array = np.asarray(parameters, dtype=np.float64)
    gradient_array = np.asarray(gradient, dtype=np.float64)
    hessian_array = np.asarray(hessian, dtype=np.float64)
    identity = np.eye(hessian_array.shape[0], dtype=np.float64)
    direction = np.linalg.solve(hessian_array + damping * identity, gradient_array)
    return parameter_array - direction
