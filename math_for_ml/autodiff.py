"""A tiny scalar automatic differentiation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Callable, Iterable, Sequence

import numpy as np

__all__ = ["Value", "finite_difference", "gradient_check"]


@dataclass(eq=False)
class Value:
    """Scalar value in a computation graph with reverse-mode autodiff.

    Args:
        data: Scalar payload.
        label: Optional human-readable label.

    Examples:
        >>> x = Value(2.0, label="x")
        >>> y = x * x + 3.0 * x
        >>> y.backward()
        >>> x.grad
        7.0
    """

    data: float
    label: str = ""
    grad: float = 0.0
    _prev: tuple["Value", ...] = field(default_factory=tuple, repr=False)
    _op: str = field(default="", repr=False)
    _backward: Callable[[], None] = field(default=lambda: None, repr=False)

    def __post_init__(self) -> None:
        """Normalize the stored scalar type."""

        self.data = float(self.data)

    def __add__(self, other: float | "Value") -> "Value":
        """Add two scalar nodes."""

        other_value = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other_value.data, _prev=(self, other_value), _op="+")

        def _backward() -> None:
            self.grad += output.grad
            other_value.grad += output.grad

        output._backward = _backward
        return output

    def __radd__(self, other: float | "Value") -> "Value":
        """Support right-hand addition."""

        return self + other

    def __sub__(self, other: float | "Value") -> "Value":
        """Subtract two scalar nodes."""

        return self + (-other)

    def __rsub__(self, other: float | "Value") -> "Value":
        """Support right-hand subtraction."""

        return (-self) + other

    def __mul__(self, other: float | "Value") -> "Value":
        """Multiply two scalar nodes."""

        other_value = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other_value.data, _prev=(self, other_value), _op="*")

        def _backward() -> None:
            self.grad += other_value.data * output.grad
            other_value.grad += self.data * output.grad

        output._backward = _backward
        return output

    def __rmul__(self, other: float | "Value") -> "Value":
        """Support right-hand multiplication."""

        return self * other

    def __truediv__(self, other: float | "Value") -> "Value":
        """Divide two scalar nodes."""

        other_value = other if isinstance(other, Value) else Value(other)
        return self * other_value**-1

    def __rtruediv__(self, other: float | "Value") -> "Value":
        """Support right-hand division."""

        other_value = other if isinstance(other, Value) else Value(other)
        return other_value / self

    def __pow__(self, exponent: float) -> "Value":
        """Raise a scalar node to a constant power."""

        output = Value(self.data**exponent, _prev=(self,), _op=f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * (self.data ** (exponent - 1.0)) * output.grad

        output._backward = _backward
        return output

    def __neg__(self) -> "Value":
        """Negate a scalar node."""

        return self * -1.0

    def tanh(self) -> "Value":
        """Apply the hyperbolic tangent nonlinearity."""

        value = math.tanh(self.data)
        output = Value(value, _prev=(self,), _op="tanh")

        def _backward() -> None:
            self.grad += (1.0 - value**2) * output.grad

        output._backward = _backward
        return output

    def exp(self) -> "Value":
        """Apply the exponential function."""

        value = math.exp(self.data)
        output = Value(value, _prev=(self,), _op="exp")

        def _backward() -> None:
            self.grad += value * output.grad

        output._backward = _backward
        return output

    def log(self) -> "Value":
        """Apply the natural logarithm.

        Raises:
            ValueError: If the node contains a non-positive number.
        """

        if self.data <= 0.0:
            raise ValueError("Logarithm is only defined for positive inputs.")
        output = Value(math.log(self.data), _prev=(self,), _op="log")

        def _backward() -> None:
            self.grad += (1.0 / self.data) * output.grad

        output._backward = _backward
        return output

    def relu(self) -> "Value":
        """Apply the ReLU nonlinearity."""

        value = self.data if self.data > 0.0 else 0.0
        output = Value(value, _prev=(self,), _op="relu")

        def _backward() -> None:
            self.grad += (1.0 if self.data > 0.0 else 0.0) * output.grad

        output._backward = _backward
        return output

    def sigmoid(self) -> "Value":
        """Apply the logistic sigmoid nonlinearity."""

        value = 1.0 / (1.0 + math.exp(-self.data))
        output = Value(value, _prev=(self,), _op="sigmoid")

        def _backward() -> None:
            self.grad += value * (1.0 - value) * output.grad

        output._backward = _backward
        return output

    def backward(self) -> None:
        """Run reverse-mode autodiff from the current node."""

        topo: list[Value] = []
        visited: set[int] = set()

        def build(node: Value) -> None:
            node_id = id(node)
            if node_id in visited:
                return
            visited.add(node_id)
            for parent in node._prev:
                build(parent)
            topo.append(node)

        build(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def zero_grad(self) -> None:
        """Reset gradients in the reachable computation graph."""

        for node in self._traverse():
            node.grad = 0.0

    def _traverse(self) -> Iterable["Value"]:
        """Traverse the reachable computation graph once."""

        stack = [self]
        visited: set[int] = set()
        while stack:
            node = stack.pop()
            node_id = id(node)
            if node_id in visited:
                continue
            visited.add(node_id)
            yield node
            stack.extend(node._prev)


def finite_difference(
    function: Callable[[float], float],
    x: float,
    *,
    eps: float = 1e-6,
) -> float:
    """Approximate a derivative using a central finite difference.

    Args:
        function: Scalar function to differentiate.
        x: Point of evaluation.
        eps: Finite-difference step size.

    Returns:
        The numerical derivative estimate.

    Examples:
        >>> round(finite_difference(lambda t: t**2, 3.0), 5)
        6.0
    """

    return float((function(x + eps) - function(x - eps)) / (2.0 * eps))


def gradient_check(
    function: Callable[[Sequence[Value]], Value],
    inputs: Sequence[float],
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare autodiff gradients with finite-difference estimates.

    Args:
        function: Callable that accepts :class:`Value` nodes and returns a
            scalar output node.
        inputs: Raw scalar inputs.
        eps: Finite-difference step size.

    Returns:
        A pair ``(analytic, numeric)`` containing gradients in matching order.
    """

    values = [Value(x) for x in inputs]
    output = function(values)
    output.backward()
    analytic = np.asarray([value.grad for value in values], dtype=np.float64)

    numeric: list[float] = []
    for index in range(len(inputs)):
        plus = list(inputs)
        minus = list(inputs)
        plus[index] += eps
        minus[index] -= eps
        output_plus = function([Value(x) for x in plus]).data
        output_minus = function([Value(x) for x in minus]).data
        numeric.append((output_plus - output_minus) / (2.0 * eps))

    return analytic, np.asarray(numeric, dtype=np.float64)
