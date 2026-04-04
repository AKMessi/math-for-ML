"""Information-theoretic utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
__all__ = [
    "binary_entropy",
    "conditional_entropy",
    "cross_entropy",
    "entropy",
    "gaussian_rate_distortion",
    "joint_entropy",
    "kl_divergence",
    "mutual_information",
    "perplexity",
]


def _normalize(probabilities: ArrayLike) -> FloatArray:
    """Normalize a non-negative array into a probability distribution."""

    array = np.asarray(probabilities, dtype=np.float64)
    if np.any(array < 0.0):
        raise ValueError("Probabilities must be non-negative.")
    total = array.sum()
    if total <= 0.0:
        raise ValueError("The distribution must have positive mass.")
    return array / total


def entropy(probabilities: ArrayLike, *, base: float = 2.0) -> float:
    """Compute Shannon entropy."""

    distribution = _normalize(probabilities)
    safe = np.where(distribution > 0.0, distribution, 1.0)
    logs = np.log(safe) / np.log(base)
    return float(-np.sum(distribution * logs))


def cross_entropy(target: ArrayLike, prediction: ArrayLike, *, base: float = 2.0) -> float:
    """Compute cross-entropy between two discrete distributions."""

    target_distribution = _normalize(target)
    predicted_distribution = _normalize(prediction)
    safe = np.clip(predicted_distribution, 1e-12, None)
    logs = np.log(safe) / np.log(base)
    return float(-np.sum(target_distribution * logs))


def kl_divergence(reference: ArrayLike, comparison: ArrayLike, *, base: float = 2.0) -> float:
    """Compute the Kullback-Leibler divergence ``KL(reference || comparison)``."""

    reference_distribution = _normalize(reference)
    comparison_distribution = np.clip(_normalize(comparison), 1e-12, None)
    safe_reference = np.where(reference_distribution > 0.0, reference_distribution, 1.0)
    ratio = safe_reference / comparison_distribution
    logs = np.log(ratio) / np.log(base)
    return float(np.sum(reference_distribution * logs))


def joint_entropy(joint_distribution: ArrayLike, *, base: float = 2.0) -> float:
    """Compute entropy of a joint distribution table."""

    joint = _normalize(joint_distribution)
    return entropy(joint.ravel(), base=base)


def conditional_entropy(joint_distribution: ArrayLike, *, base: float = 2.0) -> float:
    """Compute ``H(Y | X)`` from a joint distribution table ``p(x, y)``."""

    joint = _normalize(joint_distribution)
    if joint.ndim != 2:
        raise ValueError("Joint distribution must be a matrix.")
    return joint_entropy(joint, base=base) - entropy(joint.sum(axis=1), base=base)


def mutual_information(joint_distribution: ArrayLike, *, base: float = 2.0) -> float:
    """Compute mutual information from a joint distribution table."""

    joint = _normalize(joint_distribution)
    if joint.ndim != 2:
        raise ValueError("Joint distribution must be a matrix.")
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    product = marginal_x @ marginal_y
    mask = joint > 0.0
    logs = np.zeros_like(joint)
    logs[mask] = np.log(joint[mask] / product[mask]) / np.log(base)
    return float(np.sum(joint * logs))


def perplexity(probabilities: ArrayLike, *, base: float = 2.0) -> float:
    """Compute perplexity from a discrete distribution."""

    return float(base ** entropy(probabilities, base=base))


def binary_entropy(p: float, *, base: float = 2.0) -> float:
    """Compute the entropy of a Bernoulli random variable."""

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1].")
    return entropy([p, 1.0 - p], base=base)


def gaussian_rate_distortion(variance: float, distortion: ArrayLike) -> FloatArray:
    """Evaluate the Gaussian rate-distortion function.

    Args:
        variance: Source variance.
        distortion: Allowed distortion values.

    Returns:
        ``R(D) = 0.5 * log2(variance / D)`` for ``0 < D < variance`` and zero
        otherwise.
    """

    if variance <= 0.0:
        raise ValueError("variance must be positive.")
    distortion_array = np.asarray(distortion, dtype=np.float64)
    rates = np.zeros_like(distortion_array)
    mask = (distortion_array > 0.0) & (distortion_array < variance)
    rates[mask] = 0.5 * np.log2(variance / distortion_array[mask])
    return rates
