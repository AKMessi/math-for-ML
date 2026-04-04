"""Probability and statistics helpers used throughout the course."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
__all__ = [
    "bayes_rule",
    "bernoulli_pmf",
    "binomial_pmf",
    "covariance",
    "empirical_cdf",
    "entropy",
    "expectation",
    "gaussian_log_pdf",
    "gaussian_pdf",
    "maximum_a_posteriori_gaussian_mean",
    "maximum_likelihood_gaussian",
    "mutual_information",
    "variance",
]


def _normalize_probabilities(probabilities: ArrayLike) -> FloatArray:
    """Normalize a probability vector and validate its support."""

    array = np.asarray(probabilities, dtype=np.float64)
    if np.any(array < 0.0):
        raise ValueError("Probabilities must be non-negative.")
    total = array.sum()
    if total <= 0.0:
        raise ValueError("At least one probability must be positive.")
    return array / total


def bernoulli_pmf(k: int, p: float) -> float:
    """Evaluate the Bernoulli probability mass function.

    Args:
        k: Outcome, either 0 or 1.
        p: Success probability.

    Returns:
        Probability of observing ``k``.
    """

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must lie in [0, 1].")
    if k not in {0, 1}:
        return 0.0
    return float(p if k == 1 else 1.0 - p)


def binomial_pmf(k: int, n: int, p: float) -> float:
    """Evaluate the Binomial probability mass function."""

    if n < 0:
        raise ValueError("n must be non-negative.")
    if not 0 <= k <= n:
        return 0.0
    coefficient = float(math.comb(n, k))
    return coefficient * (p**k) * ((1.0 - p) ** (n - k))


def gaussian_pdf(x: ArrayLike, mean: float, std: float) -> FloatArray:
    """Evaluate the Gaussian probability density function.

    Args:
        x: Evaluation points.
        mean: Distribution mean.
        std: Distribution standard deviation.

    Returns:
        Density values with the same shape as ``x``.
    """

    if std <= 0.0:
        raise ValueError("Standard deviation must be positive.")
    array = np.asarray(x, dtype=np.float64)
    scale = std * np.sqrt(2.0 * np.pi)
    exponent = -0.5 * ((array - mean) / std) ** 2
    return np.exp(exponent) / scale


def gaussian_log_pdf(x: ArrayLike, mean: float, std: float) -> FloatArray:
    """Evaluate the log-density of a Gaussian random variable."""

    if std <= 0.0:
        raise ValueError("Standard deviation must be positive.")
    array = np.asarray(x, dtype=np.float64)
    return -0.5 * np.log(2.0 * np.pi * std**2) - 0.5 * ((array - mean) / std) ** 2


def expectation(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Compute the expectation of a discrete random variable."""

    value_array = np.asarray(values, dtype=np.float64)
    probability_array = _normalize_probabilities(probabilities)
    if value_array.shape != probability_array.shape:
        raise ValueError("Values and probabilities must have the same shape.")
    return float(np.sum(value_array * probability_array))


def variance(values: ArrayLike, probabilities: ArrayLike) -> float:
    """Compute the variance of a discrete random variable."""

    value_array = np.asarray(values, dtype=np.float64)
    probability_array = _normalize_probabilities(probabilities)
    mean = expectation(value_array, probability_array)
    return float(np.sum(((value_array - mean) ** 2) * probability_array))


def covariance(left: ArrayLike, right: ArrayLike) -> float:
    """Compute the empirical covariance between two samples."""

    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("Samples must share the same shape.")
    if left_array.ndim != 1:
        raise ValueError("Covariance expects one-dimensional samples.")
    left_centered = left_array - left_array.mean()
    right_centered = right_array - right_array.mean()
    return float(np.dot(left_centered, right_centered) / max(len(left_array) - 1, 1))


def maximum_likelihood_gaussian(samples: ArrayLike) -> tuple[float, float]:
    """Estimate Gaussian parameters via maximum likelihood.

    Args:
        samples: One-dimensional sample of observations.

    Returns:
        Tuple ``(mean, std)`` where the variance uses the MLE denominator ``n``.
    """

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("MLE helper expects one-dimensional samples.")
    mean = float(array.mean())
    std = float(np.sqrt(np.mean((array - mean) ** 2)))
    return mean, std


def maximum_a_posteriori_gaussian_mean(
    samples: ArrayLike,
    *,
    prior_mean: float,
    prior_var: float,
    likelihood_var: float,
) -> float:
    """Estimate a Gaussian mean with a Gaussian prior.

    Args:
        samples: Observations from ``N(mu, likelihood_var)``.
        prior_mean: Prior mean for ``mu``.
        prior_var: Prior variance for ``mu``.
        likelihood_var: Observation variance.

    Returns:
        The posterior mean, which is also the MAP estimate.
    """

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("MAP helper expects one-dimensional samples.")
    if prior_var <= 0.0 or likelihood_var <= 0.0:
        raise ValueError("Variances must be positive.")

    precision_prior = 1.0 / prior_var
    precision_likelihood = len(array) / likelihood_var
    posterior_variance = 1.0 / (precision_prior + precision_likelihood)
    posterior_mean = posterior_variance * (
        precision_prior * prior_mean + precision_likelihood * float(array.mean())
    )
    return float(posterior_mean)


def bayes_rule(prior: ArrayLike, likelihood: ArrayLike) -> FloatArray:
    """Compute a normalized posterior distribution.

    Args:
        prior: Prior probabilities over hypotheses.
        likelihood: Likelihood of the observation under each hypothesis.

    Returns:
        Posterior probabilities after applying Bayes' rule.
    """

    prior_array = _normalize_probabilities(prior)
    likelihood_array = np.asarray(likelihood, dtype=np.float64)
    if prior_array.shape != likelihood_array.shape:
        raise ValueError("Prior and likelihood must have the same shape.")
    unnormalized = prior_array * likelihood_array
    evidence = unnormalized.sum()
    if evidence <= 0.0:
        raise ValueError("Likelihoods must provide positive evidence.")
    return unnormalized / evidence


def entropy(probabilities: ArrayLike, *, base: float = 2.0) -> float:
    """Compute Shannon entropy for a discrete distribution."""

    probability_array = _normalize_probabilities(probabilities)
    safe = np.where(probability_array > 0.0, probability_array, 1.0)
    logs = np.log(safe) / np.log(base)
    return float(-np.sum(probability_array * logs))


def mutual_information(joint_distribution: ArrayLike, *, base: float = 2.0) -> float:
    """Compute mutual information from a joint distribution table."""

    joint = _normalize_probabilities(joint_distribution)
    if joint.ndim != 2:
        raise ValueError("Joint distribution must be a matrix.")
    marginal_x = joint.sum(axis=1, keepdims=True)
    marginal_y = joint.sum(axis=0, keepdims=True)
    product = marginal_x @ marginal_y
    mask = joint > 0.0
    logs = np.zeros_like(joint)
    logs[mask] = np.log(joint[mask] / product[mask]) / np.log(base)
    return float(np.sum(joint * logs))


def empirical_cdf(samples: Sequence[float], x: float) -> float:
    """Evaluate the empirical cumulative distribution function."""

    array = np.asarray(samples, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("empirical_cdf expects one-dimensional samples.")
    return float(np.mean(array <= x))
