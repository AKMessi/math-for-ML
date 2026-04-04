"""Tests for probability and statistics helpers."""

import numpy as np
import pytest

from math_for_ml.distributions import (
    bayes_rule,
    bernoulli_pmf,
    binomial_pmf,
    entropy,
    gaussian_pdf,
    maximum_a_posteriori_gaussian_mean,
    maximum_likelihood_gaussian,
    mutual_information,
)


def test_discrete_pmf_helpers_return_expected_values() -> None:
    """Bernoulli and Binomial helpers should match closed forms."""

    assert bernoulli_pmf(1, 0.3) == pytest.approx(0.3)
    assert binomial_pmf(2, 4, 0.5) == pytest.approx(0.375)


def test_gaussian_pdf_integrates_to_one_approximately() -> None:
    """A Gaussian density sampled on a wide grid should integrate near one."""

    x = np.linspace(-6.0, 6.0, 5000)
    density = gaussian_pdf(x, mean=0.0, std=1.0)
    integral = np.trapezoid(density, x)
    assert integral == pytest.approx(1.0, rel=1e-3)


def test_mle_and_map_estimators_are_reasonable() -> None:
    """MLE and MAP should produce sensible central estimates."""

    samples = np.array([1.2, 0.9, 1.4, 1.1, 1.0])
    mle_mean, mle_std = maximum_likelihood_gaussian(samples)
    map_mean = maximum_a_posteriori_gaussian_mean(
        samples,
        prior_mean=0.0,
        prior_var=1.0,
        likelihood_var=0.25,
    )
    assert mle_mean == pytest.approx(samples.mean())
    assert mle_std > 0.0
    assert 0.0 < map_mean < mle_mean


def test_bayes_rule_entropy_and_mutual_information() -> None:
    """Posterior and information helpers should be well-behaved."""

    posterior = bayes_rule([0.6, 0.4], [0.2, 0.8])
    joint = np.array([[0.4, 0.1], [0.1, 0.4]])
    assert posterior.sum() == pytest.approx(1.0)
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert mutual_information(joint) > 0.0
