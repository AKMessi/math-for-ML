"""Tests for information-theoretic helpers."""

import numpy as np
import pytest

from math_for_ml.information import (
    conditional_entropy,
    cross_entropy,
    entropy,
    gaussian_rate_distortion,
    joint_entropy,
    kl_divergence,
    mutual_information,
)


def test_entropy_and_cross_entropy_are_consistent() -> None:
    """Cross-entropy should dominate entropy when the distributions differ."""

    p = np.array([0.5, 0.5])
    q = np.array([0.75, 0.25])
    assert entropy(p) == pytest.approx(1.0)
    assert cross_entropy(p, q) > entropy(p)
    assert kl_divergence(p, q) >= 0.0


def test_joint_conditional_and_mutual_information_relationships() -> None:
    """Information measures should satisfy standard identities."""

    joint = np.array([[0.4, 0.1], [0.1, 0.4]])
    assert joint_entropy(joint) == pytest.approx(entropy(joint.ravel()))
    assert conditional_entropy(joint) < joint_entropy(joint)
    assert mutual_information(joint) > 0.0


def test_gaussian_rate_distortion_is_zero_above_variance() -> None:
    """The Gaussian rate-distortion curve should clip to zero."""

    distortions = np.array([0.1, 0.5, 1.5])
    rates = gaussian_rate_distortion(1.0, distortions)
    assert rates[0] > rates[1] > rates[2]
    assert rates[2] == pytest.approx(0.0)
