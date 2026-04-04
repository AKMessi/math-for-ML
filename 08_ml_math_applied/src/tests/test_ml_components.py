"""Tests for applied ML math components."""

import numpy as np
import pytest

from math_for_ml.ml_components import (
    batch_norm_forward,
    convolution_as_matrix_multiplication,
    cross_entropy_loss,
    direct_valid_convolution,
    dropout,
    mean_squared_error,
    scaled_dot_product_attention,
)


def test_basic_losses_are_positive_and_reasonable() -> None:
    """Loss helpers should return sensible scalars."""

    assert mean_squared_error([0.2, 0.8], [0.0, 1.0]) == pytest.approx(0.04)
    logits = np.array([[2.0, 0.5, -1.0], [0.1, 1.5, 0.2]])
    labels = np.array([0, 1])
    assert cross_entropy_loss(logits, labels) > 0.0


def test_dropout_and_batch_norm_behave_as_expected() -> None:
    """Dropout should preserve shape and batch norm should center features."""

    values = np.arange(1.0, 7.0).reshape(2, 3)
    dropped, mask = dropout(values, 0.5, rng=np.random.default_rng(0))
    assert dropped.shape == values.shape
    assert mask.shape == values.shape

    batch = np.array([[1.0, 2.0], [3.0, 0.0], [5.0, 4.0]])
    normalized, _ = batch_norm_forward(batch, gamma=np.ones(2), beta=np.zeros(2))
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)


def test_attention_weights_sum_to_one() -> None:
    """Attention probabilities should normalize along the key dimension."""

    query = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    key = np.array([[[1.0, 0.0], [0.0, 1.0]]])
    value = np.array([[[2.0, 1.0], [0.5, 3.0]]])
    _, weights = scaled_dot_product_attention(query, key, value)
    assert np.allclose(weights.sum(axis=-1), 1.0)


def test_convolution_via_matrix_multiplication_matches_direct_version() -> None:
    """The im2col view of convolution should match direct computation."""

    image = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0]])
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert np.allclose(
        convolution_as_matrix_multiplication(image, kernel),
        direct_valid_convolution(image, kernel),
    )
