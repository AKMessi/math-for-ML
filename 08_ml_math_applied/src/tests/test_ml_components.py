"""Tests for applied ML math components."""

import numpy as np
import pytest

from math_for_ml.ml_components import (
    attention_scores,
    batch_norm_forward,
    causal_attention_mask,
    convolution_as_matrix_multiplication,
    cross_entropy_loss,
    direct_valid_convolution,
    diffusion_forward_process,
    dropout,
    layer_norm_forward,
    linear_beta_schedule,
    mean_squared_error,
    predict_clean_from_noise,
    reparameterize_gaussian,
    scaled_dot_product_attention,
    vae_kl_divergence,
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

    scores = attention_scores(query, key)
    mask = causal_attention_mask(sequence_length=2)
    assert scores.shape == (1, 2, 2)
    assert np.array_equal(mask, np.array([[True, False], [True, True]]))


def test_convolution_via_matrix_multiplication_matches_direct_version() -> None:
    """The im2col view of convolution should match direct computation."""

    image = np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 3.0], [2.0, 1.0, 0.0]])
    kernel = np.array([[1.0, 0.0], [0.0, -1.0]])
    assert np.allclose(
        convolution_as_matrix_multiplication(image, kernel),
        direct_valid_convolution(image, kernel),
    )


def test_layer_norm_diffusion_and_vae_helpers_match_closed_forms() -> None:
    """Applied-model helpers should reproduce the expected algebra."""

    activations = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 8.0]])
    normalized, _ = layer_norm_forward(activations)
    assert np.allclose(normalized.mean(axis=-1), 0.0, atol=1e-6)

    betas = linear_beta_schedule(5, beta_start=1e-4, beta_end=2e-2)
    clean = np.array([[1.0, -1.0], [0.5, 0.25]])
    noise = np.array([[0.2, -0.1], [-0.3, 0.4]])
    noisy, cache = diffusion_forward_process(clean, timestep=3, betas=betas, noise=noise)
    reconstructed = predict_clean_from_noise(noisy, timestep=3, predicted_noise=noise, betas=betas)
    assert cache["signal_rate"] > 0.0
    assert np.allclose(reconstructed, clean, atol=1e-8)

    mean = np.array([[0.0, 0.5], [1.0, -0.5]])
    log_variance = np.log(np.array([[1.0, 0.5], [0.25, 2.0]]))
    sample = reparameterize_gaussian(mean, log_variance, noise=np.zeros_like(mean))
    kl = vae_kl_divergence(mean, log_variance)
    manual_kl = 0.5 * np.sum(np.exp(log_variance) + mean**2 - 1.0 - log_variance, axis=-1)
    assert np.allclose(sample, mean)
    assert np.allclose(kl, manual_kl)
