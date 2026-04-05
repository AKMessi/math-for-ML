"""Machine-learning building blocks derived from math first principles."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from math_for_ml.numerical import safe_softmax

FloatArray = NDArray[np.float64]
__all__ = [
    "attention_scores",
    "batch_norm_forward",
    "causal_attention_mask",
    "convolution_as_matrix_multiplication",
    "cross_entropy_loss",
    "direct_valid_convolution",
    "dropout",
    "diffusion_forward_process",
    "linear_beta_schedule",
    "focal_loss_binary",
    "im2col_2d",
    "l1_penalty",
    "l2_penalty",
    "layer_norm_forward",
    "mean_squared_error",
    "predict_clean_from_noise",
    "reparameterize_gaussian",
    "scaled_dot_product_attention",
    "vae_kl_divergence",
]


def mean_squared_error(prediction: ArrayLike, target: ArrayLike) -> float:
    """Compute mean squared error."""

    prediction_array = np.asarray(prediction, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must have the same shape.")
    return float(np.mean((prediction_array - target_array) ** 2))


def cross_entropy_loss(logits: ArrayLike, targets: ArrayLike) -> float:
    """Compute multi-class cross-entropy from logits.

    Args:
        logits: Array of shape ``(batch_size, num_classes)``.
        targets: Integer class labels with shape ``(batch_size,)``.

    Returns:
        Average negative log-likelihood.
    """

    logits_array = np.asarray(logits, dtype=np.float64)
    targets_array = np.asarray(targets, dtype=np.int64)
    if logits_array.ndim != 2:
        raise ValueError("logits must be a 2D array.")
    if targets_array.shape != (logits_array.shape[0],):
        raise ValueError("targets must contain one class index per row of logits.")

    probabilities = safe_softmax(logits_array, axis=1)
    clipped = np.clip(probabilities[np.arange(len(targets_array)), targets_array], 1e-12, None)
    return float(-np.mean(np.log(clipped)))


def focal_loss_binary(
    probabilities: ArrayLike,
    targets: ArrayLike,
    *,
    gamma: float = 2.0,
    alpha: float = 0.25,
) -> float:
    """Compute binary focal loss from probabilities."""

    probability_array = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    target_array = np.asarray(targets, dtype=np.float64)
    if probability_array.shape != target_array.shape:
        raise ValueError("probabilities and targets must have the same shape.")
    pt = np.where(target_array == 1.0, probability_array, 1.0 - probability_array)
    alpha_t = np.where(target_array == 1.0, alpha, 1.0 - alpha)
    loss = -alpha_t * ((1.0 - pt) ** gamma) * np.log(pt)
    return float(np.mean(loss))


def l1_penalty(parameters: ArrayLike, coefficient: float = 1.0) -> float:
    """Compute an L1 regularization penalty."""

    parameter_array = np.asarray(parameters, dtype=np.float64)
    return float(coefficient * np.sum(np.abs(parameter_array)))


def l2_penalty(parameters: ArrayLike, coefficient: float = 1.0) -> float:
    """Compute an L2 regularization penalty."""

    parameter_array = np.asarray(parameters, dtype=np.float64)
    return float(0.5 * coefficient * np.sum(parameter_array**2))


def dropout(
    values: ArrayLike,
    drop_probability: float,
    *,
    rng: np.random.Generator | None = None,
    training: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Apply inverted dropout and return the sampled mask.

    Args:
        values: Input activations.
        drop_probability: Probability of dropping an activation.
        rng: Optional NumPy generator for deterministic sampling.
        training: Whether to sample a mask or pass inputs through unchanged.

    Returns:
        Tuple ``(output, mask)``.
    """

    if not 0.0 <= drop_probability < 1.0:
        raise ValueError("drop_probability must lie in [0, 1).")
    array = np.asarray(values, dtype=np.float64)
    if not training:
        return array.copy(), np.ones_like(array)
    generator = rng if rng is not None else np.random.default_rng()
    keep_probability = 1.0 - drop_probability
    mask = generator.binomial(1, keep_probability, size=array.shape).astype(np.float64)
    return (array * mask) / keep_probability, mask


def batch_norm_forward(
    values: ArrayLike,
    gamma: ArrayLike,
    beta: ArrayLike,
    *,
    eps: float = 1e-5,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Compute the forward pass of batch normalization.

    Args:
        values: Batch of activations with shape ``(batch_size, features)``.
        gamma: Learnable scale parameter.
        beta: Learnable shift parameter.
        eps: Numerical stabilization constant.

    Returns:
        Normalized output and a cache dictionary useful for derivations.
    """

    value_array = np.asarray(values, dtype=np.float64)
    gamma_array = np.asarray(gamma, dtype=np.float64)
    beta_array = np.asarray(beta, dtype=np.float64)
    if value_array.ndim != 2:
        raise ValueError("values must be a 2D batch matrix.")
    if gamma_array.shape != (value_array.shape[1],) or beta_array.shape != (value_array.shape[1],):
        raise ValueError("gamma and beta must match the feature dimension.")

    mean = value_array.mean(axis=0)
    variance = value_array.var(axis=0)
    normalized = (value_array - mean) / np.sqrt(variance + eps)
    output = gamma_array * normalized + beta_array
    cache = {
        "mean": mean,
        "variance": variance,
        "normalized": normalized,
    }
    return output, cache


def layer_norm_forward(
    values: ArrayLike,
    gamma: ArrayLike | None = None,
    beta: ArrayLike | None = None,
    *,
    eps: float = 1e-5,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    """Compute the forward pass of layer normalization."""

    value_array = np.asarray(values, dtype=np.float64)
    if value_array.ndim < 2:
        raise ValueError("values must have at least two dimensions.")

    feature_dim = value_array.shape[-1]
    gamma_array = np.ones(feature_dim, dtype=np.float64) if gamma is None else np.asarray(gamma, dtype=np.float64)
    beta_array = np.zeros(feature_dim, dtype=np.float64) if beta is None else np.asarray(beta, dtype=np.float64)
    if gamma_array.shape != (feature_dim,) or beta_array.shape != (feature_dim,):
        raise ValueError("gamma and beta must match the feature dimension.")

    mean = value_array.mean(axis=-1, keepdims=True)
    variance = value_array.var(axis=-1, keepdims=True)
    normalized = (value_array - mean) / np.sqrt(variance + eps)
    output = normalized * gamma_array + beta_array
    cache = {
        "mean": mean,
        "variance": variance,
        "normalized": normalized,
    }
    return output, cache


def attention_scores(query: ArrayLike, key: ArrayLike) -> FloatArray:
    """Compute scaled query-key compatibility scores."""

    query_array = np.asarray(query, dtype=np.float64)
    key_array = np.asarray(key, dtype=np.float64)
    if query_array.shape[:-2] != key_array.shape[:-2]:
        raise ValueError("query and key must share batch dimensions.")
    if query_array.shape[-1] != key_array.shape[-1]:
        raise ValueError("query and key must have the same key dimension.")

    return query_array @ np.swapaxes(key_array, -1, -2) / np.sqrt(query_array.shape[-1])


def causal_attention_mask(sequence_length: int) -> NDArray[np.bool_]:
    """Create a lower-triangular causal mask."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive.")
    return np.tril(np.ones((sequence_length, sequence_length), dtype=bool))


def scaled_dot_product_attention(
    query: ArrayLike,
    key: ArrayLike,
    value: ArrayLike,
    *,
    mask: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute scaled dot-product attention.

    Args:
        query: Query tensor of shape ``(..., tokens_q, d_k)``.
        key: Key tensor of shape ``(..., tokens_k, d_k)``.
        value: Value tensor of shape ``(..., tokens_k, d_v)``.
        mask: Optional additive mask broadcastable to the attention scores.
            Masked entries should be supplied as ``0`` and valid entries as ``1``.

    Returns:
        Tuple ``(output, attention_weights)``.
    """

    query_array = np.asarray(query, dtype=np.float64)
    key_array = np.asarray(key, dtype=np.float64)
    value_array = np.asarray(value, dtype=np.float64)
    if query_array.shape[:-2] != key_array.shape[:-2]:
        raise ValueError("query and key must share batch dimensions.")
    if key_array.shape[:-2] != value_array.shape[:-2]:
        raise ValueError("key and value must share batch dimensions.")
    if query_array.shape[-1] != key_array.shape[-1]:
        raise ValueError("query and key must have the same key dimension.")
    if key_array.shape[-2] != value_array.shape[-2]:
        raise ValueError("key and value must share the same sequence length.")

    scores = attention_scores(query_array, key_array)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        scores = np.where(mask_array, scores, -1e9)
    weights = safe_softmax(scores, axis=-1)
    output = weights @ value_array
    return output, weights


def linear_beta_schedule(
    num_steps: int,
    *,
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> FloatArray:
    """Create a linear diffusion beta schedule."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if not 0.0 < beta_start < beta_end < 1.0:
        raise ValueError("betas must satisfy 0 < beta_start < beta_end < 1.")
    return np.linspace(beta_start, beta_end, num_steps, dtype=np.float64)


def diffusion_forward_process(
    clean_sample: ArrayLike,
    timestep: int,
    betas: ArrayLike,
    *,
    noise: ArrayLike | None = None,
) -> tuple[FloatArray, dict[str, Any]]:
    """Sample from the DDPM forward process q(x_t | x_0)."""

    beta_array = np.asarray(betas, dtype=np.float64)
    if beta_array.ndim != 1:
        raise ValueError("betas must be a 1D schedule.")
    if not 0 <= timestep < len(beta_array):
        raise ValueError("timestep must index the beta schedule.")

    clean_array = np.asarray(clean_sample, dtype=np.float64)
    noise_array = np.asarray(noise, dtype=np.float64) if noise is not None else np.random.default_rng().standard_normal(clean_array.shape)
    if noise_array.shape != clean_array.shape:
        raise ValueError("noise must have the same shape as clean_sample.")

    alpha_bar_t = float(np.cumprod(1.0 - beta_array)[timestep])
    signal_rate = float(np.sqrt(alpha_bar_t))
    noise_rate = float(np.sqrt(1.0 - alpha_bar_t))
    noisy_sample = signal_rate * clean_array + noise_rate * noise_array
    cache = {
        "alpha_bar_t": alpha_bar_t,
        "signal_rate": signal_rate,
        "noise_rate": noise_rate,
        "noise": noise_array,
    }
    return noisy_sample, cache


def predict_clean_from_noise(
    noisy_sample: ArrayLike,
    timestep: int,
    predicted_noise: ArrayLike,
    betas: ArrayLike,
) -> FloatArray:
    """Recover x_0 from x_t and a predicted noise sample."""

    beta_array = np.asarray(betas, dtype=np.float64)
    if beta_array.ndim != 1:
        raise ValueError("betas must be a 1D schedule.")
    if not 0 <= timestep < len(beta_array):
        raise ValueError("timestep must index the beta schedule.")

    noisy_array = np.asarray(noisy_sample, dtype=np.float64)
    predicted_noise_array = np.asarray(predicted_noise, dtype=np.float64)
    if noisy_array.shape != predicted_noise_array.shape:
        raise ValueError("noisy_sample and predicted_noise must have the same shape.")

    alpha_bar_t = float(np.cumprod(1.0 - beta_array)[timestep])
    signal_rate = np.sqrt(alpha_bar_t)
    noise_rate = np.sqrt(1.0 - alpha_bar_t)
    return (noisy_array - noise_rate * predicted_noise_array) / signal_rate


def reparameterize_gaussian(
    mean: ArrayLike,
    log_variance: ArrayLike,
    *,
    noise: ArrayLike | None = None,
) -> FloatArray:
    """Sample from a diagonal Gaussian using the reparameterization trick."""

    mean_array = np.asarray(mean, dtype=np.float64)
    log_variance_array = np.asarray(log_variance, dtype=np.float64)
    if mean_array.shape != log_variance_array.shape:
        raise ValueError("mean and log_variance must have the same shape.")
    noise_array = np.asarray(noise, dtype=np.float64) if noise is not None else np.random.default_rng().standard_normal(mean_array.shape)
    if noise_array.shape != mean_array.shape:
        raise ValueError("noise must have the same shape as mean.")
    return mean_array + np.exp(0.5 * log_variance_array) * noise_array


def vae_kl_divergence(mean: ArrayLike, log_variance: ArrayLike) -> FloatArray:
    """Compute KL(q(z|x) || N(0, I)) for a diagonal Gaussian posterior."""

    mean_array = np.asarray(mean, dtype=np.float64)
    log_variance_array = np.asarray(log_variance, dtype=np.float64)
    if mean_array.shape != log_variance_array.shape:
        raise ValueError("mean and log_variance must have the same shape.")
    return 0.5 * np.sum(np.exp(log_variance_array) + mean_array**2 - 1.0 - log_variance_array, axis=-1)


def im2col_2d(image: ArrayLike, kernel_shape: tuple[int, int], stride: int = 1) -> FloatArray:
    """Convert a 2D image into sliding patches."""

    image_array = np.asarray(image, dtype=np.float64)
    kernel_height, kernel_width = kernel_shape
    out_height = (image_array.shape[0] - kernel_height) // stride + 1
    out_width = (image_array.shape[1] - kernel_width) // stride + 1
    patches = []
    for row in range(0, out_height * stride, stride):
        for col in range(0, out_width * stride, stride):
            patch = image_array[row : row + kernel_height, col : col + kernel_width]
            patches.append(patch.ravel())
    return np.asarray(patches, dtype=np.float64)


def convolution_as_matrix_multiplication(
    image: ArrayLike,
    kernel: ArrayLike,
    *,
    stride: int = 1,
) -> FloatArray:
    """Compute valid 2D convolution via matrix multiplication."""

    image_array = np.asarray(image, dtype=np.float64)
    kernel_array = np.asarray(kernel, dtype=np.float64)
    if image_array.ndim != 2 or kernel_array.ndim != 2:
        raise ValueError("image and kernel must both be 2D arrays.")

    columns = im2col_2d(image_array, kernel_array.shape, stride=stride)
    output_vector = columns @ kernel_array.ravel()
    out_height = (image_array.shape[0] - kernel_array.shape[0]) // stride + 1
    out_width = (image_array.shape[1] - kernel_array.shape[1]) // stride + 1
    return output_vector.reshape(out_height, out_width)


def direct_valid_convolution(image: ArrayLike, kernel: ArrayLike) -> FloatArray:
    """Compute valid 2D convolution directly for verification purposes."""

    image_array = np.asarray(image, dtype=np.float64)
    kernel_array = np.asarray(kernel, dtype=np.float64)
    out_height = image_array.shape[0] - kernel_array.shape[0] + 1
    out_width = image_array.shape[1] - kernel_array.shape[1] + 1
    output = np.zeros((out_height, out_width), dtype=np.float64)
    for row in range(out_height):
        for col in range(out_width):
            window = image_array[
                row : row + kernel_array.shape[0],
                col : col + kernel_array.shape[1],
            ]
            output[row, col] = np.sum(window * kernel_array)
    return output
