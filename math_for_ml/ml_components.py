"""Machine-learning building blocks derived from math first principles."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from math_for_ml.numerical import safe_softmax

FloatArray = NDArray[np.float64]
__all__ = [
    "batch_norm_forward",
    "convolution_as_matrix_multiplication",
    "cross_entropy_loss",
    "direct_valid_convolution",
    "dropout",
    "focal_loss_binary",
    "im2col_2d",
    "l1_penalty",
    "l2_penalty",
    "mean_squared_error",
    "scaled_dot_product_attention",
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

    scale = np.sqrt(query_array.shape[-1])
    scores = query_array @ np.swapaxes(key_array, -1, -2) / scale
    if mask is not None:
        mask_array = np.asarray(mask)
        scores = np.where(mask_array, scores, -1e9)
    weights = safe_softmax(scores, axis=-1)
    output = weights @ value_array
    return output, weights


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
