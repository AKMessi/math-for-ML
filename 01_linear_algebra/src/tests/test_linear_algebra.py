"""Tests for linear algebra helpers."""

import numpy as np

from math_for_ml.linear_algebra import (
    cholesky_decomposition,
    dot,
    lu_decomposition,
    pairwise_distances,
    pca_from_scratch,
    projection,
    qr_decomposition,
)


def test_dot_and_projection_match_expected_geometry() -> None:
    """Projection should land on the target direction."""

    vector = np.array([2.0, 2.0])
    onto = np.array([1.0, 0.0])
    assert dot(vector, onto) == 2.0
    assert np.allclose(projection(vector, onto), np.array([2.0, 0.0]))


def test_lu_decomposition_reconstructs_matrix() -> None:
    """Pivoted LU should reconstruct the original matrix."""

    matrix = np.array([[0.0, 2.0], [1.0, 3.0]])
    permutation, lower, upper = lu_decomposition(matrix)
    assert np.allclose(permutation @ matrix, lower @ upper)


def test_qr_and_cholesky_are_consistent() -> None:
    """QR and Cholesky should reconstruct their inputs."""

    matrix = np.array([[2.0, 1.0], [1.0, 2.0], [0.0, 1.0]])
    q_mat, r_mat = qr_decomposition(matrix)
    assert np.allclose(q_mat @ r_mat, matrix)

    spd = np.array([[4.0, 2.0], [2.0, 3.0]])
    lower = cholesky_decomposition(spd)
    assert np.allclose(lower @ lower.T, spd)


def test_pca_from_scratch_returns_rank_one_structure() -> None:
    """PCA should identify a dominant one-dimensional trend."""

    samples = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    result = pca_from_scratch(samples, n_components=1)
    assert result.components.shape == (1, 2)
    assert result.scores.shape == (4, 1)
    assert np.isclose(result.explained_variance_ratio.sum(), 1.0)


def test_pairwise_distances_are_symmetric() -> None:
    """Distance matrix should be symmetric with zero diagonal."""

    points = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    distances = pairwise_distances(points)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
