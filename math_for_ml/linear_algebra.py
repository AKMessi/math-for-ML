"""Linear algebra utilities used throughout the curriculum."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
__all__ = [
    "PCAResult",
    "cholesky_decomposition",
    "cosine_similarity",
    "dot",
    "eigen_decomposition_symmetric",
    "gram_matrix",
    "lu_decomposition",
    "matrix_multiply",
    "pairwise_distances",
    "pca_from_scratch",
    "projection",
    "qr_decomposition",
    "singular_value_decomposition",
    "solve_linear_system",
    "vector_norm",
]


@dataclass(frozen=True)
class PCAResult:
    """Container for the result of principal component analysis.

    Attributes:
        components: Principal directions with shape
            ``(n_components, n_features)``.
        singular_values: Singular values associated with each component.
        explained_variance: Variance explained by each component.
        explained_variance_ratio: Fraction of variance explained.
        scores: Coordinates of the input data in the principal subspace.
        mean: Mean used to center the data.
    """

    components: FloatArray
    singular_values: FloatArray
    explained_variance: FloatArray
    explained_variance_ratio: FloatArray
    scores: FloatArray
    mean: FloatArray


def _to_float_array(values: ArrayLike, *, ndim: int | None = None) -> FloatArray:
    """Convert an input into a floating-point NumPy array.

    Args:
        values: Input values.
        ndim: Expected number of dimensions when validation is required.

    Returns:
        A ``float64`` NumPy array.

    Raises:
        ValueError: If ``ndim`` is supplied and the array has the wrong rank.
    """

    array = np.asarray(values, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"Expected {ndim} dimensions, received {array.ndim}.")
    return array


def dot(left: ArrayLike, right: ArrayLike) -> float:
    """Compute the Euclidean dot product of two vectors.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        The scalar dot product.

    Raises:
        ValueError: If the vectors do not share the same shape.

    Examples:
        >>> dot([1, 2], [3, 4])
        11.0
    """

    left_array = _to_float_array(left, ndim=1)
    right_array = _to_float_array(right, ndim=1)
    if left_array.shape != right_array.shape:
        raise ValueError("Dot product requires vectors with the same shape.")
    return float(np.dot(left_array, right_array))


def cosine_similarity(left: ArrayLike, right: ArrayLike) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        left: First vector.
        right: Second vector.

    Returns:
        The cosine of the angle between the vectors.

    Raises:
        ValueError: If either vector has zero magnitude.

    Examples:
        >>> round(cosine_similarity([1, 0], [1, 1]), 5)
        0.70711
    """

    left_array = _to_float_array(left, ndim=1)
    right_array = _to_float_array(right, ndim=1)
    denominator = vector_norm(left_array) * vector_norm(right_array)
    if denominator == 0.0:
        raise ValueError("Cosine similarity is undefined for the zero vector.")
    return dot(left_array, right_array) / denominator


def projection(vector: ArrayLike, onto: ArrayLike) -> FloatArray:
    """Project one vector onto another.

    Args:
        vector: Vector to project.
        onto: Direction that defines the target subspace.

    Returns:
        The projection of ``vector`` onto ``onto``.

    Raises:
        ValueError: If ``onto`` is the zero vector.

    Examples:
        >>> projection([2, 2], [1, 0]).tolist()
        [2.0, 0.0]
    """

    vector_array = _to_float_array(vector, ndim=1)
    onto_array = _to_float_array(onto, ndim=1)
    denominator = dot(onto_array, onto_array)
    if denominator == 0.0:
        raise ValueError("Cannot project onto the zero vector.")
    coefficient = dot(vector_array, onto_array) / denominator
    return coefficient * onto_array


def gram_matrix(matrix: ArrayLike) -> FloatArray:
    """Return the Gram matrix ``X X^T`` of a 2D array.

    Args:
        matrix: Matrix whose rows are vectors.

    Returns:
        A positive semidefinite Gram matrix.

    Examples:
        >>> gram_matrix([[1, 0], [0, 1]]).tolist()
        [[1.0, 0.0], [0.0, 1.0]]
    """

    array = _to_float_array(matrix, ndim=2)
    return array @ array.T


def matrix_multiply(left: ArrayLike, right: ArrayLike) -> FloatArray:
    """Multiply two matrices with validation.

    Args:
        left: Left matrix.
        right: Right matrix.

    Returns:
        The product ``left @ right``.

    Raises:
        ValueError: If the inner dimensions do not match.
    """

    left_array = _to_float_array(left, ndim=2)
    right_array = _to_float_array(right, ndim=2)
    if left_array.shape[1] != right_array.shape[0]:
        raise ValueError("Inner dimensions must agree for matrix multiplication.")
    return left_array @ right_array


def solve_linear_system(matrix: ArrayLike, rhs: ArrayLike) -> FloatArray:
    """Solve a square linear system ``Ax = b``.

    Args:
        matrix: Coefficient matrix ``A``.
        rhs: Right-hand side ``b``.

    Returns:
        The solution vector or matrix.

    Raises:
        ValueError: If the system matrix is not square.
    """

    array = _to_float_array(matrix, ndim=2)
    rhs_array = _to_float_array(rhs)
    if array.shape[0] != array.shape[1]:
        raise ValueError("Linear systems require a square coefficient matrix.")
    return np.linalg.solve(array, rhs_array)


def lu_decomposition(matrix: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the LU decomposition with partial pivoting.

    Args:
        matrix: Square matrix to factorize.

    Returns:
        A tuple ``(P, L, U)`` satisfying ``P @ matrix = L @ U``.

    Raises:
        ValueError: If the matrix is not square or is numerically singular.

    Examples:
        >>> p, l, u = lu_decomposition([[2, 1], [4, 3]])
        >>> np.allclose(p @ np.array([[2., 1.], [4., 3.]]), l @ u)
        True
    """

    array = _to_float_array(matrix, ndim=2)
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("LU decomposition requires a square matrix.")

    u = array.copy()
    l = np.zeros_like(array)
    p = np.eye(rows, dtype=np.float64)

    for pivot_index in range(rows):
        pivot_row = pivot_index + int(np.argmax(np.abs(u[pivot_index:, pivot_index])))
        if np.isclose(u[pivot_row, pivot_index], 0.0):
            raise ValueError("Matrix is singular to working precision.")
        if pivot_row != pivot_index:
            u[[pivot_index, pivot_row]] = u[[pivot_row, pivot_index]]
            p[[pivot_index, pivot_row]] = p[[pivot_row, pivot_index]]
            if pivot_index > 0:
                l[[pivot_index, pivot_row], :pivot_index] = l[
                    [pivot_row, pivot_index], :pivot_index
                ]
        l[pivot_index, pivot_index] = 1.0
        for row in range(pivot_index + 1, rows):
            multiplier = u[row, pivot_index] / u[pivot_index, pivot_index]
            l[row, pivot_index] = multiplier
            u[row, pivot_index:] -= multiplier * u[pivot_index, pivot_index:]
            u[row, pivot_index] = 0.0

    return p, l, u


def qr_decomposition(matrix: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Compute the QR decomposition using modified Gram-Schmidt.

    Args:
        matrix: Matrix to decompose.

    Returns:
        A tuple ``(Q, R)`` where ``Q`` has orthonormal columns and
        ``matrix = Q @ R``.

    Raises:
        ValueError: If the input has more columns than rows or contains
            linearly dependent columns.
    """

    array = _to_float_array(matrix, ndim=2)
    rows, cols = array.shape
    if rows < cols:
        raise ValueError("QR decomposition requires rows >= columns.")

    q = np.zeros((rows, cols), dtype=np.float64)
    r = np.zeros((cols, cols), dtype=np.float64)

    for column in range(cols):
        vector = array[:, column].copy()
        for previous in range(column):
            r[previous, column] = np.dot(q[:, previous], vector)
            vector -= r[previous, column] * q[:, previous]
        norm = np.linalg.norm(vector)
        if np.isclose(norm, 0.0):
            raise ValueError("Columns must be linearly independent.")
        r[column, column] = norm
        q[:, column] = vector / norm

    return q, r


def cholesky_decomposition(matrix: ArrayLike) -> FloatArray:
    """Compute the Cholesky factor of a symmetric positive definite matrix.

    Args:
        matrix: Symmetric positive definite matrix.

    Returns:
        Lower-triangular matrix ``L`` such that ``matrix = L @ L.T``.

    Raises:
        ValueError: If the matrix is not square, symmetric, or positive definite.
    """

    array = _to_float_array(matrix, ndim=2)
    rows, cols = array.shape
    if rows != cols:
        raise ValueError("Cholesky decomposition requires a square matrix.")
    if not np.allclose(array, array.T):
        raise ValueError("Cholesky decomposition requires a symmetric matrix.")

    lower = np.zeros_like(array)
    for row in range(rows):
        for col in range(row + 1):
            correction = np.dot(lower[row, :col], lower[col, :col])
            if row == col:
                diagonal = array[row, row] - correction
                if diagonal <= 0.0:
                    raise ValueError("Matrix must be positive definite.")
                lower[row, col] = np.sqrt(diagonal)
            else:
                lower[row, col] = (array[row, col] - correction) / lower[col, col]
    return lower


def eigen_decomposition_symmetric(matrix: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Compute the eigen-decomposition of a symmetric matrix.

    Args:
        matrix: Symmetric matrix.

    Returns:
        Eigenvalues and eigenvectors sorted in descending order by eigenvalue.

    Raises:
        ValueError: If the matrix is not square and symmetric.
    """

    array = _to_float_array(matrix, ndim=2)
    if array.shape[0] != array.shape[1]:
        raise ValueError("Eigen-decomposition requires a square matrix.")
    if not np.allclose(array, array.T):
        raise ValueError("This helper only supports symmetric matrices.")
    eigenvalues, eigenvectors = np.linalg.eigh(array)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def singular_value_decomposition(
    matrix: ArrayLike,
    *,
    full_matrices: bool = False,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute the singular value decomposition of a matrix.

    Args:
        matrix: Matrix to decompose.
        full_matrices: Whether to request the full SVD.

    Returns:
        ``(U, S, Vt)`` from ``numpy.linalg.svd``.
    """

    array = _to_float_array(matrix, ndim=2)
    u, singular_values, vt = np.linalg.svd(array, full_matrices=full_matrices)
    return u, singular_values, vt


def pca_from_scratch(
    data: ArrayLike,
    n_components: int,
    *,
    center: bool = True,
) -> PCAResult:
    """Run principal component analysis using the SVD of centered data.

    Args:
        data: Data matrix with shape ``(n_samples, n_features)``.
        n_components: Number of components to retain.
        center: Whether to subtract the empirical mean.

    Returns:
        A :class:`PCAResult` object with components, scores, and variance data.

    Raises:
        ValueError: If ``n_components`` lies outside the valid range.

    Examples:
        >>> result = pca_from_scratch([[0, 1], [1, 0], [2, 1]], n_components=1)
        >>> result.components.shape
        (1, 2)
    """

    array = _to_float_array(data, ndim=2)
    samples, features = array.shape
    if not 1 <= n_components <= min(samples, features):
        raise ValueError("n_components must be between 1 and min(n_samples, n_features).")

    mean = array.mean(axis=0) if center else np.zeros(features, dtype=np.float64)
    centered = array - mean
    _, singular_values, vt = singular_value_decomposition(centered, full_matrices=False)
    components = vt[:n_components]
    scores = centered @ components.T
    explained_variance = (singular_values**2) / max(samples - 1, 1)
    total_variance = explained_variance.sum()
    explained_variance_ratio = explained_variance / total_variance if total_variance else explained_variance

    return PCAResult(
        components=components,
        singular_values=singular_values[:n_components],
        explained_variance=explained_variance[:n_components],
        explained_variance_ratio=explained_variance_ratio[:n_components],
        scores=scores[:, :n_components],
        mean=mean,
    )


def vector_norm(
    vector: ArrayLike,
    ord: Literal[1, 2, np.inf, -np.inf] | float | None = 2,
) -> float:
    """Compute a vector norm.

    Args:
        vector: Input vector.
        ord: Norm order passed to ``numpy.linalg.norm``.

    Returns:
        The requested norm.
    """

    array = _to_float_array(vector)
    return float(np.linalg.norm(array, ord=ord))


def pairwise_distances(
    left: ArrayLike,
    right: ArrayLike | None = None,
    *,
    metric: Literal["euclidean", "sqeuclidean", "manhattan"] = "euclidean",
) -> FloatArray:
    """Compute pairwise distances between rows of two matrices.

    Args:
        left: Left data matrix of shape ``(n_samples_left, n_features)``.
        right: Optional right data matrix. If omitted, distances within
            ``left`` are returned.
        metric: Distance metric to use.

    Returns:
        A matrix of pairwise distances.

    Raises:
        ValueError: If the metric is unknown or feature dimensions mismatch.
    """

    left_array = _to_float_array(left, ndim=2)
    right_array = left_array if right is None else _to_float_array(right, ndim=2)
    if left_array.shape[1] != right_array.shape[1]:
        raise ValueError("Distance computations require matching feature dimensions.")

    differences = left_array[:, None, :] - right_array[None, :, :]
    if metric == "euclidean":
        return np.linalg.norm(differences, axis=-1)
    if metric == "sqeuclidean":
        return np.sum(differences**2, axis=-1)
    if metric == "manhattan":
        return np.sum(np.abs(differences), axis=-1)
    raise ValueError(f"Unsupported metric: {metric}")
