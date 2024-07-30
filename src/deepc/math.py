from typing import Callable
import numpy as np


def clamp(value, lower_bound, upper_bound):
    return np.minimum(np.maximum(value, lower_bound), upper_bound)


def left_pseudoinverse(mat: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    return np.linalg.inv(mat.T @ mat) @ mat.T


def right_pseudoinverse(mat: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    return mat.T @ np.linalg.inv(mat @ mat.T)


def hankel_matrix(rows: int, data: np.ndarray) -> np.ndarray:
    """
    Hankel matrix of a data sequence.
    Args:
        rows: number of rows in the Hankel matrix
        data: data sequence
    """
    # Generalization of https://en.wikipedia.org/wiki/Hankel_matrix to arbitrary rows.
    cols = len(data) - rows + 1
    return np.array([data[i : i + cols] for i in range(rows)])


def projected_gradient_method(
    mat: np.ndarray,
    target: np.ndarray,
    constrain: Callable,
    max_iterations=300,
    tolerance=1e-6,
) -> np.ndarray:
    """
    Projected Gradient Method
    Args:
        target: target vector
        constrain: function that constrains the result
    """
    step_size = 1 / np.linalg.matrix_norm(mat)
    old = constrain(target)
    for _ in range(max_iterations):
        new = constrain(old - step_size * (mat @ old - target))
        if np.linalg.norm(new - old) < tolerance:
            break
        old = new
    return old
