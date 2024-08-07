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
    x_ini: np.ndarray,
    target: np.ndarray,
    constrain: Callable,
    max_iterations=300,
    tolerance=1e-6,
) -> np.ndarray:
    """
    Projected Gradient Method
    Args:
        mat: Matrix used in the gradient step
        target: Target vector
        constrain: Function that constrains the result
        max_iterations: Maximum number of iterations
        tolerance: Tolerance for convergence
    Returns:
        The constrained optimal control vector
    """
    step_size = 1 / np.linalg.norm(mat)
    old = constrain(x_ini)
    for _ in range(max_iterations):
        new = constrain(old - step_size * (mat @ old - target))
        if np.linalg.norm(new - old) < tolerance:
            break
        old = new
    return old
