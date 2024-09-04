import math.py
from typing import Callable
import numpy as np


def left_pseudoinverse(mat: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    return np.linalg.inv(mat.T @ mat) @ mat.T


def right_pseudoinverse(mat: np.ndarray) -> np.ndarray:
    # https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse
    return mat.T @ np.linalg.inv(mat @ mat.T)


def hankel_matrix(rows: int, data: list | np.ndarray) -> np.ndarray:
    """
    Hankel matrix of a data sequence.
    Returns one row per dimension of the data.
    Args:
        rows: number of rows in the Hankel matrix
        data: data sequence
    """
    # Generalization of https://en.wikipedia.org/wiki/Hankel_matrix
    # to arbitrary rows and dimensions.
    #print("dim " ,data.ndim)
    data = np.array(data)
    cols = len(data) - rows + 1
    if data.ndim == 1:
        return np.array([data[i : i + cols] for i in range(rows)])
    if data.ndim == 2:
        return np.array([data[i : i + cols, j] for i in range(rows) for j in range(data.shape[1])])
    raise ValueError("Data must be 1D or 2D.")


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
        target: target vector
        constrain: function that constrains the result
    """
    if hasattr(np.linalg, 'matrix_norm'):
        step_size = 1 / np.linalg.matrix_norm(mat)
    else:
        step_size = 1 / np.linalg.norm(mat)
    x_old = constrain(x_ini)
    for _ in range(max_iterations):
        x_new = constrain(x_old - step_size * (mat @ x_old - target))
        if np.linalg.norm(x_new - x_old) < tolerance:
            return x_new
        x_old = x_new
    return x_old



def linear_chirp(f0: float, f1: float, samples: int, phi: float = 0) -> list[float]:
    """
    Generate a linear chirp signal.
    Args:
        f0: Start frequency in Hz.
        f1: End frequency in Hz.
        samples: Number of samples.
        phi: Phase offset in radians.
    """
    return [
        math.sin(
            phi
            + 2 * math.pi * (f0 * (i / (samples - 1)) + 0.5 * (f1 - f0) * (i / (samples - 1)) ** 2)
        )
        for i in range(samples)
    ]
