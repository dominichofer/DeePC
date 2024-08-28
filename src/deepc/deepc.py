from typing import Callable
import numpy as np
from .math import hankel_matrix, projected_gradient_method


def as_column_vector(v: list | np.ndarray) -> np.ndarray:
    v = np.array(v)
    if v.ndim == 1:
        return v.reshape(-1, 1)
    return v


def check_dimensions(var: np.ndarray, name: str, size: int, dims: int) -> None:
    "Checks the dimensions of a variable."
    assert var.shape == (size, dims), f"{name}.shape={var.shape} but should be ({size}, {dims})."


def deePC(
    u_d: list | np.ndarray,
    y_d: list | np.ndarray,
    u_ini: list | np.ndarray,
    y_ini: list | np.ndarray,
    target: list | np.ndarray,
    Q: np.ndarray | int | float | None = None,
    R: np.ndarray | int | float | None = None,
    input_constrain_fkt: Callable | None = None,
    max_pgm_iterations=300,
    pgm_tolerance=1e-6,
) -> np.ndarray:
    """
    Returns the optimal control for a given system and target system outputs.
    According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    https://arxiv.org/abs/1811.05890
    Args:
        u_d: Control inputs from an offline procedure.
        y_d: System outputs from an offline procedure.
        u_ini: Control inputs to initialize the state.
        y_ini: System outputs to initialize the state.
        target: Target system outputs, optimal control tries to match.
        Q: Output cost matrix. Defaults to identity matrix.
           If int or float, diagonal matrix with this value.
        R: Control cost matrix. Defaults to zero matrix.
           If int or float, diagonal matrix with this value.
        input_constrain_fkt: Function that constrains the system inputs.
        max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                            used to solve the constrained optimization problem.
        pgm_tolerance: Tolerance for the PGM algorithm.
    """
    u_d = as_column_vector(u_d)
    y_d = as_column_vector(y_d)
    u_ini = as_column_vector(u_ini)
    y_ini = as_column_vector(y_ini)
    target = as_column_vector(target)

    offline_len = len(u_d)
    T_ini = len(u_ini)
    target_len = len(target)
    input_dims = u_d.shape[1]
    output_dims = y_d.shape[1]

    check_dimensions(u_d, "u_d", offline_len, input_dims)
    check_dimensions(y_d, "y_d", offline_len, output_dims)
    check_dimensions(u_ini, "u_ini", T_ini, input_dims)
    check_dimensions(y_ini, "y_ini", T_ini, output_dims)
    check_dimensions(target, "target", target_len, output_dims)

    Q_size = target_len * output_dims
    if isinstance(Q, (int, float)):
        Q = np.eye(Q_size) * Q
    if Q is None:
        Q = np.eye(Q_size)
    check_dimensions(Q, "Q", Q_size, Q_size)

    R_size = target_len * input_dims
    if isinstance(R, (int, float)):
        R = np.eye(R_size) * R
    if R is None:
        R = np.zeros((R_size, R_size))
    check_dimensions(R, "R", R_size, R_size)

    # Flatten
    u_ini = np.concatenate(u_ini).reshape(-1, 1)
    y_ini = np.concatenate(y_ini).reshape(-1, 1)
    target = np.concatenate(target).reshape(-1, 1)

    U = hankel_matrix(T_ini + target_len, u_d)
    U_p = U[: T_ini * input_dims, :]  # past
    U_f = U[T_ini * input_dims :, :]  # future
    Y = hankel_matrix(T_ini + target_len, y_d)
    Y_p = Y[: T_ini * output_dims, :]  # past
    Y_f = Y[T_ini * output_dims :, :]  # future

    # Now solving
    # minimize: ||y - r||_Q^2 + ||u||_R^2
    # subject to: [U_p; Y_p; U_f; Y_f] * g = [u_ini; y_ini; u; y]

    # We define
    A = np.block([[U_p], [Y_p], [U_f]])
    x = np.concatenate([u_ini, y_ini]).reshape(-1, 1)
    # to get
    # A * g = [x; u]  (1)
    # and
    # Y_f * g = y  (2).

    # We multiply (1) from the left with the pseudo inverse of A.
    # Since pinv(A) * A = I, we get g = pinv(A) * [x; u].
    # Substituting g in (2) gives Y_f * pinv(A) * [x; u] = y.

    # We define
    M = Y_f @ np.linalg.pinv(A)
    # and get M * [x; u] = y.

    # We define [M_x, M_u] := M such that M_x * x + M_u * u = y.
    M_x = M[:, : len(x)]
    M_u = M[:, len(x) :]

    # We can now solve the unconstrained problem.
    # This is a ridge regression problem with generalized Tikhonov regularization.
    # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
    # minimize: ||y - r||_Q^2 + ||u||_R^2
    # subject to: M_u * u = y - M_x * x
    # This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y).

    G = M_u.T @ Q @ M_u + R
    w = M_u.T @ Q @ (target - M_x @ x)
    u_star = np.linalg.lstsq(G, w)[0]

    if input_constrain_fkt is not None:
        u_star = projected_gradient_method(G, u_star, w, input_constrain_fkt, max_pgm_iterations, pgm_tolerance)

    return u_star.reshape(-1, input_dims)



