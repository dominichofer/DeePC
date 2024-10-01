from typing import Callable
import numpy as np
import matplotlib.pyplot as plt
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
    u_0: list | np.ndarray | None = None,
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
        u_0: Control input offset, defaults to zero vector.
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

    if u_0 is None:
        u_0 = np.zeros((target_len, input_dims))
    else:
        u_0 = as_column_vector(u_0)
    check_dimensions(u_0, "u_0", target_len, input_dims)

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
    u_0 = np.concatenate(u_0).reshape(-1, 1)

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
    # This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y + R * u_0).

    G = M_u.T @ Q @ M_u + R
    w = M_u.T @ Q @ (target - M_x @ x) + R @ u_0
    u_star = np.linalg.lstsq(G, w)[0]

    if input_constrain_fkt is not None:
        u_star = projected_gradient_method(G, u_star, w, input_constrain_fkt, max_pgm_iterations, pgm_tolerance)

    return u_star.reshape(-1, input_dims)



def dominant_dim_and_find_max_drop(matrix: np.ndarray, name):

    # this should show the main eigenvalues and have one large drop after which it should be cut off... should... 

    # Perform Singular Value Decomposition (SVD)
    s = np.linalg.svd(matrix, compute_uv=False)

    # Calculate energy retained
    energy_retained = np.cumsum(s**2) / np.sum(s**2)
    
    # Calculate the difference between consecutive singular values
    diff = np.diff(s)
    
    # Find the largest drop in singular values
    largest_drop = np.max(np.abs(diff))
    print(f"Largest drop between consecutive singular values: {largest_drop:.4f}")

    # Plot the singular values
    plt.figure(figsize=(10, 6))
    plt.plot(s, marker='o', label="Singular Values")
    plt.title('Singular Values Series:' + name)
    plt.xlabel('Index')
    plt.ylabel('Singular Value / largest drop = '+ str(largest_drop))
    plt.legend()
    plt.grid(True)



    # Return the number of dimensions needed to retain 99% of the energy
    return np.searchsorted(energy_retained, 0.90) + 1, largest_drop


def data_quality(
    u_d: list | np.ndarray,
    y_d: list | np.ndarray,
    T_ini: int,
    target_len: int,
    Q: np.ndarray | int | float | None = None,
    R: np.ndarray | int | float | None = None,
):
    u_d = as_column_vector(u_d)
    y_d = as_column_vector(y_d)

    offline_len = len(u_d)
    input_dims = u_d.shape[1]
    output_dims = y_d.shape[1]

    check_dimensions(u_d, "u_d", offline_len, input_dims)
    check_dimensions(y_d, "y_d", offline_len, output_dims)

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

    U = hankel_matrix(T_ini + target_len, u_d)
    U_p = U[: T_ini * input_dims, :]  # past
    U_f = U[T_ini * input_dims :, :]  # future
    Y = hankel_matrix(T_ini + target_len, y_d)
    Y_p = Y[: T_ini * output_dims, :]  # past
    Y_f = Y[T_ini * output_dims :, :]  # future

    A = np.block([[U_p], [Y_p], [U_f]])

    M = Y_f @ np.linalg.pinv(A)

    dim_sum = input_dims + output_dims
    M_x = M[:, : T_ini * dim_sum]
    M_u = M[:, T_ini * dim_sum :]

    G = M_u.T @ Q @ M_u + R

    print(f"Dominant Dimensions and max drop for U: {dominant_dim_and_find_max_drop(U, 'U')}")
    print(f"Dominant Dimensions and max drop for Y: {dominant_dim_and_find_max_drop(Y,'Y')}")

    print(f"Shape of U_p: {U_p.shape}")
    print(f"Rank of U_p: {np.linalg.matrix_rank(U_p)}")
    print(f"Condition number of U_p: {np.linalg.cond(U_p)}")

    print(f"Shape of U_f: {U_f.shape}")
    print(f"Rank of U_f: {np.linalg.matrix_rank(U_f)}")
    print(f"Condition number of U_f: {np.linalg.cond(U_f)}")

    print(f"Shape of Y_p: {Y_p.shape}")
    print(f"Rank of Y_p: {np.linalg.matrix_rank(Y_p)}")
    print(f"Condition number of Y_p: {np.linalg.cond(Y_p)}")

    print(f"Shape of Y_f: {Y_f.shape}")
    print(f"Rank of Y_f: {np.linalg.matrix_rank(Y_f)}")
    print(f"Condition number of Y_f: {np.linalg.cond(Y_f)}")

    print(f"Shape of A: {A.shape}")
    print(f"Rank of A: {np.linalg.matrix_rank(A)}")
    print(f"Condition number of A: {np.linalg.cond(A)}")

    print(f"Shape of M: {M.shape}")
    print(f"Rank of M: {np.linalg.matrix_rank(M)}")
    print(f"Condition number of M: {np.linalg.cond(M)}")

    print(f"Shape of G: {G.shape}")
    print(f"Rank of G: {np.linalg.matrix_rank(G)}")
    print(f"Condition number of G: {np.linalg.cond(G)}")

