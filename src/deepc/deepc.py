import collections
from typing import Callable
import numpy as np
from .math import hankel_matrix, projected_gradient_method

def convert_to_column_vector(v: list | np.ndarray) -> np.ndarray:
    v = np.array(v)
    if v.ndim == 1:
        return v.reshape(-1, 1)
    return v

def deePC(
    u_d: list | np.ndarray,
    y_d: list | np.ndarray,
    u_ini: list | np.ndarray,
    y_ini: list | np.ndarray,
    r: list | np.ndarray,
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    control_constrain_fkt: Callable | None = None,
    max_pgm_iterations=300,
    pgm_tolerance=1e-6,
) -> np.ndarray:
    """
    Returns the optimal control for a given system and reference trajectory.
    According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    https://arxiv.org/abs/1811.05890
    Args:
        u_d: Control inputs from an offline procedure.
        y_d: Outputs from an offline procedure.
        u_ini: Control inputs to initiate the state.
        y_ini: Outputs to initiate the state.
        r: Reference trajectory.
        Q: Output cost matrix, defaults to identity matrix.
        R: Control cost matrix, defaults to zero matrix.
        control_constrain_fkt: Function that constrains the control.
        max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                            used to solve the constrained optimization problem.
        pgm_tolerance: Tolerance for the PGM algorithm.
    """
    u_d = convert_to_column_vector(u_d)
    y_d = convert_to_column_vector(y_d)
    u_ini = convert_to_column_vector(u_ini)
    y_ini = convert_to_column_vector(y_ini)
    r = convert_to_column_vector(r)

    offline_size = len(u_d)
    T_ini = len(u_ini)
    target_size = len(r)
    input_dims = u_d.shape[1]
    output_dims = y_d.shape[1]

    assert u_d.shape == (offline_size, input_dims), f"{u_d.shape=} but should be ({offline_size}, {input_dims})."
    assert y_d.shape == (offline_size, output_dims), f"{y_d.shape=} but should be ({offline_size}, {output_dims})."
    assert u_ini.shape == (T_ini, input_dims), f"{u_ini.shape=} but should be ({T_ini}, {input_dims})."
    assert y_ini.shape == (T_ini, output_dims), f"{y_ini.shape=} but should be ({T_ini}, {output_dims})."
    assert r.shape == (target_size, output_dims), f"{r.shape=} but should be ({target_size}, {output_dims})."

    Q_size = target_size * output_dims
    if Q is None:
        Q = np.eye(Q_size)
    assert Q.shape == (Q_size, Q_size), f"{Q.shape=} but should be ({Q_size}, {Q_size})."

    R_size = target_size * input_dims
    if R is None:
        R = np.zeros((R_size, R_size))
    assert R.shape == (R_size, R_size), f"{R.shape=} but should be ({R_size}, {R_size})."

    # Flatten
    u_ini = np.concatenate(u_ini).reshape(-1, 1)
    y_ini = np.concatenate(y_ini).reshape(-1, 1)
    r = np.concatenate(r).reshape(-1, 1)

    U = hankel_matrix(T_ini + target_size, u_d)
    U_p = U[: T_ini * input_dims, :]  # past
    U_f = U[T_ini * input_dims :, :]  # future
    Y = hankel_matrix(T_ini + target_size, y_d)
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

    # We define [M_x; M_u] := M
    # such that M_x * x + M_u * u = y.
    M_x = M[:, : len(x)]
    M_u = M[:, len(x) :]

    # We can now solve the unconstrained problem.
    # This is a ridge regression problem with generalized Tikhonov regularization.
    # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
    # minimize: ||y - r||_Q^2 + ||u||_R^2
    # subject to: M_u * u = y - M_x * x

    G = M_u.T @ Q @ M_u + R
    w = M_u.T @ Q @ (r - M_x @ x)
    u_star = np.linalg.lstsq(G, w)[0]

    if control_constrain_fkt is not None:
        u_star = projected_gradient_method(G, u_star, w, control_constrain_fkt, max_pgm_iterations, pgm_tolerance)

    return u_star.reshape(-1, input_dims)


class Controller:
    def __init__(
        self,
        u_d: list | np.ndarray,
        y_d: list | np.ndarray,
        T_ini: int,
        r_len: int,
        u_ini: list | np.ndarray | None = None,
        y_ini: list | np.ndarray | None = None,
        Q: np.ndarray | None = None,
        R: np.ndarray | None = None,
        control_constrain_fkt: Callable | None = None,
        max_pgm_iterations=300,
        pgm_tolerance=1e-6,
    ) -> None:
        """
        Optimal controller for a given system and reference trajectory.
        According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
        https://arxiv.org/abs/1811.05890
        Holds the last T_ini control inputs and trajectories to initiate the state.
        Args:
            u_d: Control inputs from an offline procedure.
            y_d: Outputs from an offline procedure.
            T_ini: Number of initial control inputs and trajectories.
            r_len: Length of the reference trajectory.
            Q: Output cost matrix, defaults to identity matrix.
            R: Control cost matrix, defaults to zero matrix.
            control_constrain_fkt: Function that constrains the control.
            max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                                used to solve the constrained optimization problem.
            pgm_tolerance: Tolerance for the PGM algorithm.
        """
        assert len(u_d) == len(y_d), "u_d and y_d must have the same length."
        assert T_ini > 0, "T_ini must be greater than zero."
        assert (u_ini is None) == (y_ini is None), "u_ini and y_ini must be both None or not None."
        assert u_ini is None or len(u_ini) == T_ini, "u_ini must have the same length as T_ini."
        assert y_ini is None or len(y_ini) == T_ini, "y_ini must have the same length as T_ini."

        u_d = np.array(u_d)
        self.u_is_1d = u_d.ndim == 1
        if u_d.ndim == 1:
            u_d = u_d.reshape(-1, 1)
        y_d = np.array(y_d)
        if y_d.ndim == 1:
            y_d = y_d.reshape(-1, 1)

        u_ndim = u_d.shape[1] if u_d.ndim == 2 else 1
        y_ndim = y_d.shape[1] if y_d.ndim == 2 else 1

        self.T_ini = T_ini
        self.r_len = r_len
        self.u_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        if u_ini is not None:
            u_ini = np.array(u_ini)
            if u_ini.ndim == 1:
                u_ini = u_ini.reshape(-1, 1)
            for u in u_ini:
                self.u_ini.append(np.array(u))
        self.y_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        if y_ini is not None:
            y_ini = np.array(y_ini)
            if y_ini.ndim == 1:
                y_ini = y_ini.reshape(-1, 1)
            for y in y_ini:
                self.y_ini.append(np.array(y))
        if Q is None:
            Q = np.eye(r_len * y_ndim)
        if R is None:
            R = np.zeros((r_len * u_ndim, r_len * u_ndim))
        self.Q = Q
        self.R = R
        self.control_constrain_fkt = control_constrain_fkt
        self.max_pgm_iterations = max_pgm_iterations
        self.pgm_tolerance = pgm_tolerance

        U = hankel_matrix(T_ini + r_len, u_d)
        U_p = U[: T_ini * u_ndim, :]  # past
        U_f = U[T_ini * u_ndim :, :]  # future
        Y = hankel_matrix(T_ini + r_len, y_d)
        Y_p = Y[: T_ini * y_ndim, :]  # past
        Y_f = Y[T_ini * y_ndim :, :]  # future

        # Now solving
        # minimize: ||y - r||_Q^2 + ||u||_R^2
        # subject to: [U_p; Y_p; U_f; Y_f] * g = [u_ini; y_ini; u; y]

        # We define
        A = np.block([[U_p], [Y_p], [U_f]])
        # x = [u_ini; y_ini]
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

        # We define (M_x, M_u) := M such that M_x * x + M_u * u = y.
        self.M_x = M[:, : T_ini * (u_ndim + y_ndim)]
        self.M_u = M[:, T_ini * (u_ndim + y_ndim) :]

        # We can now solve the unconstrained problem.
        # This is a ridge regression problem with generalized Tikhonov regularization.
        # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
        # minimize: ||y - r||_Q^2 + ||u||_R^2
        # subject to: M_u * u = y - M_x * x
        # This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y).

        # We precompute the matrix G = M_u^T * Q * M_u + R.
        self.G = self.M_u.T @ self.Q @ self.M_u + self.R

    def is_initialized(self) -> bool:
        "Returns whether the internal state is initialized."
        return len(self.u_ini) == self.T_ini and len(self.y_ini) == self.T_ini

    def update(self, u: list | np.ndarray, y: list | np.ndarray) -> None:
        "Updates the internal state with the given control input and trajectory."
        self.u_ini.append(np.array(u))
        self.y_ini.append(np.array(y))

    def clear(self) -> None:
        "Clears the internal state."
        self.u_ini.clear()
        self.y_ini.clear()

    def apply(self, r: list | np.ndarray) -> list[float] | None:
        """
        Returns the optimal control for a given reference trajectory.
        Args:
            r: Reference trajectory.
        """
        if not self.is_initialized():
            return None

        assert len(r) == self.r_len, "Reference trajectory has wrong length."

        # Transform to column vectors
        r = np.array(r).reshape(-1, 1)

        x = np.concatenate([self.u_ini, self.y_ini]).reshape(-1, 1)
        w = self.M_u.T @ self.Q @ (r - self.M_x @ x)
        u_star = np.linalg.lstsq(self.G, w)[0]

        if self.control_constrain_fkt is not None:
            u_star = projected_gradient_method(
                self.G,
                u_star,
                w,
                self.control_constrain_fkt,
                self.max_pgm_iterations,
                self.pgm_tolerance,
            )

        if self.u_is_1d:
            u_star = u_star[:, 0]
        else:
            u_star = u_star.reshape(-1, self.u_ini[0].shape[0])
        return u_star.tolist()
