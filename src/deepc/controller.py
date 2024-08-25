import collections
from typing import Callable
import numpy as np
from .math import hankel_matrix, projected_gradient_method
from .deepc import as_column_vector


class Controller:
    def __init__(
        self,
        u_d: list | np.ndarray,
        y_d: list | np.ndarray,
        T_ini: int,
        target_len: int,
        u_ini: list | np.ndarray | None = None,
        y_ini: list | np.ndarray | None = None,
        Q: np.ndarray | int | float | None = None,
        R: np.ndarray | int | float | None = None,
        control_constrain_fkt: Callable | None = None,
        max_pgm_iterations=300,
        pgm_tolerance=1e-6,
    ) -> None:
        """
        Optimal controller for a given system and target system outputs.
        According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
        https://arxiv.org/abs/1811.05890
        Args:
            u_d: Control inputs from an offline procedure.
            y_d: System outputs from an offline procedure.
            T_ini: Number of initial control inputs and system outputs to initialize the state.
            target_len: Length of the target system outputs.
            Q: Output cost matrix. Defaults to identity matrix.
               If int or float, it is used as a scalar to multiply the identity matrix.
            R: Control cost matrix. Defaults to zero matrix.
                If int or float, it is used as a scalar to multiply the identity matrix.
            control_constrain_fkt: Function that constrains the control inputs.
            max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                                used to solve the constrained optimization problem.
            pgm_tolerance: Tolerance for the PGM algorithm.
        """
        assert len(u_d) == len(y_d), "u_d and y_d must have the same length."
        assert T_ini > 0, "T_ini must be greater than zero."
        assert (u_ini is None) == (y_ini is None), "u_ini and y_ini must both be None or not None."
        assert (u_ini is None) or (len(u_ini) == T_ini), "u_ini must have the same length as T_ini."
        assert (y_ini is None) or (len(y_ini) == T_ini), "y_ini must have the same length as T_ini."

        u_d = as_column_vector(u_d)
        y_d = as_column_vector(y_d)
        if u_ini is not None:
            u_ini = as_column_vector(u_ini)
        if y_ini is not None:
            y_ini = as_column_vector(y_ini)

        offline_len = len(u_d)
        self.input_dims = u_d.shape[1]
        self.output_dims = y_d.shape[1]

        assert u_d.shape == (offline_len, self.input_dims), f"{u_d.shape=} but should be ({offline_len}, {input_dims})."
        assert y_d.shape == (offline_len, self.output_dims), f"{y_d.shape=} but should be ({offline_len}, {output_dims})."
        if u_ini is not None:
            assert u_ini.shape == (T_ini, self.input_dims), f"{u_ini.shape=} but should be ({T_ini}, {input_dims})."
        if y_ini is not None:
            assert y_ini.shape == (T_ini, self.output_dims), f"{y_ini.shape=} but should be ({T_ini}, {output_dims})."

        self.T_ini = T_ini
        self.target_len = target_len

        self.u_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        if u_ini is not None:
            for u in u_ini:
                self.u_ini.append(np.array(u))

        self.y_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        if y_ini is not None:
            y_ini = as_column_vector(y_ini)
            for y in y_ini:
                self.y_ini.append(np.array(y))

        Q_size = target_len * self.output_dims
        if isinstance(Q, (int, float)):
            Q = np.eye(Q_size) * Q
        if Q is None:
            Q = np.eye(Q_size)
        assert Q.shape == (Q_size, Q_size), f"{Q.shape=} but should be ({Q_size}, {Q_size})."

        R_size = target_len * self.input_dims
        if isinstance(R, (int, float)):
            R = np.eye(R_size) * R
        if R is None:
            R = np.zeros((R_size, R_size))
        assert R.shape == (R_size, R_size), f"{R.shape=} but should be ({R_size}, {R_size})."

        self.Q = Q
        self.R = R
        self.control_constrain_fkt = control_constrain_fkt
        self.max_pgm_iterations = max_pgm_iterations
        self.pgm_tolerance = pgm_tolerance

        U = hankel_matrix(T_ini + target_len, u_d)
        U_p = U[: T_ini * self.input_dims, :]  # past
        U_f = U[T_ini * self.input_dims :, :]  # future
        Y = hankel_matrix(T_ini + target_len, y_d)
        Y_p = Y[: T_ini * self.output_dims, :]  # past
        Y_f = Y[T_ini * self.output_dims :, :]  # future

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

        # We define [M_x, M_u] := M such that M_x * x + M_u * u = y.
        self.M_x = M[:, : T_ini * (self.input_dims + self.output_dims)]
        self.M_u = M[:, T_ini * (self.input_dims + self.output_dims) :]

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

    def apply(self, target: list | np.ndarray) -> list[float] | None:
        """
        Returns the optimal control for a given reference trajectory
        or None if the controller is not initialized.
        Args:
            target: Target system outputs, optimal control tries to reach.
        """
        if not self.is_initialized():
            return None

        target = as_column_vector(np.array(target))
        assert target.shape == (self.target_len, self.output_dims), f"{target.shape=} but should be ({self.target_len}, {self.output_dims})."

        # Flatten
        target = np.concatenate(target).reshape(-1, 1)

        x = np.concatenate([self.u_ini, self.y_ini]).reshape(-1, 1)
        w = self.M_u.T @ self.Q @ (target - self.M_x @ x)
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

        return u_star.reshape(-1, self.input_dims)
