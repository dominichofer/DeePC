import collections
from typing import Callable
import numpy as np
from .math import hankel_matrix, projected_gradient_method

from numpy.linalg import inv

def deePC(
    _u_d: list[int | float],
    _y_d: list[int | float],
    _u_ini: list[int | float],
    _y_ini: list[int | float],
    _r: list[int | float],
    Q: np.ndarray | None = None,
    R: np.ndarray | None = None,
    control_constrain_fkt: Callable | None = None,
    max_pgm_iterations=300,
    pgm_tolerance=1e-6,
) -> list[int | float]:
    """
    Returns the optimal control for a given system and reference trajectory.
    According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    https://arxiv.org/abs/1811.05890
    Args:
        _u_d: Control inputs from an offline procedure.
        _y_d: Outputs from an offline procedure.
        _u_ini: Control inputs to initialize the state.
        _y_ini: Trajectories to initialize the state.
        _r: Reference output trajectory.
        Q: Output cost matrix, defaults to identity matrix.
        R: Control cost matrix, defaults to zero matrix.
        control_constrain_fkt: Function that constrains the control.
        max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                            used to solve the constrained optimization problem.
        pgm_tolerance: Tolerance for the PGM algorithm.
    """
    assert len(_u_d) == len(_y_d), "u_d and y_d must have the same length."
    assert len(_u_ini) == len(_y_ini), "u_ini and y_ini must have the same length."

    u_d = np.array(_u_d)
    y_d = np.array(_y_d)
    u_ini = np.array(_u_ini)
    y_ini = np.array(_y_ini)
    r = np.array(_r)
    T_ini = len(u_ini)

    if Q is None:
        Q = np.eye(len(r))
    if R is None:
        R = np.zeros((len(r), len(r)))

    U = hankel_matrix(T_ini + len(r), u_d)
    U_p = U[:T_ini, :]  # past
    U_f = U[T_ini:, :]  # future
    Y = hankel_matrix(T_ini + len(r), y_d)
    Y_p = Y[:T_ini, :]  # past
    Y_f = Y[T_ini:, :]  # future

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
    M_x = M[:, : 2 * T_ini]
    M_u = M[:, 2 * T_ini :]
    # to get M_u_ini * u_ini + M_y_ini * y_ini + M_u * u = y.

    # TODO: Delete this!
    # B = np.zeros_like(M_u_ini)
    # B[:, : M_u.shape[1]] = M_u
    # u_bar = np.linalg.solve(B + M_u_ini, r - M_y_ini @ r[:T_ini])
    # if not np.allclose(M_u_ini @ u_bar[:T_ini] + M_y_ini @ r[:T_ini] + M_u @ u_bar, r):
    #     raise ValueError("The solution is not correct.")

    # We can now solve the unconstrained problem.
    # This is a ridge regression problem with generalized Tikhonov regularization.
    # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
    # minimize: ||y - r||_Q^2 + ||u||_R^2
    # subject to: M_u * u = y - M_x * x

    G = M_u.T @ Q @ M_u + R
    w = M_u.T @ Q @ (r - M_x @ x)
    u_star = np.linalg.solve(G, w)

    if control_constrain_fkt is not None:
        u_star = projected_gradient_method(
            G, u_star, w, control_constrain_fkt, max_pgm_iterations, pgm_tolerance
        )
    return u_star[:, 0].tolist()


class Controller:
    def __init__(
        self,
        _u_d: list[int | float],
        _y_d: list[int | float],
        T_ini: int,
        r_len: int,
        u_ini: list[int | float] | None = None,
        y_ini: list[int | float] | None = None,
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
            T_ini: Number of initial control inputs and outputs / trajectories.
            r_len: Length of the reference trajectory.
            Q: Output cost matrix, defaults to identity matrix.
            R: Control cost matrix, defaults to zero matrix.
            control_constrain_fkt: Function that constrains the control.
            max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                                used to solve the constrained optimization problem.
            pgm_tolerance: Tolerance for the PGM algorithm.
        """
        u_d = np.array(_u_d)
        y_d = np.array(_y_d)

        assert len(u_d) == len(y_d), "u_d and y_d must have the same length."
        assert T_ini > 0, "T_ini must be greater than zero."
        assert (u_ini is None) == (y_ini is None), "u_ini and y_ini must be both None or not None."
        assert u_ini is None or len(u_ini) == T_ini, "u_ini must have the same length as T_ini."
        assert y_ini is None or len(y_ini) == T_ini, "y_ini must have the same length as T_ini."

        self.T_ini = T_ini
        self.r_len = r_len
        self.u_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        if u_ini is not None:
            for u in u_ini:
                self.u_ini.append(np.array(u))
        self.y_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        if y_ini is not None:
            for y in y_ini:
                self.y_ini.append(np.array(y))
        if Q is None:
            #Q = np.eye(r_len)
            Q = np.eye(r_len)
        if R is None:
            R = np.zeros((r_len, r_len)) # todo problems with R
        self.Q = Q
        self.R = R
        self.control_constrain_fkt = control_constrain_fkt
        self.max_pgm_iterations = max_pgm_iterations
        self.pgm_tolerance = pgm_tolerance

        # todo dirty fix
        self.N = 1000
        U = hankel_matrix(T_ini + r_len, u_d)
        U_p = U[:T_ini, :]  # past
        U_f = U[T_ini:T_ini+self.N, :]  # future
        Y = hankel_matrix(T_ini + r_len, y_d)
        Y_p = Y[:T_ini, :]  # past
        Y_f = Y[T_ini:, :]  # future

        # Now solving
        # minimize: ||y - r||_Q^2 + ||u||_R^2
        # subject to: [U_p; Y_p; U_f; Y_f] * g = [u_ini; y_ini; u; y]

        #M = np.block([[self.H_u[:self.Tini,:]],[self.H_y[:self.Tini,:]],[self.H_u[self.Tini:self.Tini+self.N,:]]])
        
        # We define
        A = np.block([[U_p], [Y_p], [U_f]]) #M
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
        self.M_x = M[:, : 2 * r_len]
        self.M_u = M[:, 2 * T_ini :]

        # We can now solve the unconstrained problem.
        # This is a ridge regression problem with generalized Tikhonov regularization.
        # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
        # minimize: ||y - r||_Q^2 + ||u||_R^2
        # subject to: M_u * u = y - M_x * x
        # This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y).
        # Print shapes to debug the dimension mismatch
        print(f"M_u shape: {self.M_u.shape}")
        print(f"Q shape: {self.Q.shape}")
        print(f"R shape: {self.R.shape}")
        print("T_ini", T_ini)
        print("len r ", r_len)
        print("U p ", U_p.__len__())
        print("U f ", U_f.__len__())

        # Reconstructing M_u to align with Q and R
        #M_u_corrected = self.M_u.reshape((self.T_ini + self.r_len, -1))

        # Now M_u_corrected should have dimensions that match with Q and R
        #print(f"M_u_corrected shape: {M_u_corrected.shape}")  # Expected to be compatible

        # We precompute the matrix G = M_u^T * Q * M_u + R.
        self.G = self.M_u.T @ self.Q @ self.M_u #+ self.R

    def is_initialized(self) -> bool:
        "Returns whether the internal state is initialized."
        return len(self.u_ini) == self.T_ini and len(self.y_ini) == self.T_ini

    def update(self, u: list[int | float], y: list[int | float]) -> None:
        "Updates the internal state with the given control input and trajectory."
        self.u_ini.append(np.array(u))
        self.y_ini.append(np.array(y))

    def clear(self) -> None:
        "Clears the internal state."
        self.u_ini.clear()
        self.y_ini.clear()

    def control(self, _r: list[int | float]) -> list[int | float]:
        """
        Returns the optimal control for a given reference trajectory.
        Args:
            r: Reference trajectory.
        """
        assert self.is_initialized(), "Internal state is not initialized."
        assert len(_r) == self.r_len, "Reference trajectory has wrong length."

        r = np.array(_r).reshape(-1, 1)
        x = np.hstack((self.u_ini,self.y_ini))[:,None]
        #x = np.concatenate([self.u_ini, self.y_ini]).reshape(-1, 1)
        
        # Debugging output
        print(f"self.M_u shape: {self.M_u.shape}")
        print(f"self.M_u.T shape: {self.M_u.T.shape}")
        print(f"self.Q shape: {self.Q.shape}")
        print(f"self.M_x shape: {self.M_x.shape}")
        print(f"x shape: {x.shape}")
        print(f"r shape: {r.shape}")
        print(f"r - self.M_x @ x shape: {(r - self.M_x @ x).shape}")

        w = self.M_u.T @ self.Q @ (r - self.M_x @ x)
        u_star = np.linalg.solve(self.G, w)

        if self.control_constrain_fkt is not None:
            u_star = projected_gradient_method(
                self.G,
                u_star,
                w,
                self.control_constrain_fkt,
                self.max_pgm_iterations,
                self.pgm_tolerance,
            )
        return u_star[:, 0].tolist()
