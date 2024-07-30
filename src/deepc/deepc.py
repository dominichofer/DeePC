import collections
from typing import Callable
import numpy as np
from .math import left_pseudoinverse, hankel_matrix, projected_gradient_method


def deePC(
    u_d: np.ndarray,
    y_d: np.ndarray,
    u_ini: np.ndarray,
    y_ini: np.ndarray,
    r: np.ndarray,
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
        y_d: Trajectories from an offline procedure.
        u_ini: Control inputs to initiate the state.
        y_ini: Trajectories to initiate the state.
        r: Reference trajectory.
        Q: Output cost matrix, defaults to identity matrix.
        R: Control cost matrix, defaults to zero matrix.
        control_constrain_fkt: Function that constrains the control.
        max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                            used to solve the constrained optimization problem.
        pgm_tolerance: Tolerance for the PGM algorithm.
    """
    assert len(u_d) == len(y_d), "u_d and y_d must have the same length."
    assert len(u_ini) == len(y_ini), "u_ini and y_ini must have the same length."
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
    x = np.block([u_ini, y_ini])
    # to get
    # A * g = [x; u]  (1)
    # and
    # Y_f * g = y  (2).

    # We multiply (1) from the left with the left pseudo inverse of A.
    # Since pinv(A) * A = I, we get g = pinv(A) * [x; u].
    # Substituting g in (2) gives Y_f * pinv(A) * [x; u] = y.

    # We define
    B = Y_f @ left_pseudoinverse(A)
    # and get B * [x; u] = y.

    # We define (B_x, B_u) := B such that B_x * x + B_u * u = y.
    B_x = B[:, : 2 * T_ini]
    B_u = B[:, 2 * T_ini :]

    # We can now solve the unconstrained problem.
    # This is a ridge regression problem with generalized Tikhonov regularization.
    # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
    # minimize: ||y - r||_Q^2 + ||u||_R^2
    # subject to: B_u * u = y - B_x * x

    G = B_u.T @ Q @ B_u + R
    u_star = np.linalg.solve(G, B_u.T @ Q @ (r - B_x @ x).T)

    if control_constrain_fkt is None:
        return u_star
    else:
        return projected_gradient_method(
            G, u_star, control_constrain_fkt, max_pgm_iterations, pgm_tolerance
        )


class DeePC:
    def __init__(
        self,
        u_d: np.ndarray,
        y_d: np.ndarray,
        T_ini: int,# carefull, different from the other function
        r_len: int,# carefull, different from the other function
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
        assert len(u_d) == len(y_d), "u_d and y_d must have the same length."
        self.u_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        self.y_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        self.T_ini = T_ini
        self.r_len = r_len
        self.Q = Q if Q is not None else np.eye(r_len) #problems with truth of np.array with more than one value
        self.R = R if R is not None else np.zeros((r_len, r_len))
        self.control_constrain_fkt = control_constrain_fkt
        self.max_pgm_iterations = max_pgm_iterations
        self.pgm_tolerance = pgm_tolerance

        U = hankel_matrix(T_ini + r_len, u_d)
        U_p = U[:T_ini, :]  # past
        U_f = U[T_ini:, :]  # future
        Y = hankel_matrix(T_ini + r_len, y_d)
        Y_p = Y[:T_ini, :]  # past
        Y_f = Y[T_ini:, :]  # future

        # Now solving
        # minimize: ||y - r||_Q^2 + ||u||_R^2
        # subject to: [U_p; Y_p; U_f; Y_f] * g = [u_ini; y_ini; u; y]

        # We define
        A = np.block([[U_p], [Y_p], [U_f]]) #M
        # x = [u_ini; y_ini]
        # to get
        # A * g = [x; u]  (1)
        # and
        # Y_f * g = y  (2).

        # We multiply (1) from the left with the left pseudo inverse of A.
        # Since pinv(A) * A = I, we get g = pinv(A) * [x; u].
        # Substituting g in (2) gives Y_f * pinv(A) * [x; u] = y.

        # We define
        B = Y_f @ left_pseudoinverse(A) #Mbar
        # and get B * [x; u] = y.

        # We define (B_x, B_u) := B such that B_x * x + B_u * u = y.
        self.B_x = B[:, : 2 * T_ini]
        self.B_u = B[:, 2 * T_ini :]

        # We can now solve the unconstrained problem.
        # This is a ridge regression problem with generalized Tikhonov regularization.
        # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
        # minimize: ||y - r||_Q^2 + ||u||_R^2
        # subject to: B_u * u = y - B_x * x
        # This has an explicit solution u_star = (B_u^T * Q * B_u + R)^-1 * (B_u^T * Q * y).

        # We precompute the matrix G = B_u^T * Q * B_u + R.
        self.G = self.B_u.T @ self.Q @ self.B_u + self.R

    def is_initialized(self) -> bool:
        "Returns whether the internal state is initialized."
        return len(self.u_ini) == self.T_ini and len(self.y_ini) == self.T_ini

    def append(self, u: np.ndarray, y: np.ndarray) -> None:
        "Appends a control input and an output measurement to the internal state."
        self.u_ini.append(u) # is this a circular buffer? or keeps growing infinite?
        self.y_ini.append(y)

    def clear(self) -> None:
        "Clears the internal state."
        self.u_ini.clear()
        self.y_ini.clear()

    def control(self, r: np.ndarray) -> np.ndarray:
        """
        Returns the optimal control for a given reference trajectory.
        Args:
            r: Reference trajectory.
        """
        assert len(self.u_ini) == self.T_ini, "Not enough initial control inputs."
        assert len(self.y_ini) == self.T_ini, "Not enough initial trajectories."
        assert len(r) == self.r_len, "Reference trajectory has wrong length."

        ## to add assert the datatype. had a nasty bug beacause the inital u_ini was int and the others were np array
        #u_ini = np.array([np.array(x) for x in self.u_ini])
        #y_ini = np.array([np.array(x) for x in self.y_ini])

        #x = np.block([self.u_ini, self.y_ini])
        x = np.concatenate([self.u_ini, self.y_ini]).reshape(-1, 1)  # Reshape x to be a column vector
        y = r - self.B_x @ x #size m
        u_star = np.linalg.solve(self.G, self.B_u.T @ self.Q @ y) # size n if Bu (mxn)
        
        #print(u_star)
        # this returns a 20 by 20 matrix (in my case). but not a series of control inputs 

        if self.control_constrain_fkt is None:
            return u_star
        else:
            return projected_gradient_method(
                self.G,
                u_star,
                self.control_constrain_fkt,
                self.max_pgm_iterations,
                self.pgm_tolerance,
            )
