import collections
from typing import Callable
import numpy as np
from numpy.linalg import matrix_rank, svd
from .math_deepc import hankel_matrix, projected_gradient_method
from .deepc import as_column_vector, check_dimensions


class Controller:
    def __init__(
        self,
        u_d: list | np.ndarray,
        y_d: list | np.ndarray,
        T_ini: int,
        target_len: int,
        Q: np.ndarray | int | float | None = None,
        R: np.ndarray | int | float | None = None,
        input_constrain_fkt: Callable | None = None,
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
            T_ini: Number of system in- and outputs to initialize the state.
            target_len: Length of the target system outputs, optimal control tries to match.
            Q: Output cost matrix. Defaults to identity matrix.
               If int or float, diagonal matrix with this value.
            R: Control cost matrix. Defaults to zero matrix.
                If int or float, diagonal matrix with this value.
            input_constrain_fkt: Function that constrains the control inputs.
            max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
                                used to solve the constrained optimization problem.
            pgm_tolerance: Tolerance for the PGM algorithm.
        """
        assert T_ini > 0, f"T_ini must be greater than zero. {T_ini=}"

        u_d = as_column_vector(u_d)
        y_d = as_column_vector(y_d)

        offline_len = len(u_d)
        self.input_dims = u_d.shape[1]
        self.output_dims = y_d.shape[1]

        check_dimensions(u_d, "u_d", offline_len, self.input_dims)
        check_dimensions(y_d, "y_d", offline_len, self.output_dims)

        Q_size = target_len * self.output_dims
        if isinstance(Q, (int, float)):
            Q = np.eye(Q_size) * Q
        if Q is None:
            Q = np.eye(Q_size)
        check_dimensions(Q, "Q", Q_size, Q_size)

        R_size = target_len * self.input_dims
        if isinstance(R, (int, float)):
            R = np.eye(R_size) * R
        if R is None:
            R = np.zeros((R_size, R_size))
        check_dimensions(R, "R", R_size, R_size)

        self.T_ini = T_ini
        self.target_len = target_len
        self.u_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        self.y_ini: collections.deque[np.ndarray] = collections.deque(maxlen=T_ini)
        self.Q = Q
        self.R = R
        self.input_constrain_fkt = input_constrain_fkt
        self.max_pgm_iterations = max_pgm_iterations
        self.pgm_tolerance = pgm_tolerance

        U = hankel_matrix(T_ini + target_len, u_d)
        U_p = U[: T_ini * self.input_dims, :]  # past
        U_f = U[T_ini * self.input_dims :, :]  # future
        Y = hankel_matrix(T_ini + target_len, y_d)
        Y_p = Y[: T_ini * self.output_dims, :]  # past
        Y_f = Y[T_ini * self.output_dims :, :]  # future

        self.suggest_dimensions(U_p, U_f, Y_p, Y_f)

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
        dim_sum = self.input_dims + self.output_dims
        self.M_x = M[:, : T_ini * dim_sum]
        self.M_u = M[:, T_ini * dim_sum :]

        # We can now solve the unconstrained problem.
        # This is a ridge regression problem with generalized Tikhonov regularization.
        # https://en.wikipedia.org/wiki/Ridge_regression#Generalized_Tikhonov_regularization
        # minimize: ||y - r||_Q^2 + ||u||_R^2
        # subject to: M_u * u = y - M_x * x
        # This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y + R * u_0).

        # We precompute the matrix G = M_u^T * Q * M_u + R.
        self.G = self.M_u.T @ self.Q @ self.M_u + self.R

    def is_initialized(self) -> bool:
        "Returns whether the internal state is initialized."
        return len(self.u_ini) == self.T_ini and len(self.y_ini) == self.T_ini

    def update(self, u: list | np.ndarray, y: list | np.ndarray) -> None:
        "Updates the internal state with the given control input and trajectory."
        self.u_ini.append(as_column_vector(u))
        self.y_ini.append(as_column_vector(y))

    def clear(self) -> None:
        "Clears the internal state."
        self.u_ini.clear()
        self.y_ini.clear()

    def apply(
        self, target: list | np.ndarray, u_0: list | np.ndarray | None = None
    ) -> list[float] | None:
        """
        Returns the optimal control for a given reference trajectory
        or None if the controller is not initialized.
        Args:
            target: Target system outputs, optimal control tries to reach.
            u_0: Control input offset, defaults to zero vector.
        """
        if not self.is_initialized():
            return None

        target = as_column_vector(target)
        check_dimensions(target, "target", self.target_len, self.output_dims)

        if u_0 is None:
            u_0 = np.zeros((self.target_len, self.input_dims))
        else:
            u_0 = as_column_vector(u_0)
        check_dimensions(u_0, "u_0", self.target_len, self.input_dims)

        # Flatten
        u_ini = np.concatenate(self.u_ini).reshape(-1, 1)
        y_ini = np.concatenate(self.y_ini).reshape(-1, 1)
        target = np.concatenate(target).reshape(-1, 1)
        u_0 = np.concatenate(u_0).reshape(-1, 1)

        x = np.concatenate([u_ini, y_ini]).reshape(-1, 1)
        w = self.M_u.T @ self.Q @ ((target - self.M_x @ x) + self.R @ u_0)

        u_star = np.linalg.lstsq(self.G, w)[0]

        if self.input_constrain_fkt is not None:
            u_star = projected_gradient_method(
                self.G,
                u_star,
                w,
                self.input_constrain_fkt,
                self.max_pgm_iterations,
                self.pgm_tolerance,
            )

        return u_star.reshape(-1, self.input_dims)
    




    def apply_trajectory_tracking_version(
        self, target: list | None = None
    ) -> list[float] | None:
        from scipy.linalg import solve
        """
        Returns the optimal control for a given reference trajectory
        Args:
            target: Target system outputs, optimal control tries to reach.
            u_bar: Computes u_bar from the system
        """
        if not self.is_initialized():
            return None

        target = as_column_vector(target)
        check_dimensions(target, "target", self.target_len, self.output_dims)

        ''' get the u_ref by solving target = M_x ⋅x + M_u⋅u bar 
        where:

        - target is the desired future output trajectory.
        - x is the vector of initial conditions (past inputs and outputs).
        - u_bar is the future input sequence we aim to compute.
        '''

        # Compute effective T_ini based on target length (for r smaller T_ini error)
        # effective_T_ini = min(self.T_ini, len(target) // self.output_dims) todo, still not working

        M_x_initial_u = self.M_x[:,:self.T_ini*self.input_dims]
        M_x_initial_y = self.M_x[:,self.T_ini*self.input_dims:]
        M_bar = np.zeros_like(self.M_u)

        target = np.concatenate(target).reshape(-1, 1)

        # problem here if r is smaller than T_ini (see solution above)
        M_bar[:,:M_x_initial_u.shape[1]] = M_x_initial_u

        u_bar = solve(M_bar+self.M_u, target-M_x_initial_y@target[:self.T_ini*self.input_dims])
        '''
        LHS: Represents the total influence of the inputs (both initial and future) on the future outputs.
        RHS: target_subset = target[:self.T_ini * self.input_dims]: This is the subset of the target trajectory corresponding to the initial outputs.
             M_x_initial_y @ target_subset: Calculates the influence of the initial outputs on the future outputs.
             target - M_x_initial_y @ target_subset: Represents the desired future outputs after subtracting the effect of the initial outputs.
        '''

        # Flatten
        u_ini = np.concatenate(self.u_ini).reshape(-1, 1)
        y_ini = np.concatenate(self.y_ini).reshape(-1, 1)
        u_0 = np.concatenate(u_bar).reshape(-1, 1)
        x = np.concatenate([u_ini, y_ini]).reshape(-1, 1)

        # verification of u_bar
        should_be_target = self.M_x @ x + self.M_u @ u_bar

        if not np.allclose(should_be_target, target):
            print('u_bar computation problem')
        else:
            print('u_bar computation verified successfully')
        # ---------------------------------------------------------------------------



        w = self.M_u.T @ self.Q @ ((target - self.M_x @ x) + self.R @ u_0)

        u_star = np.linalg.lstsq(self.G, w, rcond= None)[0]

        if self.input_constrain_fkt is not None:
            u_star = projected_gradient_method(
                self.G,
                u_star,
                w,
                self.input_constrain_fkt,
                self.max_pgm_iterations,
                self.pgm_tolerance,
            )

        return u_star.reshape(-1, self.input_dims)


    def assess_matrix_quality(self,matrix):
        """ Assess the quality of the matrix using rank, condition number, and singular values. """
        rank = matrix_rank(matrix)
        _, s, _ = svd(matrix)
        cond_number = np.linalg.cond(matrix)
        energy_retained = np.cumsum(s**2) / np.sum(s**2)
        
        return rank, cond_number, s, energy_retained


    def suggest_dimensions(self,U_p, U_f, Y_p, Y_f, energy_threshold=0.99):
        """ Suggest optimal dimensions based on the energy retained in the principal components. """
        # Assess U_p and U_f
        rank_U_p, cond_U_p, s_U_p, energy_U_p = self.assess_matrix_quality(U_p)
        rank_U_f, cond_U_f, s_U_f, energy_U_f = self.assess_matrix_quality(U_f)
        
        # Assess Y_p and Y_f
        rank_Y_p, cond_Y_p, s_Y_p, energy_Y_p = self.assess_matrix_quality(Y_p)
        rank_Y_f, cond_Y_f, s_Y_f, energy_Y_f = self.assess_matrix_quality(Y_f)
        
        # Suggest number of dimensions to retain
        suggested_dims_U = np.searchsorted(energy_U_p, energy_threshold) + 1
        suggested_dims_Y = np.searchsorted(energy_Y_p, energy_threshold) + 1
        
        print("Assessment of U_p:")
        print(f"Dimensions: {U_p.shape}")
        print(f"Rank: {rank_U_p}, Condition Number: {cond_U_p}")
        print(f"Suggested dimensions (U): {suggested_dims_U} (retaining {energy_threshold*100}% energy)")
        
        print("Assessment of U_f:")
        print(f"Dimensions: {U_f.shape}")
        print(f"Rank: {rank_U_f}, Condition Number: {cond_U_f}")
        
        print("Assessment of Y_p:")
        print(f"Dimensions: {Y_p.shape}")
        print(f"Rank: {rank_Y_p}, Condition Number: {cond_Y_p}")
        print(f"Suggested dimensions (Y): {suggested_dims_Y} (retaining {energy_threshold*100}% energy)")
        
        print("Assessment of Y_f:")
        print(f"Dimensions: {Y_f.shape}")
        print(f"Rank: {rank_Y_f}, Condition Number: {cond_Y_f}")
        
        return suggested_dims_U, suggested_dims_Y
