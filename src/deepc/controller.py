import collections
from typing import Callable
import numpy as np
from numpy.linalg import matrix_rank, svd
from .math import hankel_matrix, projected_gradient_method
from .deepc import as_column_vector, check_dimensions

# noise rejection version only
from typing import List, Optional
from cvxpy import Variable, Minimize, Problem, sum_squares, norm1, hstack, vstack, Parameter, Constraint
    

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

        self.u_d = u_d
        self.y_d = y_d

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

    def apply(self, target: list | np.ndarray, u_0: list | np.ndarray | None = None) -> list[float] | None:
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
        w = self.M_u.T @ self.Q @ (target - self.M_x @ x) + self.R @ u_0

        u_star = np.linalg.lstsq(self.G, w, rcond=None)[0]

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
        #u_0 = np.zeros((self.target_len, self.input_dims))
        #u_0 = np.concatenate(u_0).reshape(-1, 1)
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

        ''' get the u_ref by solving target = M_x ⋅x + M_u⋅u_bar 
        where:

        - target is the desired future output trajectory.
        - x is the vector of initial conditions (past inputs and outputs).
        - u_bar is the steady state input sequence we aim to compute.
        '''

        # Compute effective T_ini based on target length (for r smaller T_ini error)
        # effective_T_ini = min(self.T_ini, len(target) // self.output_dims) todo, still not working

        M_x_uini = self.M_x[:,:self.T_ini*self.input_dims]
        M_x_yini = self.M_x[:,self.T_ini*self.input_dims:]
        M_x_uini_extended = np.zeros_like(self.M_u)

        target = np.concatenate(target).reshape(-1, 1)

        # problem here if r is smaller than T_ini (see solution above)
        M_x_uini_extended[:,:M_x_uini.shape[1]] = M_x_uini

        u_bar = solve(M_x_uini_extended + self.M_u, target - M_x_yini@target[:self.T_ini*self.input_dims])
        
        '''
        LHS: Represents the total influence of the inputs (both initial and future) on the future outputs.
        RHS: target_subset = target[:self.T_ini * self.input_dims]: This is the subset of the target trajectory corresponding to the initial outputs.
             M_x_yini @ target_subset: Calculates the influence of the initial outputs on the future outputs.
             target - M_x_yini @ target_subset: Represents the desired future outputs after subtracting the effect of the initial outputs.
        '''

        # Flatten
        u_ini = np.concatenate(self.u_ini).reshape(-1, 1)
        y_ini = np.concatenate(self.y_ini).reshape(-1, 1)
        u_bar = np.concatenate(u_bar).reshape(-1, 1)
        x     = np.concatenate([u_ini, y_ini]).reshape(-1, 1)

        # verification of u_bar
        should_be_target = self.M_x @ x + self.M_u @ u_bar

        if not np.allclose(should_be_target, target, rtol = 0.2): # seems to be a good boundery. interesting things happen at ref change
            print('u_bar computation off')
            print('computed ss target ', should_be_target.tolist())
            print('given       target ', target.tolist())
            u_0 = np.zeros_like(u_bar)
        else:
            print('u_bar computation verified successfully')
            u_0 = u_bar
        # ---------------------------------------------------------------------------

        w = self.M_u.T @ self.Q @ (target - self.M_x @ x) + self.R @ u_0

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
    
    def initialize_regularization(self, lambda_g, lambda_y, rank):

        self.lambda_g = lambda_g
        self.lambda_y = lambda_y
        self.rank     = rank

        # Create Hankel matrices
        U = hankel_matrix(self.T_ini + self.target_len, self.u_d)
        U_p = U[: self.T_ini * self.input_dims, :]  # past inputs
        U_f = U[self.T_ini * self.input_dims :, :]  # future inputs
        Y = hankel_matrix(self.T_ini + self.target_len, self.y_d)
        Y_p = Y[: self.T_ini * self.output_dims, :]  # past outputs
        Y_f = Y[self.T_ini * self.output_dims :, :]  # future outputs

        # Low-rank approximation
        if self.rank is not None:
            rank = self.rank
            # Concatenate data matrices
            Data = np.vstack([U_p, Y_p, U_f, Y_f])
            print(f"Original Data shape: {Data.shape}")
            print(f"Original U_p shape: {U_p.shape}")
            print(f"Original Y_p shape: {Y_p.shape}")
            print(f"Original U_f shape: {U_f.shape}")
            print(f"Original Y_f shape: {Y_f.shape}")     

            # Perform SVD
            U_svd, S_svd, Vh_svd = svd(Data, full_matrices=False)
            print(f"U_svd shape: {U_svd.shape}")
            print(f"S_svd shape: {S_svd.shape}")
            print(f"Vh_svd shape: {Vh_svd.shape}")

            # Truncate to the specified rank
            U_svd_truncated = U_svd[:, :rank]
            S_svd_truncated = S_svd[:rank]
            Vh_svd_truncated = Vh_svd[:rank, :]
            print(f"Truncated U_svd shape: {U_svd_truncated.shape}")
            print(f"Truncated S_svd shape: {S_svd_truncated.shape}")
            print(f"Truncated Vh_svd shape: {Vh_svd_truncated.shape}")

            # Reconstruct the data matrices
            Data_approx = U_svd_truncated @ np.diag(S_svd_truncated) @ Vh_svd_truncated
            print(f"Approximated Data shape: {Data_approx.shape}")

            # Split the approximated data matrices
            split_idx1 = U_p.shape[0]
            split_idx2 = split_idx1 + Y_p.shape[0]
            split_idx3 = split_idx2 + U_f.shape[0]

            U_p = Data_approx[:split_idx1, :]
            Y_p = Data_approx[split_idx1:split_idx2, :]
            U_f = Data_approx[split_idx2:split_idx3, :]
            Y_f = Data_approx[split_idx3:, :]

            # Print the final shapes
            print(f"Final U_p shape: {U_p.shape}")
            print(f"Final Y_p shape: {Y_p.shape}")
            print(f"Final U_f shape: {U_f.shape}")
            print(f"Final Y_f shape: {Y_f.shape}")                                                                                  

        # Store the data matrices
        self.U_p = U_p
        self.Y_p = Y_p
        self.U_f = U_f
        self.Y_f = Y_f

        # For convenience, precompute some dimensions
        self.dim_g = U_p.shape[1]
        self.dim_u = self.input_dims * self.target_len
        self.dim_y = self.output_dims * self.target_len
        self.dim_sigma_y = self.output_dims * self.T_ini

        # Identity matrices
        self.I_u = np.eye(self.dim_u)
        self.I_y = np.eye(self.dim_y)

        self.Q_bar = np.kron(np.eye(self.target_len), self.Q[:self.output_dims,:self.output_dims])
        self.R_bar = np.kron(np.eye(self.target_len), self.R[:self.input_dims,:self.input_dims])

    def apply_regularized(
        self,
        target: List[float],
        u_constraints: Optional[Callable[[Variable], List[Constraint]]] = None,
        y_constraints: Optional[Callable[[Variable], List[Constraint]]] = None,
    ) -> Optional[np.ndarray]:
        """
        Returns the optimal control for a given reference trajectory
        using the regularized DeePC algorithm.
        Args:
            target: Target system outputs, optimal control tries to reach.
            u_constraints: Function to apply input constraints.
            y_constraints: Function to apply output constraints.
        """
        if not self.is_initialized():
            return None

        # Convert target to column vector
        target = np.concatenate(target).reshape(-1, 1)
        check_dimensions(target, "target", self.dim_y, 1)

        target_cvx = Parameter((self.dim_y, 1), value=target)

        # Flatten initial inputs and outputs
        u_ini = np.concatenate(self.u_ini).reshape(-1, 1)
        y_ini = np.concatenate(self.y_ini).reshape(-1, 1)

        g = Variable((self.dim_g, 1))
        u = Variable((self.dim_u, 1))
        y = Variable((self.dim_y, 1))
        sigma_y = Variable((self.dim_sigma_y, 1))

        # Constraint matrices
        A = vstack([
            self.U_p,
            self.Y_p,
            self.U_f,
            self.Y_f
        ])

        # Right-hand side
        b = vstack([
            u_ini ,
            y_ini + sigma_y ,
            u,
            y
        ])

        # Formulate the constraints
        constraints = [A @ g == b]

        # Add input and output constraints if provided
        if u_constraints is not None:
            constraints += u_constraints(u)

        if y_constraints is not None:
            constraints += y_constraints(y)



        print("Shape of y:", y.shape)
        print("Shape of target_cvx:", target_cvx.shape)
        print("Shape of self.Q_bar:", self.Q_bar.shape)
        print("Shape of u:", u.shape)
        print("Shape of self.R_bar:", self.R_bar.shape)


        # Cost function
        cost = sum_squares(self.Q_bar @ (y - target_cvx)) + sum_squares(self.R_bar @ u)
        cost += self.lambda_g * norm1(g) + self.lambda_y * norm1(sigma_y)
        print("cost " , cost)
        print("constraints:" ,constraints)

        # Define and solve the problem
        problem = Problem(Minimize(cost), constraints)
        problem.solve(solver= 'OSQP', verbose = True)
        #problem.solve(solver='SCS')  # or try  OSQP 'ECOS', 'MOSEK', etc.

        # Check if the problem was solved
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print("Problem not solved to optimality.")
            return None

        # Extract optimal control input
        u_star = u.value

        print("Optimal control input u_star:", u_star)
        print("Optimal slack variable sigma_y:", sigma_y.value)

        # Reshape to match input dimensions
        u_star = u_star.reshape(-1, self.input_dims)

        return u_star

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
