import numpy as np


class DiscreteLTI:
    "Discrete Linear Time-Invariant System"

    def __init__(self, A: list, B: list, C: list, D: list, x_ini: list) -> None:
        """
        A: State matrix
        B: Input matrix
        C: Output matrix
        D: Feedforward matrix
        x_ini: Initial state
        """
        self.A = np.array(A)  # State matrix
        self.B = np.array(B)  # Input matrix
        self.C = np.array(C)  # Output matrix
        self.D = np.array(D)  # Feedforward matrix
        self.x = np.array(x_ini)  # State
        assert self.A.shape[0] == self.x.shape[0]
        assert self.A.shape[1] == self.x.shape[0]
        assert self.B.shape[0] == self.x.shape[0]
        assert self.B.shape[1] == self.D.shape[1]
        assert self.C.shape[0] == self.D.shape[0]
        assert self.C.shape[1] == self.x.shape[0]

    @property
    def input_dim(self) -> int:
        "Get the input dimension"
        return self.B.shape[1]

    @property
    def output_dim(self) -> int:
        "Get the output dimension"
        return self.C.shape[0]

    def set_state(self, x: list) -> None:
        "Set the state of the system"
        assert len(x) == self.x.shape[0]
        self.x = np.array(x)

    def is_controllable(self) -> bool:
        "Check if the system is controllable"
        # A system is controllable iff the controllability matrix has full rank
        # The controllability matrix is defined as
        # Q = [B, A @ B, A^2 @ B, ..., A^(n-1) @ B]
        # where n is the number of states
        n = self.A.shape[0]
        m = self.B.shape[1]
        Q = np.zeros((n, n * m))
        for i in range(n):
            Q[:, i * m : (i + 1) * m] = np.linalg.matrix_power(self.A, i) @ self.B
        return np.linalg.matrix_rank(Q) == n

    def is_observable(self) -> bool:
        "Check if the system is observable"
        # A system is observable iff the observability matrix has full rank
        # The observability matrix is defined as
        # Q = [C, C @ A, C @ A^2, ..., C @ A^(n-1)]
        # where n is the number of states
        n = self.A.shape[0]
        m = self.B.shape[1]
        Q = np.zeros((n * m, n))
        for i in range(n):
            Q[i * m : (i + 1) * m, :] = self.C @ np.linalg.matrix_power(self.A, i)
        return np.linalg.matrix_rank(Q) == n

    def is_stable(self) -> bool:
        "Check if the system is stable"
        # A system is stable iff all eigenvalues of the state matrix are inside the unit circle
        eig_vals = np.linalg.eigvals(self.A)
        return bool(np.all(np.abs(eig_vals) < 1))

    def apply(self, u: int | float | list) -> float | list:
        "Apply input(s) and get output(s)"
        if isinstance(u, list):
            _u = np.array(u)
        if isinstance(u, (int, float)):
            _u = np.array([u])

        assert _u.shape[0] == self.B.shape[1]

        self.x = self.A @ self.x + self.B @ _u
        y = self.C @ self.x + self.D @ _u

        if y.shape[0] == 1:
            return y[0].item()
        return y.tolist()

    def apply_multiple(self, u: list[int | float | list]) -> list[float | list]:
        "Apply multiple inputs and get multiple outputs"
        return [self.apply(u_i) for u_i in u]


class LaggedLTI(DiscreteLTI):
    "Lagged Discrete Linear Time-Invariant System"

    def __init__(self, lag: int, x_ini: list) -> None:
        "x_ini: Initial state"
        assert lag >= 0
        assert len(x_ini) == lag + 1

        A = np.zeros((lag + 1, lag + 1))
        A[:-1, 1:] = np.eye(lag)
        B = np.zeros((lag + 1, 1))
        B[-1, 0] = 1
        C = np.zeros((1, lag + 1))
        C[0, 0] = 1
        D = np.zeros((1, 1))
        super().__init__(A.tolist(), B.tolist(), C.tolist(), D.tolist(), x_ini)
