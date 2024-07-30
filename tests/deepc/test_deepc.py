import unittest
import numpy as np
from deepc import deePC, DeePC


class System:
    def __init__(self, A, B, C, D, x_ini) -> None:
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.x = x_ini

    def step(self, u):
        self.x = self.A @ self.x + self.B @ u
        y = self.C @ self.x + self.D @ u
        return y


def lagged_response(u: np.ndarray, lag: int, y_ini) -> np.ndarray:
    y = np.zeros(len(u))
    for i in range(lag):
        y[i] = y_ini
    for i in range(lag, len(u)):
        y[i] = u[i - lag]
    return y


class TestUnconstrained(unittest.TestCase):
    def test_1D_LTI(self):
        system = System(
            A=np.array([[0.9]]),
            B=np.array([[0.1]]),
            C=np.array([[1]]),
            D=np.array([[0]]),
            x_ini=np.array([0]),
        )
        u_d = np.empty(0)
        y_d = np.empty(0)
        for i in range(50):
            u = np.array([np.sin(i / 10)])
            u_d = np.append(u_d, u)
            y_d = np.append(y_d, system.step(u))

        u_ini = np.array([0, 0, 0, 0])
        y_ini = np.array([0, 0, 0, 0])
        r = np.array([0.5, 0.7])

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)
        y_star = np.empty(0)
        for u in u_ini:
            system.step(np.array([u]))
        for u in u_star:
            y_star = np.append(y_star, system.step(np.array([u])))
        print(u_star)
        print(y_star)

    def test_1D_LTI2(self):
        system = System(
            A=np.array([[0.9]]),
            B=np.array([[0.1]]),
            C=np.array([[1]]),
            D=np.array([[0]]),
            x_ini=np.array([0]),
        )
        u_d = np.empty(0)
        y_d = np.empty(0)
        for i in range(50):
            u = np.array([np.sin(i / 10)])
            u_d = np.append(u_d, u)
            y_d = np.append(y_d, system.step(u))
        print(u_d, y_d)

        controller = DeePC(u_d, y_d, T_ini=4, r_len=1)
        for i in range(4):
            controller.append(u=np.array([0]), y=np.array([0]))

        # Go to 0.1
        for _ in range(10):
            u_star = controller.control(np.array([0.1]))
            u = u_star
            y = system.step(u_star)
            print(u, y)
            controller.append(u, y)

    def test_proportional_system_lag_1(self):
        u_d = np.array(np.sin(np.linspace(0, 2 * np.pi, 15)))
        y_d = lagged_response(u_d, lag=1, y_ini=1)
        u_ini = np.array([1])
        y_ini = np.array([0])
        r = np.array([0.5, 0.5])

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)
        print(u_star)

    def test_proportional_system_lag_2(self):
        u_d = np.array(np.sin(np.linspace(0, 2 * np.pi, 15)))
        y_d = lagged_response(u_d, lag=2, y_ini=1)
        u_ini = np.array([0, 1])
        y_ini = np.array([0, 0])
        r = np.array([0.5, 0.5])

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)
        print(u_star)


class TestController(unittest.TestCase):
    def test_proportional_system_lag_1(self):
        u_d = np.array(np.sin(np.linspace(0, 2 * np.pi, 15)))
        y_d = lagged_response(u_d, lag=1, y_ini=1)
        r = np.array([0.5, 0.5])

        controller = DeePC(u_d, y_d, T_ini=1, r_len=2)
        controller.append(u=1, y=0)
        u_star = controller.control(r)
        print(u_star)


if __name__ == "__main__":
    unittest.main()
