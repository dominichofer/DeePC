import unittest
import numpy as np
from deepc import deePC


def lagged_response(u: np.ndarray, lag: int, y_ini) -> np.ndarray:
    y = np.zeros(len(u))
    for i in range(lag):
        y[i] = y_ini
    for i in range(lag, len(u)):
        y[i] = u[i - lag]
    return y


class TestUnconstrained(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
