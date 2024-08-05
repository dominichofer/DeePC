import unittest
import numpy as np
from deepc import deePC, Controller
from deepc.lti import DescreteLTI, LaggedLTI


class TestDeePC(unittest.TestCase):
    def test_unconstrained_lag_0(self):
        system = LaggedLTI(lag=0, x_ini=[1])
        assert system.is_controllable()
        assert system.is_observable()
        assert system.is_stable()

        # Offline data
        u_d = [1, 2, 3, 4, 5]
        y_d = system.apply_multiple(u_d)

        # Initial conditions
        u_ini = [2]
        y_ini = system.apply_multiple(u_ini)

        # Reference trajectory
        r = [4]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        y_star = system.apply(u_star)
        np.testing.assert_array_almost_equal(y_star, r)

    def test_unconstrained_2D_LTI(self):
        system = DescreteLTI(
            A=[[0.9, -0.2], [0.7, 0.1]],
            B=[[0.1], [0]],
            C=[[1, 0]],
            D=[[0.1]],
            x_ini=[1, 1],
        )
        assert system.is_controllable()
        assert system.is_observable()
        assert system.is_stable()

        # Offline data
        u_d = [i * np.sin(i / 20) for i in range(500)]
        y_d = system.apply_multiple(u_d)

        # Initial conditions
        u_ini = [1] * 20
        y_ini = system.apply_multiple(u_ini)

        # Reference trajectory
        r = [3, 3]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r, R=0.001 * np.eye(len(r)))

        y_star = system.apply_multiple(u_star)
        self.assertAlmostEqual(y_star[0], r[0], delta=0.1)


class TestController(unittest.TestCase):
    def test_unconstrained_2D_LTI(self):
        system = DescreteLTI(
            A=[[0.9, -0.2], [0.7, 0.1]],
            B=[[0.1], [0]],
            C=[[1, 0]],
            D=[[0.1]],
            x_ini=[1, 1],
        )
        assert system.is_controllable()
        assert system.is_observable()
        assert system.is_stable()

        # Offline data
        u_d = [i * np.sin(i / 20) for i in range(500)]
        y_d = system.apply_multiple(u_d)

        T_ini = 20
        r_len = 1

        controller = Controller(u_d, y_d, T_ini, r_len, R=0.001 * np.eye(1))
        # Initialize controller
        while not controller.is_initialized():
            u = 1
            y = system.apply(u)
            controller.update(u, y)
        r = [10] * r_len
        # Control
        for _ in range(2 * T_ini):
            u = controller.control(r)[0]
            y = system.apply(u)
            controller.update(u, y)

        # Test controller
        u = controller.control(r)[0]
        y = system.apply(u)
        self.assertAlmostEqual(y, r[0], delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
