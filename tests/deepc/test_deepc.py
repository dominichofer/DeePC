import unittest
import math
import numpy as np
from deepc import clamp, deePC, Controller, DiscreteLTI, LaggedLTI


def lti_2D() -> DiscreteLTI:
    system = DiscreteLTI(
        A=[[0.9, -0.2], [0.7, 0.1]],
        B=[[0.1], [0]],
        C=[[1, 0]],
        D=[[0.1]],
        x_ini=[1, 1],
    )
    assert system.is_controllable()
    assert system.is_observable()
    assert system.is_stable()
    return system


def gather_offline_data(system: DiscreteLTI) -> tuple:
    # Gather offline data
    u_d = [i * math.sin(i * i / 100) for i in range(500)]
    y_d = system.apply_multiple(u_d)
    return u_d, y_d


def warm_up_controller(controller: Controller, system: DiscreteLTI, u: float) -> None:
    while not controller.is_initialized():
        y = system.apply(u)
        controller.update(u, y)


def control_system(controller: Controller, system: DiscreteLTI, r: list, time_steps: int) -> float:
    for _ in range(time_steps):
        u = controller.control(r)[0]
        y = system.apply(u)
        controller.update(u, y)
    return y


class TestDeePC(unittest.TestCase):
    def test_unconstrained_lag_0(self):
        system = LaggedLTI(lag=0, x_ini=[1])

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
        system = lti_2D()

        # Offline data
        u_d, y_d = gather_offline_data(system)

        # Initial conditions
        u_ini = [1] * 20
        y_ini = system.apply_multiple(u_ini)

        # Reference trajectory
        r = [3, 3]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        y_star = system.apply_multiple(u_star)
        self.assertAlmostEqual(y_star[0], r[0])

    def test_constrained_2D_LTI(self):
        system = lti_2D()

        # Offline data
        u_d, y_d = gather_offline_data(system)

        # Initial conditions
        u_ini = [1] * 20
        y_ini = system.apply_multiple(u_ini)

        # Reference trajectory
        r = [3, 3]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        y_star = system.apply_multiple(u_star)
        self.assertAlmostEqual(y_star[0], r[0])


class TestController(unittest.TestCase):
    def test_unconstrained_2D_LTI(self):
        system = lti_2D()
        u_d, y_d = gather_offline_data(system)
        T_ini = 20
        r_len = 3
        r = [10] * r_len

        controller = Controller(u_d, y_d, T_ini, r_len)
        warm_up_controller(controller, system, 1)
        y = control_system(controller, system, r, time_steps=2 * T_ini)

        self.assertAlmostEqual(y, r[0])

    def test_constrained_2D_LTI(self):
        system = lti_2D()
        u_d, y_d = gather_offline_data(system)
        T_ini = 20
        r_len = 1
        r = [10] * r_len

        controller = Controller(
            u_d, y_d, T_ini, r_len, control_constrain_fkt=lambda u: clamp(u, 0, 25)
        )
        warm_up_controller(controller, system, 1)
        y = control_system(controller, system, r, time_steps=2 * T_ini)

        self.assertAlmostEqual(y, r[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
