import unittest
import numpy as np
from deepc import Controller, DiscreteLTI
from helpers import create_1D_in_1D_out_LTI, create_2D_in_3D_out_LTI, gather_offline_data


def warm_up_controller(controller: Controller, system: DiscreteLTI, u: list | np.ndarray) -> None:
    "Warm up the controller until it is initialized"
    while not controller.is_initialized():
        y = system.apply(u)
        controller.update(u, y)


def control_system(
    controller: Controller, system: DiscreteLTI, target: list, time_steps: int, u_0: list | None = None
) -> float:
    """
    Control the system for a given number of time steps.
    Returns the output of the system after the last time step.
    """
    for _ in range(time_steps):
        u = controller.apply(target, u_0)[0]
        y = system.apply(u)
        controller.update(u, y)
    return y


class Test_1D_in_1D_out_LTI(unittest.TestCase):
    def test_unconstrained(self):
        system = create_1D_in_1D_out_LTI()
        u_d, y_d = gather_offline_data(system, samples=1_000)
        T_ini = 20
        target = [[10], [10], [10]]

        controller = Controller(u_d, y_d, T_ini, len(target))
        warm_up_controller(controller, system, u=[1])
        y = control_system(controller, system, target, time_steps=2 * T_ini)

        np.testing.assert_array_almost_equal(y, target[0])

    def test_constrained(self):
        system = create_1D_in_1D_out_LTI()
        u_d, y_d = gather_offline_data(system, samples=1_000)
        T_ini = 20
        target = [[10]]

        controller = Controller(u_d, y_d, T_ini, len(target), input_constrain_fkt=lambda u: np.clip(u, 0, 25))
        warm_up_controller(controller, system, u=[1])
        y = control_system(controller, system, target, time_steps=2 * T_ini)

        np.testing.assert_array_almost_equal(y, target[0])


class Test_2D_in_3D_out_LTI(unittest.TestCase):
    def test_unconstrained(self):
        system = create_2D_in_3D_out_LTI()
        u_d, y_d = gather_offline_data(system, samples=1_000)
        T_ini = 20
        target = [(0.19, 0.92, 0.24)]

        controller = Controller(u_d, y_d, T_ini, len(target))
        warm_up_controller(controller, system, u=(1, 1))
        y = control_system(controller, system, target, time_steps=2 * T_ini)

        np.testing.assert_array_almost_equal(y, target[0], decimal=2)

    def test_constrained(self):
        system = create_2D_in_3D_out_LTI()
        u_d, y_d = gather_offline_data(system, samples=1_000)
        T_ini = 20
        target = [(0.19, 0.92, 0.24)]

        controller = Controller(u_d, y_d, T_ini, len(target), input_constrain_fkt=lambda u: np.clip(u, 0, 25))
        warm_up_controller(controller, system, u=(1, 1))
        y = control_system(controller, system, target, time_steps=2 * T_ini)

        np.testing.assert_array_almost_equal(y, target[0], decimal=2)

    def test_offset(self):
        system = create_2D_in_3D_out_LTI()
        u_d, y_d = gather_offline_data(system, samples=1_000)
        T_ini = 20
        target = [(0.32, 0.95, 0.24)]

        controller = Controller(u_d, y_d, T_ini, len(target), R=0.01)
        warm_up_controller(controller, system, u=(1, 1))
        y = control_system(controller, system, target, time_steps=2 * T_ini, u_0=[(1, 1)])

        np.testing.assert_array_almost_equal(y, target[0], decimal=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
