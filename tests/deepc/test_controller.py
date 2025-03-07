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
    def setUp(self):
        self.system = create_1D_in_1D_out_LTI()
        self.u_d, self.y_d = gather_offline_data(self.system, samples=1_000) # Offline data
        self.T_ini = 20
        self.target = [[3], [3]]

    def test_unconstrained(self):
        controller = Controller(self.u_d, self.y_d, self.T_ini, len(self.target))
        warm_up_controller(controller, self.system, u=[1])
        y = control_system(controller, self.system, self.target, time_steps=2 * self.T_ini)
        np.testing.assert_array_almost_equal(y, self.target[0])

    def test_constrained(self):
        controller = Controller(self.u_d, self.y_d, self.T_ini, len(self.target), input_constrain_fkt=lambda u: np.clip(u, 0, 25))
        warm_up_controller(controller, self.system, u=[1])
        y = control_system(controller, self.system, self.target, time_steps=2 * self.T_ini)
        np.testing.assert_array_almost_equal(y, self.target[0])

    def test_offset(self):
        controller = Controller(self.u_d, self.y_d, self.T_ini, len(self.target), R=0.001)
        warm_up_controller(controller, self.system, u=[1])
        y = control_system(controller, self.system, self.target, time_steps=2 * self.T_ini, u_0=[[10], [10]])
        np.testing.assert_array_almost_equal(y, self.target[0], decimal=1)


class Test_2D_in_3D_out_LTI(unittest.TestCase):
    def setUp(self):
        self.system = create_2D_in_3D_out_LTI()
        self.u_d, self.y_d = gather_offline_data(self.system, samples=1_000)
        self.T_ini = 20
        self.target = [(0.19, 0.92, 0.24)]

    def test_unconstrained(self):
        controller = Controller(self.u_d, self.y_d, self.T_ini, len(self.target))
        warm_up_controller(controller, self.system, u=(1, 1))
        y = control_system(controller, self.system, self.target, time_steps=2 * self.T_ini)
        np.testing.assert_array_almost_equal(y, self.target[0], decimal=2)

    def test_constrained(self):
        controller = Controller(self.u_d, self.y_d, self.T_ini, len(self.target), input_constrain_fkt=lambda u: np.clip(u, -15, 15))
        warm_up_controller(controller, self.system, u=(1, 1))
        y = control_system(controller, self.system, self.target, time_steps=2 * self.T_ini)
        np.testing.assert_array_almost_equal(y, self.target[0], decimal=2)

    def test_offset(self):
        controller = Controller(self.u_d, self.y_d, self.T_ini, len(self.target), R=0.001)
        warm_up_controller(controller, self.system, u=(1, 1))
        y = control_system(controller, self.system, self.target, time_steps=2 * self.T_ini, u_0=[(1, 1)])
        np.testing.assert_array_almost_equal(y, self.target[0], decimal=1)


class Test_types(unittest.TestCase):
    "Test the different types of input data."

    def test_int(self):
        u_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        y_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        u_ini = [2]
        y_ini = [2]
        u = [[4]]
        u_0 = [3]
        target = [4]

        controller = Controller(u_d, y_d, 1, 1)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_int_with_2_targets(self):
        u_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        y_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        u_ini = [2]
        y_ini = [2]
        u = [[4], [4]]
        u_0 = [[3], [3]]
        target = [4, 4]

        controller = Controller(u_d, y_d, 1, 2)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_tuple(self):
        u_d = [(1,), (2,), (3,), (4,), (-5,), (6,), (7,), (8,), (9,), (10,)]
        y_d = [(1,), (2,), (3,), (4,), (-5,), (6,), (7,), (8,), (9,), (10,)]
        u_ini = [(2,)]
        y_ini = [(2,)]
        u = [(4,)]
        u_0 = [(3,)]
        target = [(4,)]

        controller = Controller(u_d, y_d, 1, 1)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_tuple_with_2_targets(self):
        u_d = [(1,), (2,), (3,), (4,), (-5,), (6,), (7,), (8,), (9,), (10,)]
        y_d = [(1,), (2,), (3,), (4,), (-5,), (6,), (7,), (8,), (9,), (10,)]
        u_ini = [(2,)]
        y_ini = [(2,)]
        u = [(4,), (4,)]
        u_0 = [(3,), (3,)]
        target = [(4,), (4,)]

        controller = Controller(u_d, y_d, 1, 2)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_list(self):
        u_d = [[1], [2], [3], [4], [-5], [6], [7], [8], [9], [10]]
        y_d = [[1], [2], [3], [4], [-5], [6], [7], [8], [9], [10]]
        u_ini = [[2]]
        y_ini = [[2]]
        u = [[4]]
        u_0 = [[3]]
        target = [[4]]

        controller = Controller(u_d, y_d, 1, 1)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_list_with_2_targets(self):
        u_d = [[1], [2], [3], [4], [-5], [6], [7], [8], [9], [10]]
        y_d = [[1], [2], [3], [4], [-5], [6], [7], [8], [9], [10]]
        u_ini = [[2]]
        y_ini = [[2]]
        u = [[4], [4]]
        u_0 = [[3], [3]]
        target = [[4], [4]]

        controller = Controller(u_d, y_d, 1, 2)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_nparray(self):
        u_d = np.array([1, 2, 3, 4, -5, 6, 7, 8, 9, 10])
        y_d = np.array([1, 2, 3, 4, -5, 6, 7, 8, 9, 10])
        u_ini = np.array([2])
        y_ini = np.array([2])
        u = np.array([[4]])
        u_0 = np.array([3])
        target = np.array([4])

        controller = Controller(u_d, y_d, 1, 1)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_nparray_with_2_targets(self):
        u_d = np.array([1, 2, 3, 4, -5, 6, 7, 8, 9, 10])
        y_d = np.array([1, 2, 3, 4, -5, 6, 7, 8, 9, 10])
        u_ini = np.array([2])
        y_ini = np.array([2])
        u = np.array([[4], [4]])
        u_0 = np.array([[3], [3]])
        target = np.array([4, 4])

        controller = Controller(u_d, y_d, 1, 2)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_tuple_multidim(self):
        u_d = [(x, y) for x in range(3) for y in range(3)]
        y_d = [(x, y, x + y) for x, y in u_d]
        u_ini = [(2, 2)]
        y_ini = [(x, y, x + y) for x, y in u_ini]
        u = [(1, 3)]
        u_0 = [(2, 2)]
        target = [(x, y, x + y) for x, y in u]

        controller = Controller(u_d, y_d, 1, 1)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_list_multidim(self):
        u_d = [[x, y] for x in range(3) for y in range(3)]
        y_d = [[x, y, x + y] for x, y in u_d]
        u_ini = [[2, 2]]
        y_ini = [[x, y, x + y] for x, y in u_ini]
        u = [[1, 3]]
        u_0 = [[2, 2]]
        target = [[x, y, x + y] for x, y in u]

        controller = Controller(u_d, y_d, 1, 1)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)

    def test_nparray_multidim(self):
        u_d = np.array([[x, y] for x in range(3) for y in range(3)])
        y_d = np.array([[x, y, x + y] for x, y in u_d])
        u_ini = np.array([[2, 2]])
        y_ini = np.array([[x, y, x + y] for x, y in u_ini])
        u = np.array([[1, 3]])
        u_0 = np.array([[2, 2]])
        target = np.array([[x, y, x + y] for x, y in u])

        controller = Controller(u_d, y_d, 1, 1)
        controller.update(u_ini, y_ini)
        u_star = controller.apply(target, u_0)
        np.testing.assert_array_almost_equal(u_star, u)
    

if __name__ == "__main__":
    unittest.main(verbosity=2)
