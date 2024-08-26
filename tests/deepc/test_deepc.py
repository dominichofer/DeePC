import unittest
import numpy as np
from deepc import linear_chirp, deePC, Controller, DiscreteLTI


def lti_1D_input_1D_output() -> DiscreteLTI:
    "LTI system with 1D input and 1D output"
    system = DiscreteLTI(
        A=[[0.9, -0.2], [0.7, 0.1]],
        B=[[0.1], [0]],
        C=[[1, 0]],
        D=[[0.1]],
        x_ini=[1, 1],
    )
    assert system.input_dim == 1
    assert system.output_dim == 1
    assert system.is_controllable()
    assert system.is_observable()
    assert system.is_stable()
    return system


def lti_2D_input_3D_output() -> DiscreteLTI:
    "LTI system with 3D input and 2D output"
    system = DiscreteLTI(
        A=[[0.5, 0.1, 0], [0.1, 0.5, 0.1], [0, 0.1, 0.5]],
        B=[[0.1, 0], [0.1, 0.5], [0, 0.1]],
        C=[[1, 0, 0], [0, 1, 1], [0, 0, 1]],
        D=[[0, 0], [0, 0], [0, 0]],
        x_ini=[0, 0, 0],
    )
    assert system.input_dim == 2
    assert system.output_dim == 3
    assert system.is_controllable()
    assert system.is_observable()
    assert system.is_stable()
    return system


def gather_offline_data(system: DiscreteLTI) -> tuple:
    SAMPLES = 1_000
    if system.input_dim == 1:
        u_d = linear_chirp(0, SAMPLES / 2, SAMPLES)
    else:
        chirps = [linear_chirp(0, SAMPLES / 2, SAMPLES, 0.1 * i) for i in range(system.input_dim)]
        u_d = list(zip(*chirps))  # Transpose the list of chirp signals
    y_d = system.apply_multiple(u_d)
    return u_d, y_d


class Test_deePC_1D_input_1D_output(unittest.TestCase):
    def setUp(self):
        self.system = lti_1D_input_1D_output()

        # Offline data
        self.u_d, self.y_d = gather_offline_data(self.system)

        # Initial conditions
        self.u_ini = [1] * 20
        self.y_ini = self.system.apply_multiple(self.u_ini)

        # Reference trajectory
        self.target = [[3], [3]]

    def test_unconstrained(self):
        u_star = deePC(self.u_d, self.y_d, self.u_ini, self.y_ini, self.target)

        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.target)

    def test_constrained(self):
        u_star = deePC(
            self.u_d,
            self.y_d,
            self.u_ini,
            self.y_ini,
            self.target,
            control_constrain_fkt=lambda u: np.clip(u, -15, 15),
        )

        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.target)


class Test_deePC_2D_input_3D_output(unittest.TestCase):
    def setUp(self):
        self.system = lti_2D_input_3D_output()

        # Offline data
        self.u_d, self.y_d = gather_offline_data(self.system)

        # Initial conditions
        self.u_ini = [(1, 1)] * 20
        self.y_ini = self.system.apply_multiple(self.u_ini)

        # Reference trajectory
        self.r = [(0.21, 0.9, 0.36)]

    def test_unconstrained(self):
        u_star = deePC(self.u_d, self.y_d, self.u_ini, self.y_ini, self.r)

        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.r, decimal=2)

    def test_constrained(self):
        u_star = deePC(
            self.u_d,
            self.y_d,
            self.u_ini,
            self.y_ini,
            self.r,
            control_constrain_fkt=lambda u: np.clip(u, -15, 15),
        )

        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.r, decimal=2)


class deePC_simple_system_1D_input_1D_output(unittest.TestCase):
    def test_int(self):
        u_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        y_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        u_ini = [2]
        y_ini = [2]
        u = [[4]]
        r = [4]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_int_with_2_targets(self):
        u_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        y_d = [1, 2, 3, 4, -5, 6, 7, 8, 9, 10]
        u_ini = [2]
        y_ini = [2]
        u = [[4], [4]]
        r = [4, 4]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_tuple(self):
        u_d = [(1,), (2,), (3,), (4,), (5,)]
        y_d = [(1,), (2,), (3,), (4,), (5,)]
        u_ini = [(2,)]
        y_ini = [(2,)]
        u = [(4,)]
        r = [(4,)]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_tuple_with_2_targets(self):
        u_d = [(1,), (2,), (3,), (4,), (-5,), (6,), (7,), (8,), (9,), (10,)]
        y_d = [(1,), (2,), (3,), (4,), (-5,), (6,), (7,), (8,), (9,), (10,)]
        u_ini = [(2,)]
        y_ini = [(2,)]
        u = [(4,), (4,)]
        r = [(4,), (4,)]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_list(self):
        u_d = [[1], [2], [3], [4], [5]]
        y_d = [[1], [2], [3], [4], [5]]
        u_ini = [[2]]
        y_ini = [[2]]
        u = [[4]]
        r = [[4]]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_list_with_3_targets(self):
        u_d = [[1], [2], [3], [4], [-5], [6], [7], [8], [9], [10]]
        y_d = [[1], [2], [3], [4], [-5], [6], [7], [8], [9], [10]]
        u_ini = [[2]]
        y_ini = [[2]]
        u = [[4], [4], [4]]
        r = [[4], [4], [4]]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_list_of_nparray(self):
        u_d = [np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
        y_d = [np.array([1]), np.array([2]), np.array([3]), np.array([4]), np.array([5])]
        u_ini = [np.array([2])]
        y_ini = [np.array([2])]
        u = [np.array([4])]
        r = [np.array([4])]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_nparray(self):
        u_d = np.array([1, 2, 3, 4, 5])
        y_d = np.array([1, 2, 3, 4, 5])
        u_ini = np.array([2])
        y_ini = np.array([2])
        u = np.array([[4]])
        r = np.array([4])

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)


class deePC_simple_system_2D_input_3D_output(unittest.TestCase):
    def test_tuple(self):
        u_d = [(x, y) for x in range(3) for y in range(3)]
        y_d = [(x, y, x + y) for x, y in u_d]
        u_ini = [(2, 2)]
        y_ini = [(x, y, x + y) for x, y in u_ini]
        u = [(1, 3)]
        r = [(x, y, x + y) for x, y in u]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_list(self):
        u_d = [[x, y] for x in range(3) for y in range(3)]
        y_d = [[x, y, x + y] for x, y in u_d]
        u_ini = [[2, 2]]
        y_ini = [[x, y, x + y] for x, y in u_ini]
        u = [[1, 3]]
        r = [[x, y, x + y] for x, y in u]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_list_of_nparray(self):
        u_d = [np.array([x, y]) for x in range(3) for y in range(3)]
        y_d = [np.array([x, y, x + y]) for x, y in u_d]
        u_ini = [np.array([2, 2])]
        y_ini = [np.array([x, y, x + y]) for x, y in u_ini]
        u = [np.array([1, 3])]
        r = [np.array([x, y, x + y]) for x, y in u]

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)

    def test_nparray(self):
        u_d = np.array([[x, y] for x in range(3) for y in range(3)])
        y_d = np.array([[x, y, x + y] for x, y in u_d])
        u_ini = np.array([[2, 2]])
        y_ini = np.array([[x, y, x + y] for x, y in u_ini])
        u = np.array([[1, 3]])
        r = np.array([[x, y, x + y] for x, y in u])

        u_star = deePC(u_d, y_d, u_ini, y_ini, r)

        np.testing.assert_array_almost_equal(u_star, u)


def warm_up_controller(controller: Controller, system: DiscreteLTI, u: list | np.ndarray) -> None:
    "Warm up the controller until it is initialized"
    while not controller.is_initialized():
        y = system.apply(u)
        controller.update(u, y)


def control_system(controller: Controller, system: DiscreteLTI, r: list, time_steps: int) -> float:
    """
    Control the system for a given number of time steps.
    Returns the output of the system after the last time step.
    """
    for _ in range(time_steps):
        u = controller.apply(r)[0]
        y = system.apply(u)
        controller.update(u, y)
    return y


class TestController(unittest.TestCase):
    def test_unconstrained_2D_LTI(self):
        system = lti_1D_input_1D_output()
        u_d, y_d = gather_offline_data(system)
        T_ini = 20
        target = [[10], [10], [10]]
        target_len = len(target)

        controller = Controller(u_d, y_d, T_ini, target_len)
        warm_up_controller(controller, system, u=[1])
        y = control_system(controller, system, target, time_steps=2 * T_ini)

        np.testing.assert_array_almost_equal(y, target[0])

    def test_constrained_2D_LTI(self):
        system = lti_1D_input_1D_output()
        u_d, y_d = gather_offline_data(system)
        T_ini = 20
        target = [[10]]
        target_len = len(target)

        controller = Controller(u_d, y_d, T_ini, target_len, control_constrain_fkt=lambda u: np.clip(u, 0, 25))
        warm_up_controller(controller, system, u=[1])
        y = control_system(controller, system, target, time_steps=2 * T_ini)

        np.testing.assert_array_almost_equal(y, target[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
