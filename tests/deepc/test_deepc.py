import unittest
import numpy as np
from deepc import deePC
from helpers import create_1D_in_1D_out_LTI, create_2D_in_3D_out_LTI, gather_offline_data


class Test_1D_in_1D_out_LTI(unittest.TestCase):
    def setUp(self):
        self.system = create_1D_in_1D_out_LTI()

        # Offline data
        self.u_d, self.y_d = gather_offline_data(self.system, samples=1_000)

        # Initial conditions
        self.u_ini = [1] * 20
        self.y_ini = self.system.apply_multiple(self.u_ini)

        # Reference trajectory
        self.target = [[3], [3]]

    def test_unconstrained(self):
        u_star = deePC(self.u_d, self.y_d, self.u_ini, self.y_ini, self.target)

        # Apply the control input to the system
        # to see if the output matches the target
        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.target)

    def test_constrained(self):
        u_star = deePC(
            self.u_d,
            self.y_d,
            self.u_ini,
            self.y_ini,
            self.target,
            input_constrain_fkt=lambda u: np.clip(u, -15, 15),
        )

        # Apply the control input to the system
        # to see if the output matches the target
        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.target)


class Test_2D_in_3D_out_LTI(unittest.TestCase):
    def setUp(self):
        self.system = create_2D_in_3D_out_LTI()

        # Offline data
        self.u_d, self.y_d = gather_offline_data(self.system, samples=1_000)

        # Initial conditions
        self.u_ini = [(1, 1)] * 20
        self.y_ini = self.system.apply_multiple(self.u_ini)

        # Reference trajectory
        self.target = [(0.21, 0.9, 0.36)]

    def test_unconstrained(self):
        u_star = deePC(self.u_d, self.y_d, self.u_ini, self.y_ini, self.target)

        # Apply the control input to the system
        # to see if the output matches the target
        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.target, decimal=2)

    def test_constrained(self):
        u_star = deePC(
            self.u_d,
            self.y_d,
            self.u_ini,
            self.y_ini,
            self.target,
            input_constrain_fkt=lambda u: np.clip(u, -15, 15),
        )

        # Apply the control input to the system
        # to see if the output matches the target
        y_star = self.system.apply_multiple(u_star)
        np.testing.assert_array_almost_equal(y_star, self.target, decimal=2)


class Test_types(unittest.TestCase):
    """
    Test the different types of input data that can be used with deePC
    on a trivial 1D input 1D output system.
    """

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


class Test_2D_types(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
