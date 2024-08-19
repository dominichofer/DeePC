import unittest
import numpy as np
from deepc.lti import DiscreteLTI


class TestDiscreteLTI(unittest.TestCase):
    def test_dimensions_1D(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[1], [1]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[0, 0],
        )
        self.assertEqual(system.input_dim, 1)
        self.assertEqual(system.output_dim, 1)

    def test_dimensions_2D(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[1, 0], [0, 1]],
            C=[[1, 0], [0, 1]],
            D=[[0, 0], [0, 0]],
            x_ini=[0, 0],
        )
        self.assertEqual(system.input_dim, 2)
        self.assertEqual(system.output_dim, 2)

    def test_controllable(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[1], [1]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[0, 0],
        )
        self.assertTrue(system.is_controllable())

    def test_uncontrollable(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[0], [0]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[0, 0],
        )
        self.assertFalse(system.is_controllable())

    def test_observable(self):
        system = DiscreteLTI(
            A=[[1, 1], [0, 1]],
            B=[[1], [0]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[0, 0],
        )
        self.assertTrue(system.is_observable())

    def test_unobservable(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[1], [0]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[0, 0],
        )
        self.assertFalse(system.is_observable())

    def test_stable(self):
        system = DiscreteLTI(
            A=[[0.5, 0], [0.6, 0]],
            B=[[1], [1]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[0, 0],
        )
        self.assertTrue(system.is_stable())

    def test_unstable(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[1], [1]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[0, 0],
        )
        self.assertFalse(system.is_stable())

    def test_apply_1d(self):
        system = DiscreteLTI(
            A=[[1, 1], [1, 1]],
            B=[[1], [1]],
            C=[[1, 1]],
            D=[[1]],
            x_ini=[1, 1],
        )
        y = system.apply(1)
        self.assertAlmostEqual(y, 7)

    def test_apply_2d(self):
        system = DiscreteLTI(
            A=[[1, 1], [1, 1]],
            B=[[1, 1], [1, 1]],
            C=[[1, 1]],
            D=[[1, 1]],
            x_ini=[1, 1],
        )
        y = system.apply([1, 1])
        self.assertAlmostEqual(y, 10)

    def test_apply_multiple_1d(self):
        system = DiscreteLTI(
            A=[[1, 1], [1, 1]],
            B=[[1], [1]],
            C=[[1, 1]],
            D=[[1]],
            x_ini=[1, 1],
        )
        y = system.apply_multiple([1, 2, 3])
        np.testing.assert_almost_equal(y, [7, 18, 41])

    def test_apply_multiple_2d(self):
        system = DiscreteLTI(
            A=[[1, 1], [1, 1]],
            B=[[1, 1], [1, 1]],
            C=[[1, 1]],
            D=[[1, 1]],
            x_ini=[1, 1],
        )
        y = system.apply_multiple([[1, 1], [2, 2]])
        np.testing.assert_almost_equal(y, [10, 28])


if __name__ == "__main__":
    unittest.main(verbosity=2)
