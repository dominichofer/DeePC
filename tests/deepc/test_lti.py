import unittest
from deepc.lti import DiscreteLTI, LaggedLTI


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
            x_ini=[1, 1],
        )
        self.assertFalse(system.is_controllable())

    def test_observable(self):
        system = DiscreteLTI(
            A=[[1, 1], [0, 1]],
            B=[[1], [0]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[1, 1],
        )
        self.assertTrue(system.is_observable())

    def test_unobservable(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[1], [0]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[1, 1],
        )
        self.assertFalse(system.is_observable())

    def test_stable(self):
        system = DiscreteLTI(
            A=[[0.5, 0], [0.6, 0]],
            B=[[1], [1]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[1, 1],
        )
        self.assertTrue(system.is_stable())

    def test_unstable(self):
        system = DiscreteLTI(
            A=[[2, 0], [0, 3]],
            B=[[1], [1]],
            C=[[1, 0]],
            D=[[0]],
            x_ini=[1, 1],
        )
        self.assertFalse(system.is_stable())

    def test_apply(self):
        system = DiscreteLTI(
            A=[[0.5, 0], [0.6, 0.1]],
            B=[[0.1], [1]],
            C=[[1, 0]],
            D=[[0.5]],
            x_ini=[1, 1],
        )
        y = system.apply(1)
        self.assertAlmostEqual(y, 1.1)

    def test_apply_multiple(self):
        system = DiscreteLTI(
            A=[[0.5, 0], [0.6, 0.1]],
            B=[[0.1], [1]],
            C=[[1, 0]],
            D=[[0.5]],
            x_ini=[1, 1],
        )
        y = system.apply_multiple([1, 2, 3])
        expected = [1.1, 1.5, 2.05]
        self.assertAlmostEqual(y, expected)

    def test_apply_control_2d(self):
        system = DiscreteLTI(
            A=[[0.5, 0], [0.6, 0.1]],
            B=[[1, 0.1], [0.1, 1]],
            C=[[1, 0]],
            D=[[0.5, 0]],
            x_ini=[1, 1],
        )
        y = system.apply([1, 2])
        self.assertAlmostEqual(y, 2.2)


class TestLaggedLTI(unittest.TestCase):
    def test_lag_0(self):
        system = LaggedLTI(lag=0, x_ini=[1])
        self.assertTrue(system.is_controllable())
        self.assertTrue(system.is_observable())
        self.assertTrue(system.is_stable())

        y = system.apply(2)
        self.assertAlmostEqual(y, 2)

    def test_lag_1(self):
        system = LaggedLTI(lag=1, x_ini=[1, 2])
        self.assertTrue(system.is_controllable())
        self.assertTrue(system.is_observable())
        self.assertTrue(system.is_stable())

        y = system.apply(3)
        self.assertAlmostEqual(y, 2)
        y = system.apply(4)
        self.assertAlmostEqual(y, 3)

    def test_lag_2(self):
        system = LaggedLTI(lag=2, x_ini=[1, 2, 3])
        self.assertTrue(system.is_controllable())
        self.assertTrue(system.is_observable())
        self.assertTrue(system.is_stable())

        y = system.apply(4)
        self.assertAlmostEqual(y, 2)
        y = system.apply(5)
        self.assertAlmostEqual(y, 3)

    def test_lag_5(self):
        system = LaggedLTI(lag=5, x_ini=[1, 2, 3, 4, 5, 6])
        self.assertTrue(system.is_controllable())
        self.assertTrue(system.is_observable())
        self.assertTrue(system.is_stable())

        for i in range(10):
            y = system.apply(7 + i)
            self.assertAlmostEqual(y, 2 + i)


if __name__ == "__main__":
    unittest.main(verbosity=2)
