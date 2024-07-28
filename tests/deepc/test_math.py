import unittest
import numpy as np
from deepc import clamp, hankel_matrix


class TestClamp(unittest.TestCase):
    def test_value_below(self):
        self.assertEqual(clamp(0, 1, 2), 1)

    def test_value_above(self):
        self.assertEqual(clamp(3, 1, 2), 2)

    def test_value_inside(self):
        self.assertEqual(clamp(1, 1, 2), 1)


class TestHankelMatrix(unittest.TestCase):
    def test_size_1_1_row(self):
        vec = np.array([1])
        expected = np.matrix([[1]])
        np.testing.assert_array_equal(hankel_matrix(1, vec), expected)

    def test_size_2_1_row(self):
        vec = np.array([1, 2])
        expected = np.matrix([[1, 2]])
        np.testing.assert_array_equal(hankel_matrix(1, vec), expected)

    def test_size_3_2_rows(self):
        vec = np.array([1, 2, 3])
        expected = np.matrix([[1, 2], [2, 3]])
        np.testing.assert_array_equal(hankel_matrix(2, vec), expected)

    def test_size_4_2_rows(self):
        vec = np.array([1, 2, 3, 4])
        expected = np.matrix([[1, 2, 3], [2, 3, 4]])
        np.testing.assert_array_equal(hankel_matrix(2, vec), expected)


if __name__ == "__main__":
    unittest.main()