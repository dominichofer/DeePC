import unittest
import numpy as np
from deepc.math import clamp, left_pseudoinverse, right_pseudoinverse, hankel_matrix


class TestClamp(unittest.TestCase):
    def test_value_below(self):
        self.assertEqual(clamp(0, 1, 2), 1)

    def test_value_above(self):
        self.assertEqual(clamp(3, 1, 2), 2)

    def test_value_inside(self):
        self.assertEqual(clamp(1, 1, 2), 1)


class TestLeftPseudoinverse(unittest.TestCase):
    def test_1x1(self):
        mat = np.array([[2]])
        i = left_pseudoinverse(mat) @ mat
        np.testing.assert_array_equal(i, np.eye(1))

    def test_2x2(self):
        mat = np.array([[1, 2], [3, 4]])
        i = left_pseudoinverse(mat) @ mat
        np.testing.assert_array_almost_equal(i, np.eye(2))

    def test_3x2(self):
        mat = np.array([[1, 2], [3, 4], [5, 6]])
        i = left_pseudoinverse(mat) @ mat
        np.testing.assert_array_almost_equal(i, np.eye(2))


class TestRightPseudoinverse(unittest.TestCase):
    def test_1x1(self):
        mat = np.array([[2]])
        i = mat @ right_pseudoinverse(mat)
        np.testing.assert_array_equal(i, np.eye(1))

    def test_2x2(self):
        mat = np.array([[1, 2], [3, 4]])
        i = mat @ right_pseudoinverse(mat)
        np.testing.assert_array_almost_equal(i, np.eye(2))

    def test_3x2(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        i = mat @ right_pseudoinverse(mat)
        np.testing.assert_array_almost_equal(i, np.eye(2))


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
