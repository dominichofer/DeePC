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
        mat = np.array([[1]])
        expected = np.array([[1]])
        np.testing.assert_array_equal(left_pseudoinverse(mat), expected)

    def test_2x2(self):
        mat = np.array([[1, 2], [3, 4]])
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(left_pseudoinverse(mat), expected)

    def test_3x2(self):
        mat = np.array([[1, 2], [3, 4], [5, 6]])
        expected = np.array(
            [[-1.33333333, -0.33333333, 0.66666667], [1.08333333, 0.33333333, -0.41666667]]
        )
        np.testing.assert_array_almost_equal(left_pseudoinverse(mat), expected)


class TestRightPseudoinverse(unittest.TestCase):
    def test_1x1(self):
        mat = np.array([[1]])
        expected = np.array([[1]])
        np.testing.assert_array_equal(right_pseudoinverse(mat), expected)

    def test_2x2(self):
        mat = np.array([[1, 2], [3, 4]])
        expected = np.array([[-2, 1], [1.5, -0.5]])
        np.testing.assert_array_almost_equal(right_pseudoinverse(mat), expected)

    def test_3x2(self):
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        expected = np.array(
            [[-0.94444444, 0.44444444], [-0.11111111, 0.11111111], [0.72222222, -0.22222222]]
        )
        np.testing.assert_array_almost_equal(right_pseudoinverse(mat), expected)


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
