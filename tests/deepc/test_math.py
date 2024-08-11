import unittest
import numpy as np
from deepc.math import clamp, linear_chirp, left_pseudoinverse, right_pseudoinverse, hankel_matrix


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


class TestHankelMatrix_1D_data(unittest.TestCase):
    def test_size_1_and_1_row(self):
        np.testing.assert_array_equal(hankel_matrix(1, [1]), [[1]])

    def test_size_2_and_1_row(self):
        np.testing.assert_array_equal(hankel_matrix(1, [1, 2]), [[1, 2]])

    def test_size_2_and_2_row(self):
        np.testing.assert_array_equal(hankel_matrix(2, [1, 2]), [[1], [2]])

    def test_size_3_and_1_rows(self):
        np.testing.assert_array_equal(hankel_matrix(1, [1, 2, 3]), [[1, 2, 3]])

    def test_size_3_and_2_rows(self):
        np.testing.assert_array_equal(hankel_matrix(2, [1, 2, 3]), [[1, 2], [2, 3]])

    def test_size_3_and_3_rows(self):
        np.testing.assert_array_equal(hankel_matrix(3, [1, 2, 3]), [[1], [2], [3]])


class TestHankelMatrix_2D_data(unittest.TestCase):
    def test_size_1_and_1_row(self):
        np.testing.assert_array_equal(hankel_matrix(1, [(1, 2)]), [[1], [2]])

    def test_size_2_and_1_row(self):
        np.testing.assert_array_equal(hankel_matrix(1, [(1, 2), (3, 4)]), [[1, 3], [2, 4]])

    def test_size_2_and_2_row(self):
        np.testing.assert_array_equal(hankel_matrix(2, [(1, 2), (3, 4)]), [[1], [2], [3], [4]])

    def test_size_3_and_1_rows(self):
        np.testing.assert_array_equal(
            hankel_matrix(1, [(1, 2), (3, 4), (5, 6)]), [[1, 3, 5], [2, 4, 6]]
        )

    def test_size_3_and_2_rows(self):
        np.testing.assert_array_equal(
            hankel_matrix(2, [(1, 2), (3, 4), (5, 6)]), [[1, 3], [2, 4], [3, 5], [4, 6]]
        )

    def test_size_3_and_3_rows(self):
        np.testing.assert_array_equal(
            hankel_matrix(3, [(1, 2), (3, 4), (5, 6)]), [[1], [2], [3], [4], [5], [6]]
        )


class TestLinearChirp(unittest.TestCase):
    def test_len(self):
        signal = linear_chirp(100, 0, 1000)
        self.assertEqual(len(signal), 1000)

    def test_start(self):
        signal = linear_chirp(100, 0, 1000)
        self.assertAlmostEqual(signal[0], 0)

    def test_end(self):
        signal = linear_chirp(100, 0, 1000)
        self.assertAlmostEqual(signal[-1], 0)

    def test_symmetry(self):
        signal1 = linear_chirp(100, 0, 1000)
        signal2 = [-x for x in reversed(linear_chirp(0, 100, 1000))]
        np.testing.assert_allclose(signal1, signal2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
