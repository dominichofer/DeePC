import unittest
import numpy as np
from deepc.math import linear_chirp, left_pseudoinverse, right_pseudoinverse, hankel_matrix


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
        np.testing.assert_array_equal(hankel_matrix(1, [(1, 2), (3, 4), (5, 6)]), [[1, 3, 5], [2, 4, 6]])

    def test_size_3_and_2_rows(self):
        np.testing.assert_array_equal(hankel_matrix(2, [(1, 2), (3, 4), (5, 6)]), [[1, 3], [2, 4], [3, 5], [4, 6]])

    def test_size_3_and_3_rows(self):
        np.testing.assert_array_equal(hankel_matrix(3, [(1, 2), (3, 4), (5, 6)]), [[1], [2], [3], [4], [5], [6]])


class TestLinearChirp(unittest.TestCase):
    def test_len(self):
        chirp = linear_chirp(0, 100, 1000)
        self.assertEqual(len(chirp), 1000)

    def test_start(self):
        chirp = linear_chirp(0, 100, 1000)
        self.assertAlmostEqual(chirp[0], 0)

    def test_end(self):
        chirp = linear_chirp(0, 100, 1000)
        self.assertAlmostEqual(chirp[-1], 0)

    def test_symmetry(self):
        chirp1 = linear_chirp(0, 100, 1_000)
        chirp2 = [-x for x in reversed(linear_chirp(100, 0, 1_000))]
        np.testing.assert_allclose(chirp1, chirp2, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
