#include "gtest/gtest.h"
#include "matrix.h"
#include <vector>
#include <iostream>

TEST(DenseMatrix, Constructor)
{
	std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	DenseMatrix m(data, /*cols*/ 2);

	EXPECT_EQ(m.rows(), 3);
	EXPECT_EQ(m.cols(), 2);
	EXPECT_EQ(m(0, 0), 1.0);
	EXPECT_EQ(m(0, 1), 2.0);
	EXPECT_EQ(m(1, 0), 3.0);
	EXPECT_EQ(m(1, 1), 4.0);
	EXPECT_EQ(m(2, 0), 5.0);
	EXPECT_EQ(m(2, 1), 6.0);
}

TEST(DenseMatrix, Zeros)
{
	DenseMatrix m = DenseMatrix::Zeros(1, 2);
	EXPECT_EQ(m.rows(), 1);
	EXPECT_EQ(m.cols(), 2);
	EXPECT_EQ(m(0, 0), 0.0);
	EXPECT_EQ(m(0, 1), 0.0);
}

TEST(DenseMatrix, MatrixVectorMultiplication)
{
	std::vector<double> data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	DenseMatrix mat(data, 2);
	std::vector<double> vec = {1.0, 2.0};

	std::vector<double> result = mat * vec;

	std::vector<double> expected = {5.0, 11.0, 17.0};
	EXPECT_EQ(result, expected);
}

TEST(DenseMatrix, MatrixMatrixMultiplication)
{
	std::vector<double> data1 = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
	DenseMatrix mat1(data1, 2);
	std::vector<double> data2 = {1.0, 2.0, 3.0, 4.0};
	DenseMatrix mat2(data2, 2);

	DenseMatrix result = mat1 * mat2;

	std::vector<double> expected_data = {7.0, 10.0, 15.0, 22.0, 23.0, 34.0};
	DenseMatrix expected(expected_data, 2);
	EXPECT_EQ(result, expected);
}

TEST(DenseMatrix, MatrixAddition)
{
	std::vector<double> data1 = {1.0, 2.0};
	DenseMatrix m1(data1, 2);
	std::vector<double> data2 = {3.0, 4.0};
	DenseMatrix m2(data2, 2);

	DenseMatrix result = m1 + m2;
	std::vector<double> expected_data = {4.0, 6.0};
	DenseMatrix expected(expected_data, 2);
	EXPECT_EQ(result, expected);
}

TEST(DenseMatrix, MatrixSubtraction)
{
	std::vector<double> data1 = {1.0, 2.0};
	DenseMatrix m1(data1, 2);
	std::vector<double> data2 = {3.0, 4.0};
	DenseMatrix m2(data2, 2);

	DenseMatrix result = m1 - m2;
	std::vector<double> expected_data = {-2.0, -2.0};
	DenseMatrix expected(expected_data, 2);
	EXPECT_EQ(result, expected);
}

TEST(DiagonalMatrix, Constructor)
{
	DiagonalMatrix m({1.0, 2.0, 3.0});

	std::vector<double> data = {1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0};
	DenseMatrix expected(data, 3);
	EXPECT_EQ(m, expected);
}

TEST(DiagonalMatrix, IdentityMatrix)
{
	DiagonalMatrix m = IdentityMatrix(3);
	std::vector<double> data = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
	DenseMatrix expected(data, 3);
	EXPECT_EQ(m, expected);
}

TEST(DiagonalMatrix, ZeroMatrix)
{
	DiagonalMatrix m = ZeroMatrix(3);
	std::vector<double> data = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
	DenseMatrix expected(data, 3);
	EXPECT_EQ(m, expected);
}

TEST(TransposedMatrix, transposed)
{
	std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
	DenseMatrix mat(data, 2);

	auto t = transposed(mat);

	std::vector<double> expected_data = {1.0, 3.0, 2.0, 4.0};
	DenseMatrix expected(expected_data, 2);
	EXPECT_TRUE(t == expected);
}

// TODO: Implement the HankelMatrix tests
// TODO: Implement the SubMatrix tests
// TODO: Implement the VStackMatrix tests
