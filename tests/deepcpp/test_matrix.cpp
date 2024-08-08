#include "pch.h"
#include "Matrix.h"
#include <vector>
#include <iostream>

TEST(DenseMatrixTest, Constructor)
{
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	DenseMatrix m(data, 2);

	EXPECT_EQ(m.rows(), 3);
	EXPECT_EQ(m.cols(), 2);
	EXPECT_EQ(m(0, 0), 1.0);
	EXPECT_EQ(m(0, 1), 2.0);
	EXPECT_EQ(m(1, 0), 3.0);
	EXPECT_EQ(m(1, 1), 4.0);
	EXPECT_EQ(m(2, 0), 5.0);
	EXPECT_EQ(m(2, 1), 6.0);
}

TEST(DenseMatrixTest, Zeros)
{
	DenseMatrix m = DenseMatrix::Zeros(2, 3);
	EXPECT_EQ(m.rows(), 2);
	EXPECT_EQ(m.cols(), 3);
	EXPECT_EQ(m(0, 0), 0.0);
	EXPECT_EQ(m(0, 1), 0.0);
	EXPECT_EQ(m(0, 2), 0.0);
	EXPECT_EQ(m(1, 0), 0.0);
	EXPECT_EQ(m(1, 1), 0.0);
	EXPECT_EQ(m(1, 2), 0.0);
}

TEST(DenseMatrixTest, MatrixVectorMultiplication)
{
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	DenseMatrix m(data, 2);
	std::vector<double> v = { 1.0, 2.0 };

	std::vector<double> result = m * v;
	std::vector<double> expected = { 5.0, 11.0, 17.0 };
	EXPECT_EQ(result, expected);
}

TEST(DenseMatrixTest, MatrixMatrixMultiplication)
{
	std::vector<double> data1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	DenseMatrix m1(data1, 2);
	std::vector<double> data2 = { 1.0, 2.0, 3.0, 4.0 };
	DenseMatrix m2(data2, 2);

	DenseMatrix result = m1 * m2;
	std::vector<double> expected_data = { 7.0, 10.0, 15.0, 22.0, 23.0, 34.0 };
	DenseMatrix expected(expected_data, 2);
	EXPECT_EQ(result, expected);
}

TEST(DenseMatrixTest, MatrixAddition)
{
	std::vector<double> data1 = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	DenseMatrix m1(data1, 2);
	std::vector<double> data2 = { 2.0, 3.0, 4.0, 5.0, 6.0, 7.0 };
	DenseMatrix m2(data2, 2);

	DenseMatrix result = m1 + m2;
	std::vector<double> expected_data = { 3.0, 5.0, 7.0, 9.0, 11.0, 13.0 };
	DenseMatrix expected(expected_data, 2);
	EXPECT_EQ(result, expected);
}

TEST(DenseMatrixTest, MatrixNorm)
{
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	DenseMatrix m(data, 2);

	double n = norm(m);
	EXPECT_DOUBLE_EQ(n, norm(data));
}

TEST(DiagonalMatrixTest, Constructor)
{
	DiagonalMatrix m(2.0, 3);
	std::vector<double> v = { 2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0 };
	DenseMatrix expected(v, 3);
	EXPECT_EQ(m, expected);
}

TEST(DiagonalMatrixTest, IdentityMatrix)
{
	DiagonalMatrix m = IdentityMatrix(3);
	std::vector<double> v = { 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
	DenseMatrix expected(v, 3);
	EXPECT_EQ(m, expected);
}

TEST(DiagonalMatrixTest, ZeroMatrix)
{
	DiagonalMatrix m = ZeroMatrix(3);
	std::vector<double> v = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
	DenseMatrix expected(v, 3);
	EXPECT_EQ(m, expected);
}

TEST(TransposedMatrixTest, Constructor)
{
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0 };
	DenseMatrix m(data, 2);
	TransposedMatrix t(m);

	std::vector<double> expected_data = { 1.0, 3.0, 2.0, 4.0 };
	DenseMatrix expected(expected_data, 2);
	EXPECT_TRUE(t == expected);
}