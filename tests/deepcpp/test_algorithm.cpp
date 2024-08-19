#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "algorithm.h"
#include "helpers.h"

using Eigen::VectorXd;

TEST(Algorithm, Concat)
{
	auto l = Vector(1, 2);
	auto r = Vector(3, 4);

	auto res = concat(l, r);

	EXPECT_EQ(res, Vector(1, 2, 3, 4));
}

TEST(Algorithm, Vstack2)
{
	auto upper = Vector(1, 2);
	auto lower = Vector(3, 4);

	auto res = vstack(upper, lower);

	EXPECT_EQ(res, Vector(1, 2, 3, 4));
}

TEST(Algorithm, Vstack3)
{
	auto upper = Vector(1, 2);
	auto middle = Vector(3, 4);
	auto lower = Vector(5, 6);

	auto res = vstack(upper, middle, lower);

	EXPECT_EQ(res, Vector(1, 2, 3, 4, 5, 6));
}

TEST(HankelMatrix, Size_1_and_1_row)
{
	auto res = HankelMatrix(1, Vector(1));
	auto expected = Matrix({1});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix, Size_2_and_1_row)
{
	auto res = HankelMatrix(1, Vector(1, 2));
	auto expected = Matrix({1, 2});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix, Size_2_and_2_row)
{
	auto res = HankelMatrix(2, Vector(1, 2));
	auto expected = Matrix({1}, {2});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix, Size_3_and_1_row)
{
	auto res = HankelMatrix(1, Vector(1, 2, 3));
	auto expected = Matrix({1, 2, 3});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix, Size_3_and_2_row)
{
	auto res = HankelMatrix(2, Vector(1, 2, 3));
	auto expected = Matrix({1, 2}, {2, 3});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix, Size_3_and_3_row)
{
	auto res = HankelMatrix(3, Vector(1, 2, 3));
	auto expected = Matrix({1}, {2}, {3});
	EXPECT_EQ(res, expected);
}

TEST(Algorithm, ProjectedGradientMethod)
{
	auto mat = Matrix({1, 0}, {0, 1});
	auto initial_guess = Vector(1, 1);
	auto target = Vector(0, 0);
	auto projection = [](const VectorXd& x) { return x; };

	auto res = projected_gradient_method(mat, initial_guess, target, projection);

	EXPECT_EQ(res, Vector(0, 0));
}

TEST(LinearChirp, Size)
{
	auto chirp = linear_chirp(0, 100, 1'000);
	EXPECT_EQ(chirp.size(), 1'000);
}

TEST(LinearChirp, Start)
{
	auto chirp = linear_chirp(0, 100, 1'000);
	EXPECT_NEAR(chirp(0), 0, 1e-6);
}

TEST(LinearChirp, End)
{
	auto chirp = linear_chirp(0, 100, 1'000);
	EXPECT_NEAR(chirp(999), 0, 1e-6);
}

TEST(LinearChirp, Symmetry)
{
	auto chirp1 = linear_chirp(0, 100, 1'000);
	auto chirp2 = linear_chirp(100, 0, 1'000);
	for (int i = 0; i < 1'000; ++i)
		EXPECT_NEAR(chirp1(i), -chirp2(999 - i), 1e-12);
}
