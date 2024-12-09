#include "gtest/gtest.h"
#include <Eigen/Dense>
#include "algorithm.h"
#include "helpers.h"

using Eigen::VectorXd;

TEST(Algorithm, To_String)
{
	auto v = Vectors({1, 2}, {3, 4});
	auto res = to_string(v);
	EXPECT_EQ(res, "[[1.000000, 2.000000][3.000000, 4.000000]]");
}

TEST(Algorithm, Clamp1)
{
	auto v = Vector(1, 2, 3, 4);
	auto res = clamp(v, 2, 3);
	EXPECT_EQ(res, Vector(2, 2, 3, 3));
}

TEST(Algorithm, Clamp2)
{
	auto v = Vector(1, 2, 3, 4);
	auto low = Vector(2, 2, 2, 2);
	auto high = Vector(3, 3, 3, 3);
	
	auto res = clamp(v, 2, 3);

	EXPECT_EQ(res, Vector(2, 2, 3, 3));
}

TEST(Algorithm, Concat_VectorXd)
{
	auto l = Vector(1, 2);
	auto r = Vector(3, 4);

	auto res = concat(l, r);

	EXPECT_EQ(res, Vector(1, 2, 3, 4));
}

TEST(Algorithm, Concat_std_vector)
{
	auto v = std::vector{Vector(1, 2), Vector(3, 4)};
	auto res = concat(v);

	EXPECT_EQ(res, Vector(1, 2, 3, 4));
}

TEST(Algorithm, Concat_std_vector_VectorXd)
{
	auto l = Vectors({1, 2}, {3, 4});
	auto r = Vectors({5, 6}, {7, 8});

	auto res = concat(l, r);

	EXPECT_EQ(res, Vector(1, 2, 3, 4, 5, 6, 7, 8));
}

TEST(Algorithm, Split)
{
	auto vec = Vector(1, 2, 3, 4, 5, 6);
	auto res = split(vec, 2);

	EXPECT_EQ(res, Vectors({1, 2, 3}, {4, 5, 6}));
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

TEST(HankelMatrix_1D_data, Size_1_and_1_row)
{
	auto res = HankelMatrix(1, Vectors({1}));
	auto expected = Matrix({1});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_1D_data, Size_2_and_1_row)
{
	auto res = HankelMatrix(1, Vectors({1}, {2}));
	auto expected = Matrix({1, 2});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_1D_data, Size_2_and_2_row)
{
	auto res = HankelMatrix(2, Vectors({1}, {2}));
	auto expected = Matrix({1}, {2});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_1D_data, Size_3_and_1_row)
{
	auto res = HankelMatrix(1, Vectors({1}, {2}, {3}));
	auto expected = Matrix({1, 2, 3});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_1D_data, Size_3_and_2_row)
{
	auto res = HankelMatrix(2, Vectors({1}, {2}, {3}));
	auto expected = Matrix({1, 2}, {2, 3});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_1D_data, Size_3_and_3_row)
{
	auto res = HankelMatrix(3, Vectors({1}, {2}, {3}));
	auto expected = Matrix({1}, {2}, {3});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_2D_data, Size_1_and_1_row)
{
	auto res = HankelMatrix(1, Vectors({1, 2}));
	auto expected = Matrix({1}, {2});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_2D_data, Size_2_and_1_row)
{
	auto res = HankelMatrix(1, Vectors({1, 2}, {3, 4}));
	auto expected = Matrix({1, 3}, {2, 4});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_2D_data, Size_2_and_2_row)
{
	auto res = HankelMatrix(2, Vectors({1, 2}, {3, 4}));
	auto expected = Matrix({1}, {2}, {3}, {4});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_2D_data, Size_3_and_1_row)
{
	auto res = HankelMatrix(1, Vectors({1, 2}, {3, 4}, {5, 6}));
	auto expected = Matrix({1, 3, 5}, {2, 4, 6});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_2D_data, Size_3_and_2_row)
{
	auto res = HankelMatrix(2, Vectors({1, 2}, {3, 4}, {5, 6}));
	auto expected = Matrix({1, 3}, {2, 4}, {3, 5}, {4, 6});
	EXPECT_EQ(res, expected);
}

TEST(HankelMatrix_2D_data, Size_3_and_3_row)
{
	auto res = HankelMatrix(3, Vectors({1, 2}, {3, 4}, {5, 6}));
	auto expected = Matrix({1}, {2}, {3}, {4}, {5}, {6});
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
	EXPECT_NEAR(chirp[0], 0, 1e-12);
}

TEST(LinearChirp, End)
{
	auto chirp = linear_chirp(0, 100, 1'000);
	EXPECT_NEAR(chirp[999], 0, 1e-12);
}

TEST(LinearChirp, Symmetry)
{
	auto chirp1 = linear_chirp(0, 100, 1'000);
	auto chirp2 = linear_chirp(100, 0, 1'000);
	for (int i = 0; i < 1'000; ++i)
		EXPECT_NEAR(chirp1[i], -chirp2[999 - i], 1e-12);
}
