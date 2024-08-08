#include "pch.h"
#include "Vector.h"
#include <vector>

TEST(VectorTest, Constructor)
{
    Vector v(3, 1.0);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v[0], 1.0);
    EXPECT_EQ(v[1], 1.0);
    EXPECT_EQ(v[2], 1.0);
}

TEST(VectorTest, Addition)
{
    Vector v1(3, 1.0);
    Vector v2(3, 2.0);
    Vector sum = v1 + v2;

    EXPECT_EQ(sum.size(), 3);
    EXPECT_EQ(sum[0], 3.0);
    EXPECT_EQ(sum[1], 3.0);
    EXPECT_EQ(sum[2], 3.0);
}

TEST(VectorTest, Subtraction)
{
    Vector v1(3, 2.0);
    Vector v2(3, 1.0);
    Vector diff = v1 - v2;

    EXPECT_EQ(diff.size(), 3);
    EXPECT_EQ(diff[0], 1.0);
    EXPECT_EQ(diff[1], 1.0);
    EXPECT_EQ(diff[2], 1.0);
}

TEST(VectorTest, ScalarMultiplication)
{
    Vector v(3, 2.0);
    Vector scaled = v * 3.0;

    EXPECT_EQ(scaled.size(), 3);
    EXPECT_EQ(scaled[0], 6.0);
    EXPECT_EQ(scaled[1], 6.0);
    EXPECT_EQ(scaled[2], 6.0);
}

TEST(VectorTest, ScalarDivision)
{
    Vector v(3, 6.0);
    Vector divided = v / 2.0;

    EXPECT_EQ(divided.size(), 3);
    EXPECT_EQ(divided[0], 3.0);
    EXPECT_EQ(divided[1], 3.0);
    EXPECT_EQ(divided[2], 3.0);
}

TEST(VectorTest, Inverse)
{
    Vector v(std::vector{ 1.0, 2.0, 4.0 });
    Vector inverse = inv(v);

    EXPECT_EQ(inverse.size(), 3);
    EXPECT_EQ(inverse[0], 1.0);
    EXPECT_EQ(inverse[1], 0.5);
    EXPECT_EQ(inverse[2], 0.25);
}

TEST(VectorTest, DotProduct)
{
    Vector v1(3, 2.0);
    Vector v2(3, 3.0);
    double dotProduct = dot(v1, v2);

    EXPECT_EQ(dotProduct, 18.0);
}

TEST(VectorTest, Norm)
{
    Vector v(std::vector{ 3.0, 4.0 });
    double Norm = norm(v);

    EXPECT_DOUBLE_EQ(Norm, 5.0);
}

TEST(VectorTest, Sum)
{
    Vector v(3, 2.0);
    double result = sum(v);

    EXPECT_DOUBLE_EQ(result, 6.0);
}

TEST(VectorTest, SizeMismatch)
{
    Vector v1(3, 2.0);
    Vector v2(4, 3.0);

    EXPECT_THROW(dot(v1, v2), std::runtime_error);
}

TEST(VectorTest, Concat)
{
	Vector v1(3, 2.0);
	Vector v2(4, 3.0);
	Vector v3 = concat(v1, v2);
    Vector expected(std::vector{ 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0 });

	EXPECT_EQ(v3, expected);
}