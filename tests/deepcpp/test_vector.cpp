#include "gtest/gtest.h"
#include "vector.h"
#include <vector>

TEST(Vector, Addition)
{
    Vector v1{1.0, 2.0};
    Vector v2{3.0, 4.0};

    Vector sum = v1 + v2;

    EXPECT_EQ(sum.size(), 2);
    EXPECT_EQ(sum[0], 4.0);
    EXPECT_EQ(sum[1], 6.0);
}

TEST(Vector, Subtraction)
{
    Vector v1{1.0, 2.0};
    Vector v2{3.0, 4.0};

    Vector sum = v1 - v2;

    EXPECT_EQ(sum.size(), 2);
    EXPECT_EQ(sum[0], -2.0);
    EXPECT_EQ(sum[1], -2.0);
}

TEST(Vector, ScalarMultiplicationRight)
{
    Vector v{1.0, 2.0};

    Vector scaled = v * 3.0;

    EXPECT_EQ(scaled.size(), 2);
    EXPECT_EQ(scaled[0], 3.0);
    EXPECT_EQ(scaled[1], 6.0);
}

TEST(Vector, ScalarMultiplicationLeft)
{
    Vector v{1.0, 2.0};

    Vector scaled = 3.0 * v;

    EXPECT_EQ(scaled.size(), 2);
    EXPECT_EQ(scaled[0], 3.0);
    EXPECT_EQ(scaled[1], 6.0);
}

TEST(Vector, ScalarDivision)
{
    Vector v{2.0, 4.0};

    Vector divided = v / 2.0;

    EXPECT_EQ(divided.size(), 2);
    EXPECT_EQ(divided[0], 1.0);
    EXPECT_EQ(divided[1], 2.0);
}

TEST(Vector, DotProduct)
{
    Vector v1{1.0, 2.0};
    Vector v2{3.0, 4.0};

    double dotProduct = dot(v1, v2);

    EXPECT_EQ(dotProduct, 11.0);
}

TEST(Vector, DotProductSizeMismatch)
{
    Vector v1{1.0};
    Vector v2{2.0, 3.0};
    EXPECT_THROW(dot(v1, v2), std::runtime_error);
}

TEST(Vector, Norm)
{
    Vector v{3.0, 4.0};
    double Norm = norm(v);
    EXPECT_DOUBLE_EQ(Norm, 5.0);
}

TEST(Vector, Sum)
{
    Vector v{1.0, 2.0, 3.0};
    double result = sum(v);
    EXPECT_DOUBLE_EQ(result, 6.0);
}

TEST(Vector, Concat)
{
    Vector v1{1.0, 2.0};
    Vector v2{3.0};

    Vector v3 = concat(v1, v2);

    Vector expected{1.0, 2.0, 3.0};
    EXPECT_EQ(v3, expected);
}

TEST(Vector, ToString)
{
    Vector v{1.0, 2.0, 3.0};
    std::string str = to_string(v);
    EXPECT_EQ(str, "[1.000000, 2.000000, 3.000000]");
}
