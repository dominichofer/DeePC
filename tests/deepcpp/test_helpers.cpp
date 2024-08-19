#include "gtest/gtest.h"
#include "helpers.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(Helpers, Matrix_1_row)
{
    auto m = Matrix({1, 2, 3});
    EXPECT_EQ(m.rows(), 1);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(0, 2), 3);
}

TEST(Helpers, Matrix_2_rows)
{
    auto m = Matrix({1, 2, 3}, {4, 5, 6});
    EXPECT_EQ(m.rows(), 2);
    EXPECT_EQ(m.cols(), 3);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 2);
    EXPECT_EQ(m(0, 2), 3);
    EXPECT_EQ(m(1, 0), 4);
    EXPECT_EQ(m(1, 1), 5);
    EXPECT_EQ(m(1, 2), 6);
}

TEST(Helpers, Matrix_3_rows)
{
    auto m = Matrix({1}, {2}, {3});
    EXPECT_EQ(m.rows(), 3);
    EXPECT_EQ(m.cols(), 1);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 0), 2);
    EXPECT_EQ(m(2, 0), 3);
}

TEST(Helpers, Matrix_4_rows)
{
    auto m = Matrix({1}, {2}, {3}, {4});
    EXPECT_EQ(m.rows(), 4);
    EXPECT_EQ(m.cols(), 1);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 0), 2);
    EXPECT_EQ(m(2, 0), 3);
    EXPECT_EQ(m(3, 0), 4);
}

TEST(Helpers, Matrix_5_rows)
{
    auto m = Matrix({1}, {2}, {3}, {4}, {5});
    EXPECT_EQ(m.rows(), 5);
    EXPECT_EQ(m.cols(), 1);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 0), 2);
    EXPECT_EQ(m(2, 0), 3);
    EXPECT_EQ(m(3, 0), 4);
    EXPECT_EQ(m(4, 0), 5);
}

TEST(Helpers, Matrix_6_rows)
{
    auto m = Matrix({1}, {2}, {3}, {4}, {5}, {6});
    EXPECT_EQ(m.rows(), 6);
    EXPECT_EQ(m.cols(), 1);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 0), 2);
    EXPECT_EQ(m(2, 0), 3);
    EXPECT_EQ(m(3, 0), 4);
    EXPECT_EQ(m(4, 0), 5);
    EXPECT_EQ(m(5, 0), 6);
}

TEST(Helpers, Vector)
{
    auto v = Vector(1, 2, 3);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v(0), 1);
    EXPECT_EQ(v(1), 2);
    EXPECT_EQ(v(2), 3);
}

TEST(Helpers, Vectors_1)
{
    auto v = Vectors({1});
    EXPECT_EQ(v.size(), 1);
    EXPECT_EQ(v[0], Vector(1));
}

TEST(Helpers, Vectors_2)
{
    auto v = Vectors({1}, {2});
    EXPECT_EQ(v.size(), 2);
    EXPECT_EQ(v[0], Vector(1));
    EXPECT_EQ(v[1], Vector(2));
}

TEST(Helpers, Vectors_3)
{
    auto v = Vectors({1}, {2}, {3});
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v[0], Vector(1));
    EXPECT_EQ(v[1], Vector(2));
    EXPECT_EQ(v[2], Vector(3));
}
