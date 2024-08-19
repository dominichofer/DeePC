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


TEST(Helpers, Vector)
{
    auto v = Vector(1, 2, 3);
    EXPECT_EQ(v.size(), 3);
    EXPECT_EQ(v(0), 1);
    EXPECT_EQ(v(1), 2);
    EXPECT_EQ(v(2), 3);
}
