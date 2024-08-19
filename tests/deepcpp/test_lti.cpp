#include "gtest/gtest.h"
#include "lti.h"
#include "helpers.h"
#include <Eigen/Dense>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

TEST(DiscreteLTI, Dimensions1D)
{
    DiscreteLTI system(
        Matrix({2, 0}, {0, 3}),
        Matrix({1}, {0}),
        Matrix({1, 0}),
        Matrix({0}),
        Vector(0, 0));
    EXPECT_EQ(system.input_dim(), 1);
    EXPECT_EQ(system.output_dim(), 1);
}

TEST(DiscreteLTI, Dimensions2D)
{
    DiscreteLTI system(
        Matrix({2, 0}, {0, 3}),
        Matrix({1, 0}, {0, 1}),
        Matrix({1, 0}, {0, 1}),
        Matrix({0, 0}, {0, 0}),
        Vector(0, 0));
    EXPECT_EQ(system.input_dim(), 2);
    EXPECT_EQ(system.output_dim(), 2);
}

TEST(DiscreteLTI, Controllable)
{
    DiscreteLTI system(
        Matrix({2, 0}, {0, 3}),
        Matrix({1}, {1}),
        Matrix({1, 0}),
        Matrix({0}),
        Vector(0, 0));
    EXPECT_TRUE(system.is_controllable());
}

TEST(DiscreteLTI, Uncontrollable)
{
    DiscreteLTI system(
        Matrix({2, 0}, {0, 3}),
        Matrix({0}, {0}),
        Matrix({1, 0}),
        Matrix({0}),
        Vector(0, 0));
    EXPECT_FALSE(system.is_controllable());
}

TEST(DiscreteLTI, Observable)
{
    DiscreteLTI system(
        Matrix({1, 1}, {0, 1}),
        Matrix({1}, {0}),
        Matrix({1, 0}),
        Matrix({0}),
        Vector(0, 0));
    EXPECT_TRUE(system.is_observable());
}

TEST(DiscreteLTI, Unobservable)
{
    DiscreteLTI system(
        Matrix({2, 0}, {0, 3}),
        Matrix({1}, {0}),
        Matrix({1, 0}),
        Matrix({0}),
        Vector(0, 0));
    EXPECT_FALSE(system.is_observable());
}

TEST(DiscreteLTI, Stable)
{
    DiscreteLTI system(
        Matrix({0.5, 0}, {0.6, 0}),
        Matrix({1}, {1}),
        Matrix({1, 0}),
        Matrix({0}),
        Vector(0, 0));
    EXPECT_TRUE(system.is_stable());
}

TEST(DiscreteLTI, Unstable)
{
    DiscreteLTI system(
        Matrix({2, 0}, {0, 3}),
        Matrix({1}, {1}),
        Matrix({1, 0}),
        Matrix({0}),
        Vector(0, 0));
    EXPECT_FALSE(system.is_stable());
}

TEST(DiscreteLTI, Apply1D)
{
    DiscreteLTI system(
        Matrix({1, 1}, {1, 1}),
        Matrix({1}, {1}),
        Matrix({1, 1}),
        Matrix({1}),
        Vector(1, 1));

    VectorXd y = system.apply(1);

    EXPECT_EQ(y, Vector(7));
}

TEST(DiscreteLTI, Apply2D)
{
    DiscreteLTI system(
        Matrix({1, 1}, {1, 1}),
        Matrix({1, 1}, {1, 1}),
        Matrix({1, 1}),
        Matrix({1, 1}),
        Vector(1, 1));

    VectorXd y = system.apply(Vector(1, 1));

    EXPECT_EQ(y, Vector(10));
}

TEST(DiscreteLTI, ApplyMultiple1D)
{
    DiscreteLTI system(
        Matrix({1, 1}, {1, 1}),
        Matrix({1}, {1}),
        Matrix({1, 1}),
        Matrix({1}),
        Vector(1, 1));

    std::vector<VectorXd> y = system.apply_multiple(std::vector{1.0, 2.0, 3.0});

    std::vector<VectorXd> expected{Vector(7), Vector(18), Vector(41)};
    EXPECT_EQ(y, expected);
}

TEST(DiscreteLTI, ApplyMultiple2D)
{
    DiscreteLTI system(
        Matrix({1, 1}, {1, 1}),
        Matrix({1, 1}, {1, 1}),
        Matrix({1, 1}),
        Matrix({1, 1}),
        Vector(1, 1));

    std::vector<VectorXd> y = system.apply_multiple(std::vector{Vector(1, 1), Vector(2, 2)});

    std::vector<VectorXd> expected{Vector(10), Vector(28)};
    EXPECT_EQ(y, expected);
}
