#include <gtest/gtest.h>
#include "math.h"

TEST(MathTest, VectorAddition)
{
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    std::vector<double> expected = {5.0, 7.0, 9.0};
    std::vector<double> result = a + b;
    ASSERT_EQ(result, expected);
}

TEST(MathTest, VectorSubtraction)
{
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    std::vector<double> expected = {-3.0, -3.0, -3.0};
    std::vector<double> result = a - b;
    ASSERT_EQ(result, expected);
}

TEST(MathTest, ScalarVectorMultiplication)
{
    double scalar = 2.0;
    std::vector<double> vector = {1.0, 2.0, 3.0};
    std::vector<double> expected = {2.0, 4.0, 6.0};
    std::vector<double> result = scalar * vector;
    ASSERT_EQ(result, expected);
}

TEST(MathTest, VectorConcatenation)
{
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    std::vector<double> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    std::vector<double> result = concat(a, b);
    ASSERT_EQ(result, expected);
}

TEST(MathTest, VectorNorm)
{
    std::vector<double> vector = {3.0, 4.0};
    double expected = 5.0;
    double result = norm(vector);
    ASSERT_EQ(result, expected);
}

// Add more test cases as needed

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}