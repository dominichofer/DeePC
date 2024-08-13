#include "gtest/gtest.h"
#include "algorithm.h"
#include <vector>

TEST(IMatrixTest, Norm)
{
	std::vector<double> data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
	DenseMatrix m(data, 2);

	double n = norm(m);
	EXPECT_DOUBLE_EQ(n, norm(data));
}
