#include "algorithm.h"
#include <cassert>
#include <cmath>

DenseMatrix inv(const IMatrix& m)
{
    assert(m.rows() == m.cols());
    int n = m.rows();

    auto result = DenseMatrix::Zeros(n, n);
    for (int i = 0; i < n; ++i)
        result(i, i) = 1;
    
    // Gauss-Jordan elimination with full pivoting
    auto a = DenseMatrix(m);
    for (int i = 0; i < n; ++i)
    {
        double pivot = a(i, i);
        for (int j = 0; j < n; ++j)
        {
            a(i, j) /= pivot;
            result(i, j) /= pivot;
        }
        for (int j = 0; j < n; ++j)
        {
            if (j == i)
                continue;
            double factor = a(j, i);
            for (int k = 0; k < n; ++k)
            {
                a(j, k) -= factor * a(i, k);
                result(j, k) -= factor * result(i, k);
            }
        }
    }
}

double norm(const IMatrix& m)
{
    double sum = 0;
    for (int i = 0; i < m.rows(); ++i)
    {
        double sub_sum = 0; // Increases numerical accuracy
        for (int j = 0; j < m.cols(); ++j)
            sub_sum += m(i, j) * m(i, j);
        sum += sub_sum;
    }
    return std::sqrt(sum);
}

std::tuple<DenseMatrix, DenseMatrix, DenseMatrix> singular_value_decomposition(const IMatrix& m)
{
    int rows = m.rows();
    int cols = m.cols();
    auto mt = TransposedMatrix(m);
    auto mtm = mt * m;
    auto mmt = m * mt;
    auto mtm_eigen = inv(mtm);
    auto mmt_eigen = inv(mmt);
    auto u = m * mtm_eigen;
    auto v = mt * mmt_eigen;
    auto s = DenseMatrix::Zeros(rows, cols);
    for (int i = 0; i < std::min(rows, cols); ++i)
        s(i, i) = std::sqrt(mtm(i, i));
    return {u, s, v};
}

DenseMatrix pseudoinverse(const IMatrix& m)
{
    auto [u, s, v] = singular_value_decomposition(m);
    return v * inv(s) * u;
}

DenseMatrix left_pseudoinverse(const IMatrix& m)
{
    auto mt = TransposedMatrix(m);
    return inv(mt * m) * mt;
}

DenseMatrix right_pseudoinverse(const IMatrix& m)
{
    auto mt = TransposedMatrix(m);
    return mt * inv(m * mt);
}

std::vector<double> solve(const IMatrix& a, const std::vector<double>& b)
{
    return inv(a) * b;
}

std::vector<double> projected_gradient_method(
    const IMatrix& mat,
    const std::vector<double>& x_ini,
    const std::vector<double>& target,
    std::function<std::vector<double>(const std::vector<double>&)> constrain,
    int max_iterations,
    double tolerance)
{
    double step_size = 1 / norm(mat);
    std::vector<double> x_old = constrain(target);
    for (int i = 0; i < max_iterations; ++i)
    {
        std::vector<double> x_new = constrain(x_old - step_size * (mat * x_old - target));
        if (norm(x_new - x_old) < tolerance)
            return x_new;
        x_old = x_new;
    }
    return x_old;
}