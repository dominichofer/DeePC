#include "algorithm.h"
#include <cassert>
#include <vector>

std::string to_string(const std::vector<VectorXd>& v)
{
    std::string s = "[";
    for (const auto &x : v)
    {
        s += "[";
        for (int i = 0; i < x.size(); ++i)
        {
            s += std::to_string(x(i));
            if (i < x.size() - 1)
                s += ", ";
        }
        s += "]";
    }
    s += "]";
    return s;
}

VectorXd clamp(VectorXd value, double low, double high)
{
    for (int i = 0; i < value.size(); ++i)
        value(i) = std::clamp(value(i), low, high);
    return value;
}

VectorXd clamp(VectorXd value, const VectorXd& low, const VectorXd& high)
{
    assert(value.size() == low.size());
    assert(value.size() == high.size());
    for (int i = 0; i < value.size(); ++i)
        value(i) = std::clamp(value(i), low(i), high(i));
    return value;
}

VectorXd concat(const VectorXd& l, const VectorXd& r)
{
    VectorXd res(l.size() + r.size());
    res << l, r;
    return res;
}

VectorXd concat(const std::vector<VectorXd>& v)
{
    int size = v.front().size();
    VectorXd res(v.size() * size);
    for (std::size_t i = 0; i < v.size(); ++i)
        res.segment(i * size, size) = v[i];
    return res;
}

VectorXd concat(const std::vector<VectorXd>& l, const std::vector<VectorXd>& r)
{
    assert(l.size() == r.size());
    VectorXd res(l.size() * l.front().size() + r.size() * r.front().size());
    for (std::size_t i = 0; i < l.size(); ++i)
    {
        res.segment(i * l.front().size(), l.front().size()) = l[i];
        res.segment(i * r.front().size() + l.size() * l.front().size(), r.front().size()) = r[i];
    }
    return res;
}

std::vector<VectorXd> split(const VectorXd& vec, int size)
{
    assert(vec.size() % size == 0);
    std::vector<VectorXd> res(size);
    for (int i = 0; i < size; ++i)
        res[i] = vec.segment(i * vec.size() / size, vec.size() / size);
    return res;
}

MatrixXd vstack(const MatrixXd& upper, const MatrixXd& lower)
{
    assert(upper.cols() == lower.cols());

    MatrixXd res(upper.rows() + lower.rows(), upper.cols());
    res << upper, lower;
    return res;
}

#include <iostream>
#include <cstdint>  // for uintptr_t

MatrixXd vstack(const MatrixXd& upper,
                const MatrixXd& middle,
                const MatrixXd& lower)
{
    assert(upper.cols() == middle.cols() && upper.cols() == lower.cols());
    const int ru = upper.rows(),  cu = upper.cols();
    const int rm = middle.rows(), cm = middle.cols(); // == cu
    const int rl = lower.rows(),  cl = lower.cols(); // == cu
    const int rows = ru + rm + rl, cols = cu;

    std::cout 
      << "[DeePCpp::vstack] upper=" << ru  << "×" << cu
      << ", middle="    << rm  << "×" << cm
      << ", lower="     << rl  << "×" << cl
      << " → total="    << rows << "×" << cols << "\n";

    MatrixXd res(rows, cols);  // heap → guaranteed 32B‐aligned by Eigen
    auto ptr = reinterpret_cast<uintptr_t>(res.data());
    std::cout 
      << "[DeePCpp::vstack] res.data()=" << static_cast<void*>(res.data())
      << "  (mod32=" << (ptr % 32) << ")\n";

    // Safe, block‐by‐block copies:
    res.topRows   (ru)                 = upper;
    res.middleRows(ru, rm)             = middle;
    res.bottomRows(rl)                 = lower;

    std::cout 
      << "[DeePCpp::vstack] copy done: first=" << res(0,0)
      << " last=" << res(rows-1, cols-1) << "\n";
    return res;
}

MatrixXd HankelMatrix(int rows, const std::vector<VectorXd>& vec)
{
    int dims = vec.front().size();
    MatrixXd res(rows * dims, vec.size() - rows + 1);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < dims; ++j)
            for (std::size_t k = 0; k < vec.size() - rows + 1; ++k)
                res(i * dims + j, k) = vec[k + i](j);
    return res;
}

VectorXd projected_gradient_method(
    const MatrixXd& mat,
    const VectorXd& initial_guess,
    const VectorXd& target,
    std::function<VectorXd(const VectorXd&)> projection,
    int max_iterations,
    double tolerance)
{
    double step_size = 1 / mat.norm();
    VectorXd x_old = projection(initial_guess);
    for (int i = 0; i < max_iterations; ++i)
    {
        VectorXd x_new = projection(x_old - step_size * (mat * x_old - target));
        if ((x_new - x_old).norm() < tolerance)
            return x_new;
        x_old = x_new;
    }
    return x_old;
}
