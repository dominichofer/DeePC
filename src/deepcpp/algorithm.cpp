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

VectorXd clamp(VectorXd v, double min, double max)
{
    for (int i = 0; i < v.size(); ++i)
        v(i) = std::clamp(v(i), min, max);
    return v;
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

MatrixXd vstack(const MatrixXd& upper, const MatrixXd& middle, const MatrixXd& lower)
{
    assert(upper.cols() == middle.cols());
    assert(upper.cols() == lower.cols());

    MatrixXd res(upper.rows() + middle.rows() + lower.rows(), upper.cols());
    res << upper, middle, lower;
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
    VectorXd x_old = projection(target);
    for (int i = 0; i < max_iterations; ++i)
    {
        VectorXd x_new = projection(x_old - step_size * (mat * x_old - target));
        if ((x_new - x_old).norm() < tolerance)
            return x_new;
        x_old = x_new;
    }
    return x_old;
}

std::vector<double> linear_chirp(double f0, double f1, int samples, int phi)
{
    double Pi = 3.14159265358979323846;
    std::vector<double> res;
    res.reserve(samples);
    for (int i = 0; i < samples; ++i)
    {
        double t = i / (samples - 1.0);
        res.push_back(std::sin(phi + 2 * Pi * (f0 * t + 0.5 * (f1 - f0) * t * t)));
    }
    return res;
}
