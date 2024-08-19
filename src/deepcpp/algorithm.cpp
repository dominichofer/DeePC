#include "algorithm.h"
#include <cassert>

VectorXd concat(const VectorXd& l, const VectorXd& r)
{
    VectorXd res(l.size() + r.size());
    res << l, r;
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

MatrixXd HankelMatrix(int rows, const VectorXd& vec)
{
    MatrixXd res(rows, vec.size() - rows + 1);
    for (int i = 0; i < rows; ++i)
        res.row(i) = vec.segment(i, vec.size() - rows + 1);
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

std::vector<VectorXd> linear_chirp(double f0, double f1, int samples, int phi)
{
    double Pi = 3.14159265358979323846;
    std::vector<VectorXd> res;
    res.reserve(samples);
    for (int i = 0; i < samples; ++i)
    {
        double t = 1 / (samples - 1);
        double phase = f0 * t + 0.5 * (f1 - f0) * t * t;
        res.push_back(VectorXd::Constant(1, std::sin(phi + 2 * Pi * phase)));
    }
    return res;
}
