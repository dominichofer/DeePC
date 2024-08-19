#pragma once
#include <Eigen/Dense>
#include <functional>

using Eigen::MatrixXd;
using Eigen::VectorXd;

VectorXd concat(const VectorXd&, const VectorXd&);
MatrixXd vstack(const MatrixXd&, const MatrixXd&);
MatrixXd vstack(const MatrixXd&, const MatrixXd&, const MatrixXd&);

MatrixXd HankelMatrix(int rows, const VectorXd&);

VectorXd projected_gradient_method(
    const MatrixXd& mat,
    const VectorXd& initial_guess,
    const VectorXd& target,
    std::function<VectorXd(const VectorXd&)> projection,
    int max_iterations = 300,
    double tolerance = 1e-6);


std::vector<VectorXd> linear_chirp(double f0, double f1, int samples, int phi = 0);
