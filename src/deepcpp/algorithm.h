#pragma once
#include <Eigen/Dense>
#include <functional>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

std::string to_string(const std::vector<VectorXd>&);

VectorXd clamp(VectorXd, double min, double max);

VectorXd concat(const VectorXd&, const VectorXd&);
VectorXd concat(const std::vector<VectorXd>&);
VectorXd concat(const std::vector<VectorXd>&, const std::vector<VectorXd>&);

// Splits a vector into a vector of vectors of size `size`.
std::vector<VectorXd> split(const VectorXd&, int size);

MatrixXd vstack(const MatrixXd&, const MatrixXd&);
MatrixXd vstack(const MatrixXd&, const MatrixXd&, const MatrixXd&);

// Returns a generalized Hankel matrix with one row per dimension of the input vectors.
MatrixXd HankelMatrix(int rows, const std::vector<VectorXd>&);

VectorXd projected_gradient_method(
    const MatrixXd& mat,
    const VectorXd& initial_guess,
    const VectorXd& target,
    std::function<VectorXd(const VectorXd&)> projection,
    int max_iterations = 300,
    double tolerance = 1e-6);


std::vector<VectorXd> linear_chirp(double f0, double f1, int samples, int phi = 0);
