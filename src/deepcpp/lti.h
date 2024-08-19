#pragma once
#include <Eigen/Dense>
#include <random>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Discrete Linear Time Invariant System
class DiscreteLTI
{
    MatrixXd A, B, C, D;
    VectorXd x;
public:
    DiscreteLTI() = default;
    DiscreteLTI(MatrixXd A, MatrixXd B, MatrixXd C, MatrixXd D, VectorXd x) noexcept
        : A(std::move(A)), B(std::move(B)), C(std::move(C)), D(std::move(D)), x(std::move(x)) {}

    int input_dim() const { return B.cols(); }
    int output_dim() const { return C.rows(); }
    void set_state(VectorXd x) { this->x = std::move(x); }
    bool is_controllable() const;
    bool is_observable() const;
    bool is_stable() const;

    // Apply input(s) and get output(s)
    VectorXd apply(double u);
    virtual VectorXd apply(const VectorXd& u);

    // Apply multiple inputs and get multiple outputs
    std::vector<VectorXd> apply_multiple(const std::vector<double>& u);
    virtual std::vector<VectorXd> apply_multiple(const std::vector<VectorXd>& u);
};

// Discrete Linear Time Invariant System with random noise
class RandomNoiseDiscreteLTI : public DiscreteLTI
{
    std::mt19937 gen;
    std::normal_distribution<double> dist;
public:
    RandomNoiseDiscreteLTI(MatrixXd A, MatrixXd B, MatrixXd C, MatrixXd D, VectorXd x, double noise_stddev)
        : DiscreteLTI(std::move(A), std::move(B), std::move(C), std::move(D), std::move(x)), gen(std::random_device()()), dist(0, noise_stddev) {}

    // Apply input(s) and get output(s)
    VectorXd apply(const VectorXd& u) override;
};