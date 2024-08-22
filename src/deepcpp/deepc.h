#pragma once
#include <Eigen/Dense>
#include "algorithm.h"
#include <cassert>
#include <deque>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

void check_dimensions(const std::vector<VectorXd>& var, std::string name, int size, int dims)
{
    if (var.size() != size)
        throw std::invalid_argument(name + ".size()=" + std::to_string(var.size()) + " but should be " + std::to_string(size) + ".");
    for (int i = 0; i < size; ++i)
        if (var[i].size() != dims)
            throw std::invalid_argument(name + "[" + std::to_string(i) + "].size()=" + std::to_string(var[i].size()) + " but should be " + std::to_string(dims) + ".");
}

// Returns the optimal control for a given system and reference trajectory.
// According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
// https://arxiv.org/abs/1811.05890
// Args:
//     u_d: Control inputs from an offline procedure.
//     y_d: Outputs from an offline procedure.
//     u_ini: Control inputs to initiate the state.
//     y_ini: Outputs to initiate the state.
//     r: Reference trajectory.
//     Q: Output cost matrix, defaults to identity matrix.
//     R: Control cost matrix, defaults to zero matrix.
//     control_constrain_fkt: Function that constrains the control.
//     max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
//                         used to solve the constrained optimization problem.
//     pgm_tolerance: Tolerance for the PGM algorithm.
inline std::vector<VectorXd> deePC(
    const std::vector<VectorXd>& u_d,
    const std::vector<VectorXd>& y_d,
    const std::vector<VectorXd>& u_ini,
    const std::vector<VectorXd>& y_ini,
    const std::vector<VectorXd>& r,
    const MatrixXd &Q,
    const MatrixXd &R,
    std::function<VectorXd(const VectorXd&)> control_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6)
{
    const int offline_size = u_d.size();
    const int T_ini = u_ini.size();
    const int target_size = r.size();
    const int input_dims = u_d.front().size();
    const int output_dims = y_d.front().size();

    check_dimensions(u_d, "u_d", offline_size, input_dims);
    check_dimensions(y_d, "y_d", offline_size, output_dims);
    check_dimensions(u_ini, "u_ini", T_ini, input_dims);
    check_dimensions(y_ini, "y_ini", T_ini, output_dims);
    check_dimensions(r, "r", target_size, output_dims);

    // Check Q
    assert(Q.rows() == Q.cols());
    assert(Q.rows() == target_size * output_dims);

    // Check R
    assert(R.rows() == R.cols());
    assert(R.rows() == target_size * input_dims);

    auto U = HankelMatrix(T_ini + target_size, u_d);
    auto U_p = U.block(0, 0, T_ini * input_dims, U.cols()); // past
    auto U_f = U.block(T_ini * input_dims, 0, U.rows() - T_ini * input_dims, U.cols()); // future
    auto Y = HankelMatrix(T_ini + target_size, y_d);
    auto Y_p = Y.block(0, 0, T_ini * output_dims, Y.cols()); // past
    auto Y_f = Y.block(T_ini * output_dims, 0, Y.rows() - T_ini * output_dims, Y.cols()); // future

    // Now solving
    // minimize: ||y - r||_Q^2 + ||u||_R^2
    // subject to: [U_p; Y_p; U_f; Y_f] * g = [u_ini; y_ini; u; y]

    // We define
    auto A = vstack(U_p, Y_p, U_f);
    auto x = concat(u_ini, y_ini);
    // to get
    // A * g = [x; u]  (1)
    // and
    // Y_f * g = y  (2).

    // We multiply (1) from the left with the pseudo inverse of A.
    // Since pinv(A) * A = I, we get g = pinv(A) * [x; u].
    // Substituting g in (2) gives Y_f * pinv(A) * [x; u] = y.

    // We define
    auto M = Y_f * A.completeOrthogonalDecomposition().pseudoInverse();
    // and get M * [x; u] = y.

    // We define [M_x; M_u] := M
    // such that M_x * x + M_u * u = y.
    auto M_x = M.block(0, 0, M.rows(), x.size());
    auto M_u = M.block(0, x.size(), M.rows(), M.cols() - x.size());
    auto M_u_T = M_u.transpose();

    // We can now solve the unconstrained problem.
    // This is a ridge regression problem with generalized Tikhonov regularization.
    // https://en.wikipedia.org/wiki/Ridge_regression//Generalized_Tikhonov_regularization
    // minimize: ||y - r||_Q^2 + ||u||_R^2
    // subject to: M_u * u = y - M_x * x
    // This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y).
    
    // Flatten r
    VectorXd r_ = concat(r);

    auto G = M_u_T * Q * M_u + R;
    auto w = M_u_T * Q * (r_ - M_x * x);
    auto u_star = G.ldlt().solve(w).eval();

    if (control_constrain_fkt != nullptr)
        u_star = projected_gradient_method(
            G,
            u_star,
            w,
            control_constrain_fkt,
            max_pgm_iterations,
            pgm_tolerance);

    return split(u_star, target_size);
}

inline std::vector<VectorXd> deePC(
    const std::vector<VectorXd>& u_d,
    const std::vector<VectorXd>& y_d,
    const std::vector<VectorXd>& u_ini,
    const std::vector<VectorXd>& y_ini,
    const std::vector<VectorXd>& target_output,
    std::function<VectorXd(const VectorXd&)> control_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6)
{
    const int target_size = target_output.size();
    const int input_dims = u_d.front().size();
    const int output_dims = y_d.front().size();

    const int Q_size = target_size * output_dims;
    auto Q = MatrixXd::Identity(Q_size, Q_size);

    const int R_size = target_size * input_dims;
    auto R = MatrixXd::Zero(R_size, R_size);

    return deePC(u_d, y_d, u_ini, y_ini, target_output, Q, R, control_constrain_fkt, max_pgm_iterations, pgm_tolerance);
}

class FiniteQueue
{
    std::deque<VectorXd> data;
    int max_size;

public:
    FiniteQueue(int max_size) : max_size(max_size) {}

    void push_back(VectorXd value)
    {
        data.push_back(std::move(value));
        if (data.size() > max_size)
            data.pop_front();
    }
    std::vector<VectorXd> get() const { return std::vector<VectorXd>(data.begin(), data.end()); }
    int size() const { return data.size(); }
    void clear() { data.clear(); }
};

// Controller
class DeePC
{
    int T_ini;
    int r_size;
    FiniteQueue u_ini, y_ini;
    MatrixXd M, G;
    const MatrixXd &Q;
    const MatrixXd &R;
    std::function<VectorXd(const VectorXd &)> control_constrain_fkt;
    int max_pgm_iterations;
    double pgm_tolerance;

public:
    DeePC(
        const std::vector<VectorXd> &u_d,
        const std::vector<VectorXd> &y_d,
        int T_ini,
        int r_size,
        const MatrixXd &Q,
        const MatrixXd &R,
        std::function<VectorXd(const VectorXd &)> control_constrain_fkt = nullptr,
        int max_pgm_iterations = 300,
        double pgm_tolerance = 1e-6)
        : T_ini(T_ini), r_size(r_size), u_ini(T_ini), y_ini(T_ini), Q(Q), R(R), control_constrain_fkt(control_constrain_fkt), max_pgm_iterations(max_pgm_iterations), pgm_tolerance(pgm_tolerance)
    {
        assert(u_d.size() == y_d.size());

        auto U = HankelMatrix(T_ini + r_size, u_d);
        auto U_p = U.block(0, 0, T_ini, U.cols());                // past
        auto U_f = U.block(T_ini, 0, U.rows() - T_ini, U.cols()); // future
        auto Y = HankelMatrix(T_ini + r_size, y_d);
        auto Y_p = Y.block(0, 0, T_ini, Y.cols());                // past
        auto Y_f = Y.block(T_ini, 0, Y.rows() - T_ini, Y.cols()); // future

        // Now solving
        // minimize: ||y - r||_Q^2 + ||u||_R^2
        // subject to: [U_p; Y_p; U_f; Y_f] * g = [u_ini; y_ini; u; y]

        // We define
        auto A = vstack(U_p, Y_p, U_f);
        // x = [u_ini; y_ini]
        // to get
        // A * g = [x; u]  (1)
        // and
        // Y_f * g = y  (2).

        // We multiply (1) from the left with the left pseudo inverse of A.
        // Since pinv(A) * A = I, we get g = pinv(A) * [x; u].
        // Substituting g in (2) gives Y_f * pinv(A) * [x; u] = y.

        // We define
        M = Y_f * A.completeOrthogonalDecomposition().pseudoInverse();
        // and get M * [x; u] = y.

        // We define [M_x; M_u] := M
        // such that M_x * x + M_u * u = y.
        auto M_u = M.block(0, T_ini * 2, M.rows(), M.cols() - T_ini * 2);

        // We can now solve the unconstrained problem.
        // This is a ridge regression problem with generalized Tikhonov regularization.
        // https://en.wikipedia.org/wiki/Ridge_regression//Generalized_Tikhonov_regularization
        // minimize: ||y - r||_Q^2 + ||u||_R^2
        // subject to: M_u * u = y - M_x * x
        // This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y).

        // We precompute the matrix G = M_u^T * Q * M_u + R.
        G = M_u.transpose() * Q * M_u + R;
    }

    bool is_initialized() const
    {
        return u_ini.size() == T_ini && y_ini.size() == T_ini;
    }

    void append(VectorXd u, VectorXd y)
    {
        u_ini.push_back(std::move(u));
        y_ini.push_back(std::move(y));
    }

    void clear()
    {
        u_ini.clear();
        y_ini.clear();
    }

    VectorXd control(const VectorXd &r)
    {
        assert(is_initialized());
        assert(r.size() == r_size);

        // We define [M_x; M_u] := M
        // such that M_x * x + M_u * u = y.
        auto M_x = M.block(0, 0, M.rows(), T_ini * 2);
        auto M_u = M.block(0, T_ini * 2, M.rows(), M.cols() - T_ini * 2);

        auto x = concat(u_ini.get(), y_ini.get());
        auto w = M_u.transpose() * Q * (r - M_x * x);
        auto u_star = G.ldlt().solve(w);

        if (control_constrain_fkt == nullptr)
            return u_star;
        else
            return projected_gradient_method(
                G,
                u_star,
                w,
                control_constrain_fkt,
                max_pgm_iterations,
                pgm_tolerance);
    }
};
