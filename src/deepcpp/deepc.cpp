#include "deepc.h"
#include "algorithm.h"
#include <cassert>
#include <stdexcept>
#include <string>

static void check_dimensions(const std::vector<VectorXd> &var, std::string name, int size, int dims)
{
    if (var.size() != size)
        throw std::invalid_argument(name + ".size()=" + std::to_string(var.size()) + " but should be " + std::to_string(size) + ".");
    for (int i = 0; i < size; ++i)
        if (var[i].size() != dims)
            throw std::invalid_argument(name + "[" + std::to_string(i) + "].size()=" + std::to_string(var[i].size()) + " but should be " + std::to_string(dims) + ".");
}

std::vector<VectorXd> deePC(
    const std::vector<VectorXd> &u_d,
    const std::vector<VectorXd> &y_d,
    const std::vector<VectorXd> &u_ini,
    const std::vector<VectorXd> &y_ini,
    const std::vector<VectorXd> &target,
    const MatrixXd &Q,
    const MatrixXd &R,
    std::function<VectorXd(const VectorXd &)> control_constrain_fkt,
    int max_pgm_iterations,
    double pgm_tolerance)
{
    const int offline_size = u_d.size();
    const int T_ini = u_ini.size();
    const int target_size = target.size();
    const int input_dims = u_d.front().size();
    const int output_dims = y_d.front().size();

    check_dimensions(u_d, "u_d", offline_size, input_dims);
    check_dimensions(y_d, "y_d", offline_size, output_dims);
    check_dimensions(u_ini, "u_ini", T_ini, input_dims);
    check_dimensions(y_ini, "y_ini", T_ini, output_dims);
    check_dimensions(target, "target", target_size, output_dims);

    // Check Q
    if (Q.rows() != Q.cols())
        throw std::invalid_argument("Q must be a square matrix. Q.rows()=" + std::to_string(Q.rows()) + ", Q.cols()=" + std::to_string(Q.cols()));
    if (Q.rows() != target_size * output_dims)
        throw std::invalid_argument("Q.rows()=" + std::to_string(Q.rows()) + " but should be " + std::to_string(target_size * output_dims) + ".");

    // Check R
    if (R.rows() != R.cols())
        throw std::invalid_argument("R must be a square matrix. R.rows()=" + std::to_string(R.rows()) + ", R.cols()=" + std::to_string(R.cols()));
    if (R.rows() != target_size * input_dims)
        throw std::invalid_argument("R.rows()=" + std::to_string(R.rows()) + " but should be " + std::to_string(target_size * input_dims) + ".");

    auto U = HankelMatrix(T_ini + target_size, u_d);
    auto U_p = U.block(0, 0, T_ini * input_dims, U.cols()); // past
    auto U_f = U.block(T_ini * input_dims, 0, U.rows() - T_ini * input_dims, U.cols()); // future
    auto Y = HankelMatrix(T_ini + target_size, y_d);
    auto Y_p = Y.block(0, 0, T_ini * output_dims, Y.cols()); // past
    auto Y_f = Y.block(T_ini * output_dims, 0, Y.rows() - T_ini * output_dims, Y.cols()); // future

    // Now solving
    // minimize: ||y - target||_Q^2 + ||u||_R^2
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
    // minimize: ||y - target||_Q^2 + ||u||_R^2
    // subject to: M_u * u = y - M_x * x
    // This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y).

    // Flatten
    VectorXd target_ = concat(target);

    auto G = M_u_T * Q * M_u + R;
    auto w = M_u_T * Q * (target_ - M_x * x);
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

std::vector<VectorXd> deePC(
    const std::vector<VectorXd> &u_d,
    const std::vector<VectorXd> &y_d,
    const std::vector<VectorXd> &u_ini,
    const std::vector<VectorXd> &y_ini,
    const std::vector<VectorXd> &target_output,
    std::function<VectorXd(const VectorXd &)> control_constrain_fkt,
    int max_pgm_iterations,
    double pgm_tolerance)
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
