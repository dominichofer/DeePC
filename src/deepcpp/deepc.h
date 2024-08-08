#include "algorithm.h"
#include "matrix.h"
#include "vector.h"
#include <cassert>
#include <deque>
#include <functional>
#include <vector>

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
std::vector<double> deePC(
    const std::vector<double>& u_d,
    const std::vector<double>& y_d,
    const std::vector<double>& u_ini,
    const std::vector<double>& y_ini,
    const std::vector<double>& r,
    const IMatrix& Q,
    const IMatrix& R,
    std::function<std::vector<double>(const std::vector<double>&)> control_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6)
{
    assert(u_d.size() == y_d.size(), "u_d and y_d must have the same size.");
    assert(u_ini.size() == y_ini.size(), "u_ini and y_ini must have the same size.");
    int T_ini = u_ini.size();

    auto U = HankelMatrix(u_d, T_ini + r.size());
    auto U_p = SubMatrix(U, 0, T_ini, 0, U.cols()); // past
    auto U_f = SubMatrix(U, T_ini, U.rows(), 0, U.cols()); // future
    auto Y = HankelMatrix(y_d, T_ini + r.size());
    auto Y_p = SubMatrix(Y, 0, T_ini, 0, Y.cols()); // past
    auto Y_f = SubMatrix(Y, T_ini, Y.rows(), 0, Y.cols()); // future

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

    // We multiply (1) from the left with the left pseudo inverse of A.
    // Since pinv(A) * A = I, we get g = pinv(A) * [x; u].
    // Substituting g in (2) gives Y_f * pinv(A) * [x; u] = y.

    // We define
    auto B = Y_f * left_pseudoinverse(A);
    // and get B * [x; u] = y.

    // We define (B_x, B_u) := B such that B_x * x + B_u * u = y.
    auto B_x = SubMatrix(B, 0, B.rows(), 0, 2 * T_ini);
    auto B_u = SubMatrix(B, 0, B.rows(), 2 * T_ini, B.cols());

    // We can now solve the unconstrained problem.
    // This is a ridge regression problem with generalized Tikhonov regularization.
    // https://en.wikipedia.org/wiki/Ridge_regression//Generalized_Tikhonov_regularization
    // minimize: ||y - r||_Q^2 + ||u||_R^2
    // subject to: B_u * u = y - B_x * x

    auto G = transposed(B_u) * Q * B_u + R;
    auto u_star = solve(G, transposed(B_u) * Q * (r - B_x * x));

    if (control_constrain_fkt == nullptr)
        return u_star;
    else
        return projected_gradient_method(
            G,
            u_star,
            control_constrain_fkt,
            max_pgm_iterations,
            pgm_tolerance);
}

std::vector<double> deePC(
    const std::vector<double>& u_d,
    const std::vector<double>& y_d,
    const std::vector<double>& u_ini,
    const std::vector<double>& y_ini,
    const std::vector<double>& r,
    std::function<std::vector<double>(const std::vector<double>&)> control_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6)
{
    auto Q = IdentityMatrix(y_ini.size());
    auto R = ZeroMatrix(u_ini.size());
    return deePC(u_d, y_d, u_ini, y_ini, r, Q, R, control_constrain_fkt, max_pgm_iterations, pgm_tolerance);
}


class MaxSizeQueue
{
    std::deque<double> data;
    int max_size;
public:
    MaxSizeQueue(int max_size) : max_size(max_size) {}

    void push_back(double value)
    {
        data.push_back(value);
        if (data.size() > max_size)
            data.pop_front();
    }
    std::vector<double> get() const { return std::vector<double>(data.begin(), data.end()); }
    int size() const { return data.size(); }
    void clear() { data.clear(); }
};


// Controller
class DeePC
{
    int T_ini;
    int r_size;
    MaxSizeQueue u_ini, y_ini;
    DenseMatrix B, G;
    const IMatrix& Q;
    const IMatrix& R;
    std::function<std::vector<double>(const std::vector<double>&)> control_constrain_fkt;
    int max_pgm_iterations;
    double pgm_tolerance;
public:
    DeePC(
        const std::vector<double>& u_d,
        const std::vector<double>& y_d,
        int T_ini,
        int r_size,
        const IMatrix& Q,
        const IMatrix& R,
        std::function<std::vector<double>(const std::vector<double>&)> control_constrain_fkt = nullptr,
        int max_pgm_iterations = 300,
        double pgm_tolerance = 1e-6)
        : T_ini(T_ini)
        , r_size(r_size)
        , u_ini(T_ini)
        , y_ini(T_ini)
        , Q(Q)
        , R(R)
        , control_constrain_fkt(control_constrain_fkt)
        , max_pgm_iterations(max_pgm_iterations)
        , pgm_tolerance(pgm_tolerance)
    {
        assert(u_d.size() == y_d.size(), "u_d and y_d must have the same size.");

        auto U = HankelMatrix(u_d, T_ini + r_size);
        auto U_p = SubMatrix(U, 0, T_ini, 0, U.cols()); // past
        auto U_f = SubMatrix(U, T_ini, U.rows(), 0, U.cols()); // future
        auto Y = HankelMatrix(y_d, T_ini + r_size);
        auto Y_p = SubMatrix(Y, 0, T_ini, 0, Y.cols()); // past
        auto Y_f = SubMatrix(Y, T_ini, Y.rows(), 0, Y.cols()); // future

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
        auto B = Y_f * left_pseudoinverse(A);
        // and get B * [x; u] = y.

        // We define (B_x, B_u) := B such that B_x * x + B_u * u = y.
        auto B_x = SubMatrix(B, 0, B.rows(), 0, 2 * T_ini);
        auto B_u = SubMatrix(B, 0, B.rows(), 2 * T_ini, B.cols());

        // We can now solve the unconstrained problem.
        // This is a ridge regression problem with generalized Tikhonov regularization.
        // https://en.wikipedia.org/wiki/Ridge_regression//Generalized_Tikhonov_regularization
        // minimize: ||y - r||_Q^2 + ||u||_R^2
        // subject to: B_u * u = y - B_x * x
        // This has an explicit solution u_star = (B_u^T * Q * B_u + R)^-1 * (B_u^T * Q * y).

        // We precompute G = B_u^T * Q * B_u + R.
        G = transposed(B_u) * Q * B_u + R;
    }

    bool is_initialized() const
    {
        return u_ini.size() == T_ini && y_ini.size() == T_ini;
    }

    void append(double u, double y)
    {
        u_ini.push_back(u);
        y_ini.push_back(y);
    }

    void clear()
    {
        u_ini.clear();
        y_ini.clear();
    }

    std::vector<double> control(const std::vector<double>& r)
    {
        assert(is_initialized(), "Internal state is not initialized.");
        assert(r.size() == r_size, "Reference trajectory has the wrong length.");

        // We define (B_x, B_u) := B such that B_x * x + B_u * u = y.
        auto B_x = SubMatrix(B, 0, B.rows(), 0, 2 * T_ini);
        auto B_u = SubMatrix(B, 0, B.rows(), 2 * T_ini, B.cols());

        auto x = concat(u_ini.get(), y_ini.get());
        auto u_star = solve(G, transposed(B_u) * Q * (r - B_x * x));
        if (control_constrain_fkt == nullptr)
            return u_star;
        else
            return projected_gradient_method(
                G,
                u_star,
                control_constrain_fkt,
                max_pgm_iterations,
                pgm_tolerance);
    }
};
