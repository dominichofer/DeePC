#include "controller.h"
#include "algorithm.h"
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

Controller::Controller(
    const std::vector<VectorXd> &u_d,
    const std::vector<VectorXd> &y_d,
    int T_ini,
    int target_size,
    std::variant<MatrixXd, double> Q_,
    std::variant<MatrixXd, double> R_,
    std::function<VectorXd(const VectorXd &)> input_constrain_fkt,
    int max_pgm_iterations,
    double pgm_tolerance)
    : T_ini(T_ini)
    , target_size(target_size)
    , u_ini(T_ini)
    , y_ini(T_ini)
    , input_constrain_fkt(input_constrain_fkt)
    , max_pgm_iterations(max_pgm_iterations)
    , pgm_tolerance(pgm_tolerance)
    , input_dims(u_d.front().size())
    , output_dims(y_d.front().size())
{
    if (T_ini <= 0)
        throw std::invalid_argument("T_ini must be greater than 0. T_ini=" + std::to_string(T_ini));

    const int offline_size = u_d.size();

    check_dimensions(u_d, "u_d", offline_size, input_dims);
    check_dimensions(y_d, "y_d", offline_size, output_dims);

    const int Q_size = target_size * output_dims;
    if (std::holds_alternative<double>(Q_))
        Q = std::get<double>(Q_) * MatrixXd::Identity(Q_size, Q_size);
    else
        Q = std::get<MatrixXd>(Q_);

    const int R_size = target_size * input_dims;
    if (std::holds_alternative<double>(R_))
        R = std::get<double>(R_) * MatrixXd::Identity(R_size, R_size);
    else
        R = std::get<MatrixXd>(R_);

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
    // x = [u_ini; y_ini]
    // to get
    // A * g = [x; u]  (1)
    // and
    // Y_f * g = y  (2).

    // We multiply (1) from the left with the left pseudo inverse of A.
    // Since pinv(A) * A = I, we get g = pinv(A) * [x; u].
    // Substituting g in (2) gives Y_f * pinv(A) * [x; u] = y.

    // We define
    auto M = Y_f * A.completeOrthogonalDecomposition().pseudoInverse();
    // and get M * [x; u] = y.

    // We define [M_x; M_u] := M
    // such that M_x * x + M_u * u = y.
    const int dim_sum = input_dims + output_dims;
    M_x = M.block(0, 0, M.rows(), T_ini * dim_sum);
    M_u = M.block(0, T_ini * dim_sum, M.rows(), M.cols() - T_ini * dim_sum);

    // We can now solve the unconstrained problem.
    // This is a ridge regression problem with generalized Tikhonov regularization.
    // https://en.wikipedia.org/wiki/Ridge_regression//Generalized_Tikhonov_regularization
    // minimize: ||y - target||_Q^2 + ||u||_R^2
    // subject to: M_u * u = y - M_x * x
    // This has an explicit solution u_star = (M_u^T * Q * M_u + R)^-1 * (M_u^T * Q * y).

    // We precompute the matrix G = M_u^T * Q * M_u + R.
    G = M_u.transpose() * Q * M_u + R;
}

Controller::Controller(
    const std::vector<VectorXd> &u_d,
    const std::vector<VectorXd> &y_d,
    int T_ini,
    int target_size,
    std::function<VectorXd(const VectorXd &)> input_constrain_fkt,
    int max_pgm_iterations,
    double pgm_tolerance)
    : Controller(u_d, y_d, T_ini, target_size, 1.0, 0.0, input_constrain_fkt, max_pgm_iterations, pgm_tolerance)
{
}

bool Controller::is_initialized() const
{
    return u_ini.size() == T_ini && y_ini.size() == T_ini;
}

void Controller::update(VectorXd u, VectorXd y)
{
    u_ini.push_back(std::move(u));
    y_ini.push_back(std::move(y));
}

void Controller::clear()
{
    u_ini.clear();
    y_ini.clear();
}

std::vector<VectorXd> Controller::apply(const std::vector<VectorXd> &target, const std::vector<VectorXd> &offset)
{
    if (!is_initialized())
        return {};

    check_dimensions(target, "target", target_size, output_dims);

    if (!offset.empty())
        check_dimensions(offset, "offset", target_size, output_dims);
    
    // Flatten
    VectorXd target_ = concat(target);
    VectorXd offset_ = offset.empty() ? VectorXd::Zero(target_size * output_dims) : concat(offset);

    auto x = concat(u_ini.get(), y_ini.get());
    auto w = M_u.transpose() * Q * (target_ - M_x * x) + R * offset_;
    auto u_star = G.ldlt().solve(w).eval();

    if (input_constrain_fkt != nullptr)
        u_star = projected_gradient_method(
            G,
            u_star,
            w,
            input_constrain_fkt,
            max_pgm_iterations,
            pgm_tolerance);

    return split(u_star, target_size);
}

std::vector<VectorXd> Controller::apply(const VectorXd &target, const VectorXd &offset)
{
    return apply(
        std::vector<VectorXd>{target},
        offset.size() == 0 ? std::vector<VectorXd>{} : std::vector<VectorXd>{offset}
    );
}
