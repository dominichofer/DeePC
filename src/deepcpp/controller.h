#pragma once
#include "finite_queue.h"
#include <Eigen/Dense>
#include <functional>
#include <variant>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Controller
{
    int T_ini;
    int target_size;
    FiniteQueue u_ini, y_ini;
    MatrixXd Q, R;
    std::function<VectorXd(const VectorXd &)> input_constrain_fkt;
    int max_pgm_iterations;
    double pgm_tolerance;
    int input_dims;
    int output_dims;
    MatrixXd M_x, M_u;

public:
    MatrixXd G;
    // Optimal controller for a given system and target system outputs.
    // According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    // https://arxiv.org/abs/1811.05890
    // Args:
    //     u_d: Control inputs from an offline procedure.
    //     y_d: System outputs from an offline procedure.
    //     T_ini: Number of system in- and outputs to initialize the state.
    //     target_len: Length of the target system outputs, optimal control tries to match.
    //     Q: Output cost matrix. Defaults to identity matrix.
    //         If double, diagonal matrix with this value.
    //     R: Control cost matrix. Defaults to zero matrix.
    //         If double, diagonal matrix with this value.
    //     input_constrain_fkt: Function that constrains the control inputs.
    //     max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
    //                         used to solve the constrained optimization problem.
    //     pgm_tolerance: Tolerance for the PGM algorithm.
    Controller(
        const std::vector<VectorXd> &u_d,
        const std::vector<VectorXd> &y_d,
        int T_ini,
        int target_size,
        std::variant<MatrixXd, double> Q,
        std::variant<MatrixXd, double> R,
        std::function<VectorXd(const VectorXd &)> input_constrain_fkt = nullptr,
        int max_pgm_iterations = 300,
        double pgm_tolerance = 1e-6);

    // Optimal controller for a given system and target system outputs.
    // According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
    // https://arxiv.org/abs/1811.05890
    // Args:
    //     u_d: Control inputs from an offline procedure.
    //     y_d: System outputs from an offline procedure.
    //     T_ini: Number of system in- and outputs to initialize the state.
    //     target_len: Length of the target system outputs, optimal control tries to match.
    //     input_constrain_fkt: Function that constrains the control inputs.
    //     max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
    //                         used to solve the constrained optimization problem.
    //     pgm_tolerance: Tolerance for the PGM algorithm.
    Controller(
        const std::vector<VectorXd> &u_d,
        const std::vector<VectorXd> &y_d,
        int T_ini,
        int target_size,
        std::function<VectorXd(const VectorXd &)> input_constrain_fkt = nullptr,
        int max_pgm_iterations = 300,
        double pgm_tolerance = 1e-6);

    bool is_initialized() const;
    void update(VectorXd u, VectorXd y);
    void clear();
    std::vector<VectorXd> apply(const std::vector<VectorXd> &target, const std::vector<VectorXd> &offset = {});
    std::vector<VectorXd> apply(const VectorXd &target, const VectorXd &offset = {});
};
