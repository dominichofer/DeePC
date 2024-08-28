#pragma once
#include <Eigen/Dense>
#include <functional>
#include <variant>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Returns the optimal control for a given system and target system outputs.
// According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
// https://arxiv.org/abs/1811.05890
// Args:
//     u_d: Control inputs from an offline procedure.
//     y_d: System outputs from an offline procedure.
//     u_ini: Control inputs to initialize the state.
//     y_ini: System outputs to initialize the state.
//     target: Target system outputs, optimal control tries to match.
//     Q: Output cost matrix. Defaults to identity matrix.
//        If double, diagonal matrix with this value.
//     R: Control cost matrix. Defaults to zero matrix.
//        If double, diagonal matrix with this value.
//     input_constrain_fkt: Function that constrains the system inputs.
//     max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
//                         used to solve the constrained optimization problem.
//     pgm_tolerance: Tolerance for the PGM algorithm.
std::vector<VectorXd> deePC(
    const std::vector<VectorXd>& u_d,
    const std::vector<VectorXd>& y_d,
    const std::vector<VectorXd>& u_ini,
    const std::vector<VectorXd>& y_ini,
    const std::vector<VectorXd>& target,
    std::variant<MatrixXd, double> Q,
    std::variant<MatrixXd, double> R,
    std::function<VectorXd(const VectorXd&)> input_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6);

// Returns the optimal control for a given system and target system outputs.
// According to the paper Data-Enabled Predictive Control: In the Shallows of the DeePC
// https://arxiv.org/abs/1811.05890
// Args:
//     u_d: Control inputs from an offline procedure.
//     y_d: System outputs from an offline procedure.
//     u_ini: Control inputs to initialize the state.
//     y_ini: System outputs to initialize the state.
//     target: Target system outputs, optimal control tries to match.
//     input_constrain_fkt: Function that constrains the system inputs.
//     max_pgm_iterations: Maximum number of iterations of the projected gradient method (PGM)
//                         used to solve the constrained optimization problem.
//     pgm_tolerance: Tolerance for the PGM algorithm.
std::vector<VectorXd> deePC(
    const std::vector<VectorXd>& u_d,
    const std::vector<VectorXd>& y_d,
    const std::vector<VectorXd>& u_ini,
    const std::vector<VectorXd>& y_ini,
    const std::vector<VectorXd>& target,
    std::function<VectorXd(const VectorXd&)> input_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6);
