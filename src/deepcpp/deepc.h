#pragma once
#include <Eigen/Dense>
#include <functional>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

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
std::vector<VectorXd> deePC(
    const std::vector<VectorXd>& u_d,
    const std::vector<VectorXd>& y_d,
    const std::vector<VectorXd>& u_ini,
    const std::vector<VectorXd>& y_ini,
    const std::vector<VectorXd>& r,
    const MatrixXd &Q,
    const MatrixXd &R,
    std::function<VectorXd(const VectorXd&)> control_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6);

std::vector<VectorXd> deePC(
    const std::vector<VectorXd>& u_d,
    const std::vector<VectorXd>& y_d,
    const std::vector<VectorXd>& u_ini,
    const std::vector<VectorXd>& y_ini,
    const std::vector<VectorXd>& target_output,
    std::function<VectorXd(const VectorXd&)> control_constrain_fkt = nullptr,
    int max_pgm_iterations = 300,
    double pgm_tolerance = 1e-6);
