#pragma once
#include "finite_queue.h"
#include <Eigen/Dense>
#include <functional>
#include <vector>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Controller
{
    int T_ini;
    int target_size;
    FiniteQueue u_ini, y_ini;
    MatrixXd Q, R;
    std::function<VectorXd(const VectorXd &)> control_constrain_fkt;
    int max_pgm_iterations;
    double pgm_tolerance;
    int input_dims;
    int output_dims;
    MatrixXd M_x, M_u, G;

public:
    Controller(
        const std::vector<VectorXd> &u_d,
        const std::vector<VectorXd> &y_d,
        int T_ini,
        int target_size,
        MatrixXd Q,
        MatrixXd R,
        std::function<VectorXd(const VectorXd &)> control_constrain_fkt = nullptr,
        int max_pgm_iterations = 300,
        double pgm_tolerance = 1e-6);
    Controller(
        const std::vector<VectorXd> &u_d,
        const std::vector<VectorXd> &y_d,
        int T_ini,
        int target_size,
        std::function<VectorXd(const VectorXd &)> control_constrain_fkt = nullptr,
        int max_pgm_iterations = 300,
        double pgm_tolerance = 1e-6);

    bool is_initialized() const;
    void update(VectorXd u, VectorXd y);
    void clear();
    std::vector<VectorXd> apply(const std::vector<VectorXd> &);
    std::vector<VectorXd> apply(const VectorXd &);
};
