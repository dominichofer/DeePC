#include "algorithm.h"
#include "controller.h"
#include "lti.h"
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <chrono>

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main()
{
    // Define the system
    MatrixXd A(2, 2);
    MatrixXd B(2, 1);
    MatrixXd C(1, 2);
    MatrixXd D(1, 1);
    A << 1.9154297228892199, -0.9159698592594919, 1.0, 0.0;
    B << 0.0024407501942859677, 0.0;
    C << 1.0, 0.0;
    D << 0.0;
    DiscreteLTI system(A, B, C, D, VectorXd::Zero(2));

    // Define constraints for the input
    double min_input = 0;
    double max_input = 5;
    auto input_constrain_fkt = [min_input, max_input](const VectorXd &u) {
        return clamp(u, min_input, max_input);
    };

    // Gather offline data
    int N = 100'000;
    // by defining an input sequence
    std::vector<VectorXd> u_d(N);
    for (int i = 0; i < N; i++)
    {
        u_d[i] = VectorXd::Random(system.input_dim());
        u_d[i] = clamp(u_d[i], min_input, max_input);
    }
    // and applying it to the system
    std::vector<VectorXd> y_d = system.apply_multiple(u_d);

    // Define how many steps the controller should look back
    // to grasp the current state of the system
    int T_ini = 10;

    // Define how many steps the constroller should look forward
    int r_len = 10;

    // Define the controller
    auto start = std::chrono::high_resolution_clock::now();
    Controller controller(u_d, y_d, T_ini, r_len, input_constrain_fkt);
    auto stop = std::chrono::high_resolution_clock::now();

    // Reset the system
    // to separate the offline data from the online data
    system.set_state(VectorXd::Zero(2));

    // Warm up the controller
    while (!controller.is_initialized())
    {
        VectorXd u = VectorXd::Constant(system.input_dim(), 0);
        VectorXd y = system.apply(u);
        controller.update(y, u);
    }

    // Simulate the system
    std::vector<VectorXd> R;
    for (int i = 0; i < 20; i++)
        R.push_back(VectorXd::Constant(system.output_dim(), 0));
    for (int i = 0; i < 200; i++)
        R.push_back(VectorXd::Constant(system.output_dim(), 10));
    for (int i = 0; i < 100; i++)
        R.push_back(VectorXd::Constant(system.output_dim(), 7));
    for (int i = 0; i < R.size() - r_len; i++)
    {
        std::vector<VectorXd> r(R.begin() + i, R.begin() + i + r_len);
        VectorXd u = controller.apply(r)[0];
        VectorXd y = system.apply(u);
        controller.update(u, y);
        std::cout << "r: " << r[0] << " u: " << u << " y: " << y << std::endl;
    }
    std::cout << "Controller initialization time: " << std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count() << "ms" << std::endl;
    return 0;
}