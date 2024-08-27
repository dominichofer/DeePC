#include "gtest/gtest.h"
#include "deepc.h"
#include "controller.h"
#include "algorithm.h"
#include "lti.h"
#include "helpers.h"
#include <Eigen/Dense>
#include <cassert>
#include <tuple>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// LTI system with 1D input and 1D output
DiscreteLTI lti_1D_input_1D_output()
{
    DiscreteLTI system(
        Matrix({0.9, -0.2}, {0.7, 0.1}),
        Matrix({0.1}, {0}),
        Matrix({1, 0}),
        Matrix({0.1}),
        Vector(1, 1));
    assert(system.input_dim() == 1);
    assert(system.output_dim() == 1);
    assert(system.is_controllable());
    assert(system.is_observable());
    assert(system.is_stable());
    return system;
}

// LTI system with 3D input and 2D output
DiscreteLTI lti_2D_input_3D_output()
{
    DiscreteLTI system(
        Matrix({0.5, 0.1, 0}, {0.1, 0.5, 0.1}, {0, 0.1, 0.5}),
        Matrix({0.1, 0}, {0.1, 0.5}, {0, 0.1}),
        Matrix({1, 0, 0}, {0, 1, 1}, {0, 0, 1}),
        Matrix({0, 0}, {0, 0}, {0, 0}),
        Vector(0, 0, 0));
    assert(system.input_dim() == 2);
    assert(system.output_dim() == 3);
    assert(system.is_controllable());
    assert(system.is_observable());
    assert(system.is_stable());
    return system;
}

std::tuple<std::vector<VectorXd>, std::vector<VectorXd>> gather_offline_data(DiscreteLTI &system)
{
    const int samples = 1'000;
    std::vector<VectorXd> u_d(samples, VectorXd::Zero(system.input_dim()));
    for (int i = 0; i < system.input_dim(); ++i)
    {
        std::vector<double> chirp = linear_chirp(0, samples / 2, samples, 0.1 * i);
        for (int j = 0; j < samples; ++j)
            u_d[j](i) = chirp[j];
    }
    std::vector<VectorXd> y_d = system.apply_multiple(u_d);
    return {u_d, y_d};
}

void expect_near(const std::vector<VectorXd>& value, const std::vector<VectorXd>& expected, double abs_error)
{
    // Equal sizes
    EXPECT_EQ(value.size(), expected.size());
    const int size = value.size();

    // Equal dimensions
    for (int i = 0; i < size; ++i)
        EXPECT_EQ(value[i].size(), expected[i].size());
    const int dim = value[0].size();

    // Near values
    for (int i = 0; i < size; ++i)
        for (int j = 0; j < dim; ++j)
            EXPECT_NEAR(value[i](j), expected[i](j), abs_error);
}

class Test_deePC_1D_input_1D_output : public ::testing::Test
{
protected:
    DiscreteLTI system;
    std::vector<VectorXd> u_d, y_d;
    std::vector<VectorXd> u_ini, y_ini;
    std::vector<VectorXd> target;

    void SetUp() override
    {
        system = lti_1D_input_1D_output();

        // Offline data
        std::tie(u_d, y_d) = gather_offline_data(system);

        // Initial conditions
        u_ini = std::vector<VectorXd>(20, Vector(1));
        y_ini = system.apply_multiple(u_ini);

        // Reference trajectory
        target = std::vector<VectorXd>(2, Vector(3));
    }
};

TEST_F(Test_deePC_1D_input_1D_output, Unconstrained)
{
    std::vector<VectorXd> u_star = deePC(u_d, y_d, u_ini, y_ini, target);

    std::vector<VectorXd> y_star = system.apply_multiple(u_star);
    expect_near(y_star, target, 1e-5);
}

TEST_F(Test_deePC_1D_input_1D_output, Constrained)
{
    std::vector<VectorXd> u_star = deePC(
        u_d, y_d, u_ini, y_ini, target,
        [](const VectorXd &u)
        {
            return clamp(u, -15, 15);
        });

    std::vector<VectorXd> y_star = system.apply_multiple(u_star);
    expect_near(y_star, target, 1e-5);
}

class Test_deePC_2D_input_3D_output : public ::testing::Test
{
protected:
    DiscreteLTI system;
    std::vector<VectorXd> u_d, y_d;
    std::vector<VectorXd> u_ini, y_ini;
    std::vector<VectorXd> target;

    void SetUp() override
    {
        system = lti_2D_input_3D_output();

        // Offline data
        std::tie(u_d, y_d) = gather_offline_data(system);

        // Initial conditions
        u_ini = std::vector<VectorXd>(20, Vector(1, 1));
        y_ini = system.apply_multiple(u_ini);

        // Reference trajectory
        target = Vectors({0.35, 1.1, 0.35});
    }
};

TEST_F(Test_deePC_2D_input_3D_output, Unconstrained)
{
    std::vector<VectorXd> u_star = deePC(u_d, y_d, u_ini, y_ini, target);

    std::vector<VectorXd> y_star = system.apply_multiple(u_star);
    expect_near(y_star, target, 0.05);
}

TEST_F(Test_deePC_2D_input_3D_output, Constrained)
{
    std::vector<VectorXd> u_star = deePC(
        u_d, y_d, u_ini, y_ini, target,
        [](const VectorXd &u)
        {
            return clamp(u, -15, 15);
        });

    std::vector<VectorXd> y_star = system.apply_multiple(u_star);
    expect_near(y_star, target, 0.05);
}

TEST(Test_deePC_simple_system, 1D_input_1D_output)
{
    std::vector<VectorXd> u_star = deePC(
        Vectors(1, 2, 3, 4, -5, 6, 7, 8, 9, 10),
        Vectors(1, 2, 3, 4, -5, 6, 7, 8, 9, 10),
        Vectors(2),
        Vectors(2),
        Vectors(4));

    expect_near(u_star, Vectors(4), 1e-5);
}

TEST(Test_deePC_simple_system, 1D_input_1D_output_3_targets)
{
    std::vector<VectorXd> u_star = deePC(
        Vectors(1, 2, 3, 4, -5, 6, 7, 8, 9, 10),
        Vectors(1, 2, 3, 4, -5, 6, 7, 8, 9, 10),
        Vectors(2),
        Vectors(2),
        Vectors(4, 4, 4));

    expect_near(u_star, Vectors(4, 4, 4), 1e-5);
}

TEST(Test_deePC_simple_system, 2D_input_3D_output)
{
    std::vector<VectorXd> u_d, y_d, u_ini, y_ini, u, target;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            u_d.push_back(Vector(i, j));
    for (const VectorXd &u : u_d)
        y_d.push_back(Vector(u(0), u(1), u(0) + u(1)));
    u_ini = Vectors({2, 2});
    for (const VectorXd &u : u_ini)
        y_ini.push_back(Vector(u(0), u(1), u(0) + u(1)));
    u = Vectors({1, 3});
    for (const VectorXd &u : u)
        target.push_back(Vector(u(0), u(1), u(0) + u(1)));

    auto u_star = deePC(u_d, y_d, u_ini, y_ini, target);

    expect_near(u_star, u, 1e-5);
}

// Warm up the controller until it is initialized
void warm_up_controller(Controller& controller, DiscreteLTI& system, const VectorXd& u)
{
    while (!controller.is_initialized())
    {
        VectorXd y = system.apply(u);
        controller.update(u, y);
    }
}

// Control the system for a given number of time steps.
// Returns the output of the system after the last time step.
VectorXd control_system(Controller& controller, DiscreteLTI& system, const std::vector<VectorXd>& target, int time_steps)
{
    VectorXd u, y;
    for (int i = 0; i < time_steps; ++i)
    {
        u = controller.apply(target).front();
        y = system.apply(u);
        controller.update(u, y);
    }
    return y;
}

TEST(Test_Controller, unconstrained_2D_LTI)
{
    DiscreteLTI system = lti_1D_input_1D_output();
    auto [u_d, y_d] = gather_offline_data(system);
    int T_ini = 20;
    std::vector<VectorXd> target = Vectors({10}, {10}, {10});

    Controller controller(u_d, y_d, T_ini, target.size());
    warm_up_controller(controller, system, Vector(1));
    VectorXd y = control_system(controller, system, target, 2 * T_ini);
}

TEST(Test_Controller, constrained_2D_LTI)
{
    DiscreteLTI system = lti_1D_input_1D_output();
    auto [u_d, y_d] = gather_offline_data(system);
    int T_ini = 20;
    std::vector<VectorXd> target = Vectors({10}, {10}, {10});

    auto input_constrain_fkt = [](const VectorXd &u) { return clamp(u, 0, 25); };
    Controller controller(u_d, y_d, T_ini, target.size(), input_constrain_fkt);
    warm_up_controller(controller, system, Vector(1));
    VectorXd y = control_system(controller, system, target, 2 * T_ini);
}