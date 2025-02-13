#include "gtest/gtest.h"
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
DiscreteLTI create_1D_in_1D_out_LTI()
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
DiscreteLTI create_2D_in_3D_out_LTI()
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

std::tuple<std::vector<VectorXd>, std::vector<VectorXd>> gather_offline_data(DiscreteLTI& system, int samples)
{
    std::vector<VectorXd> u_d;
    for (int i = 0; i < samples; ++i)
        u_d.push_back(VectorXd::Random(system.input_dim()));
    std::vector<VectorXd> y_d = system.apply_multiple(u_d);
    return {u_d, y_d};
}

void expect_near(const VectorXd& value, const VectorXd& expected, double abs_error)
{
    // Equal dimensions
    EXPECT_EQ(value.size(), expected.size());
    const int dim = value.size();

    // Near values
    for (int i = 0; i < dim; ++i)
        EXPECT_NEAR(value(i), expected(i), abs_error);
}

// Warm up the controller until it is initialized.
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
VectorXd control_system(
    Controller& controller,
    DiscreteLTI& system,
    const std::vector<VectorXd>& target,
    int time_steps,
    const std::vector<VectorXd>& offset = {})
{
    VectorXd u, y;
    for (int i = 0; i < time_steps; ++i)
    {
        u = controller.apply(target, offset).front();
        y = system.apply(u);
        controller.update(u, y);
    }
    return y;
}

class Test_1D_in_1D_out_LTI : public ::testing::Test
{
protected:
    DiscreteLTI system = create_1D_in_1D_out_LTI();
    std::vector<VectorXd> u_d, y_d, u_ini, y_ini;
    int T_ini = 20;
    std::vector<VectorXd> target = std::vector<VectorXd>(2, Vector(3));

    void SetUp() override
    {
        std::tie(u_d, y_d) = gather_offline_data(system, 1'000);
    }
};

TEST_F(Test_1D_in_1D_out_LTI, Unconstrained)
{
    Controller controller{u_d, y_d, T_ini, target.size()};
    warm_up_controller(controller, system, Vector(1));
    VectorXd y = control_system(controller, system, target, 2 * T_ini);
    expect_near(y, target[0], 1e-5);
}

TEST_F(Test_1D_in_1D_out_LTI, Constrained)
{
    Controller controller{u_d, y_d, T_ini, target.size(),
                          [](const VectorXd &u) { return clamp(u, 0, 25); }};
    warm_up_controller(controller, system, Vector(1));
    VectorXd y = control_system(controller, system, target, 2 * T_ini);
    expect_near(y, target[0], 1e-5);
}

TEST_F(Test_1D_in_1D_out_LTI, Offset)
{
    Controller controller{u_d, y_d, T_ini, target.size(), /*R*/0.001};
    warm_up_controller(controller, system, Vector(1));
    VectorXd y = control_system(controller, system, target, T_ini, {Vector(10), Vector(10)});
    expect_near(y, target[0], 1e-5);
}


class Test_2D_in_3D_out_LTI : public ::testing::Test
{
protected:
    DiscreteLTI system = create_2D_in_3D_out_LTI();
    std::vector<VectorXd> u_d, y_d, u_ini, y_ini;
    int T_ini = 20;
    std::vector<VectorXd> target = Vectors({0.19, 0.92, 0.24});

    void SetUp() override
    {
        std::tie(u_d, y_d) = gather_offline_data(system, 1'000);
    }
};

TEST_F(Test_2D_in_3D_out_LTI, Unconstrained)
{
    Controller controller{u_d, y_d, T_ini, target.size()};
    warm_up_controller(controller, system, Vector(1, 1));
    VectorXd y = control_system(controller, system, target, 2 * T_ini);
    expect_near(y, target[0], 0.05);
}

TEST_F(Test_2D_in_3D_out_LTI, Constrained)
{
    Controller controller{u_d, y_d, T_ini, target.size(),
                          [](const VectorXd &u) { return clamp(u, -15, 15); }};
    warm_up_controller(controller, system, Vector(1, 1));
    VectorXd y = control_system(controller, system, target, 2 * T_ini);
    expect_near(y, target[0], 0.05);
}

TEST_F(Test_2D_in_3D_out_LTI, Offset)
{
    Controller controller{u_d, y_d, T_ini, target.size(), /*R*/0.001};
    warm_up_controller(controller, system, Vector(1, 1));
    VectorXd y = control_system(controller, system, target, T_ini, {Vector(10, 10), Vector(10, 10), Vector(10, 10)});
    expect_near(y, target[0], 0.05);
}
