#include "lti.h"
#include <cassert>

bool DiscreteLTI::is_controllable() const
{
    // A system is controllable iff the controllability matrix has full rank
    // The controllability matrix is defined as
    // Q = [B, A @ B, A^2 @ B, ..., A^(n-1) @ B]
    // where n is the number of states
    int n = A.rows();
    int m = B.cols();
    MatrixXd Q(n, n * m);
    MatrixXd A_pow_i = MatrixXd::Identity(n, n);

    // Fill the controllability matrix
    Q.block(0, 0, n, m) = B;
    for (int i = 1; i < n; ++i) {
        A_pow_i *= A;
        Q.block(0, i * m, n, m) = A_pow_i * B;
    }
    return Q.fullPivLu().rank() == n;
}

bool DiscreteLTI::is_observable() const
{
    // A system is observable iff the observability matrix has full rank
    // The observability matrix is defined as
    // Q = [C, C @ A, C @ A^2, ..., C @ A^(n-1)]
    // where n is the number of states
    int n = A.rows();
    int m = C.rows();
    MatrixXd Q(n * m, n);
    MatrixXd A_pow_i = MatrixXd::Identity(n, n);

    // Fill the observability matrix
    Q.block(0, 0, C.rows(), n) = C;
    for (int i = 1; i < n; ++i) {
        A_pow_i *= A;
        Q.block(i * C.rows(), 0, C.rows(), n) = C * A_pow_i;
    }
    return Q.fullPivLu().rank() == n;
}

bool DiscreteLTI::is_stable() const
{
    // A system is stable iff all eigenvalues of the state matrix are inside the unit circle
    auto s = A.eigenvalues();
    return s.array().abs().maxCoeff() < 1;
}

VectorXd DiscreteLTI::apply(double u)
{
    return apply(VectorXd::Constant(1, u));
}

#include <iostream>
VectorXd DiscreteLTI::apply(const VectorXd& u)
{
    assert(u.size() == B.cols());
    assert(u.size() == D.cols());

    // Debug: Print current state x and input u
    std::cout << "LTI Apply - Current State x: " << x.transpose() << std::endl;
    std::cout << "LTI Apply - Input u: " << u.transpose() << std::endl;

    x = A * x + B * u;
    VectorXd y = C * x + D * u;

    // Debug: Print new state x and output y
    std::cout << "LTI Apply - New State x: " << x.transpose() << std::endl;
    std::cout << "LTI Apply - Output y: " << y.transpose() << std::endl;


    return y;
}

std::vector<VectorXd> DiscreteLTI::apply_multiple(const std::vector<double>& u)
{
    std::vector<VectorXd> u_vec;
    u_vec.reserve(u.size());
    for (double ui : u)
        u_vec.push_back(VectorXd::Constant(1, ui));
    return apply_multiple(u_vec);
}

std::vector<VectorXd> DiscreteLTI::apply_multiple(const std::vector<VectorXd>& u)
{
    std::vector<VectorXd> y;
    y.reserve(u.size());
    for (const auto& ui : u)
        y.push_back(apply(ui));
    return y;
}


void DiscreteLTI::printMatrices() 
{
    std::cout << "Matrix A:\n" << A << "\n";
    std::cout << "Matrix B:\n" << B << "\n";
    std::cout << "Matrix C:\n" << C << "\n";
    std::cout << "Matrix D:\n" << D << "\n";
    std::cout << "State vector x:\n" << x << "\n";
}

VectorXd RandomNoiseDiscreteLTI::apply(const VectorXd& u)
{
    VectorXd y = DiscreteLTI::apply(u);
    for (int i = 0; i < y.size(); i++)
        y[i] += dist(gen);
    return y;
}
