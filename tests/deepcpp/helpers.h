#pragma once
#include <Eigen/Dense>
#include <type_traits>
#include <initializer_list>

using Eigen::MatrixXd;
using Eigen::VectorXd;

inline MatrixXd Matrix(std::initializer_list<double> row)
{
    MatrixXd mat(1, row.size());
    mat.row(0) = VectorXd::Map(row.begin(), row.size());
    return mat;
}

inline MatrixXd Matrix(std::initializer_list<double> row1, std::initializer_list<double> row2)
{
    MatrixXd mat(2, row1.size());
    mat.row(0) = VectorXd::Map(row1.begin(), row1.size());
    mat.row(1) = VectorXd::Map(row2.begin(), row2.size());
    return mat;
}

inline MatrixXd Matrix(std::initializer_list<double> row1, std::initializer_list<double> row2, std::initializer_list<double> row3)
{
    MatrixXd mat(3, row1.size());
    mat.row(0) = VectorXd::Map(row1.begin(), row1.size());
    mat.row(1) = VectorXd::Map(row2.begin(), row2.size());
    mat.row(2) = VectorXd::Map(row3.begin(), row3.size());
    return mat;
}

template <typename... Args>
VectorXd Vector(Args... args)
{
    static_assert((std::is_arithmetic_v<Args> && ...), "All arguments must be numeric types");

    VectorXd vec(sizeof...(args));
    int index = 0;
    ((vec[index++] = args), ...); // Fold expression to unpack arguments
    return vec;
}
