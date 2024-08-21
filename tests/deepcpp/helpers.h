#pragma once
#include <Eigen/Dense>
#include <type_traits>
#include <initializer_list>
#include <vector>

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

inline MatrixXd Matrix(std::initializer_list<double> row1, std::initializer_list<double> row2, std::initializer_list<double> row3, std::initializer_list<double> row4)
{
    MatrixXd mat(4, row1.size());
    mat.row(0) = VectorXd::Map(row1.begin(), row1.size());
    mat.row(1) = VectorXd::Map(row2.begin(), row2.size());
    mat.row(2) = VectorXd::Map(row3.begin(), row3.size());
    mat.row(3) = VectorXd::Map(row4.begin(), row4.size());
    return mat;
}

inline MatrixXd Matrix(std::initializer_list<double> row1, std::initializer_list<double> row2, std::initializer_list<double> row3, std::initializer_list<double> row4, std::initializer_list<double> row5)
{
    MatrixXd mat(5, row1.size());
    mat.row(0) = VectorXd::Map(row1.begin(), row1.size());
    mat.row(1) = VectorXd::Map(row2.begin(), row2.size());
    mat.row(2) = VectorXd::Map(row3.begin(), row3.size());
    mat.row(3) = VectorXd::Map(row4.begin(), row4.size());
    mat.row(4) = VectorXd::Map(row5.begin(), row5.size());
    return mat;
}

inline MatrixXd Matrix(std::initializer_list<double> row1, std::initializer_list<double> row2, std::initializer_list<double> row3, std::initializer_list<double> row4, std::initializer_list<double> row5, std::initializer_list<double> row6)
{
    MatrixXd mat(6, row1.size());
    mat.row(0) = VectorXd::Map(row1.begin(), row1.size());
    mat.row(1) = VectorXd::Map(row2.begin(), row2.size());
    mat.row(2) = VectorXd::Map(row3.begin(), row3.size());
    mat.row(3) = VectorXd::Map(row4.begin(), row4.size());
    mat.row(4) = VectorXd::Map(row5.begin(), row5.size());
    mat.row(5) = VectorXd::Map(row6.begin(), row6.size());
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

inline std::vector<VectorXd> Vectors(std::initializer_list<double> vec)
{
    return {VectorXd::Map(vec.begin(), vec.size())};
}

inline std::vector<VectorXd> Vectors(std::initializer_list<double> vec1, std::initializer_list<double> vec2)
{
    return {VectorXd::Map(vec1.begin(), vec1.size()), VectorXd::Map(vec2.begin(), vec2.size())};
}

inline std::vector<VectorXd> Vectors(std::initializer_list<double> vec1, std::initializer_list<double> vec2, std::initializer_list<double> vec3)
{
    return {VectorXd::Map(vec1.begin(), vec1.size()), VectorXd::Map(vec2.begin(), vec2.size()), VectorXd::Map(vec3.begin(), vec3.size())};
}

template <typename... Args>
std::vector<VectorXd> Vectors(Args... args)
{
    static_assert((std::is_arithmetic_v<Args> && ...), "All arguments must be numeric types");

    std::vector<VectorXd> vecs;
    vecs.reserve(sizeof...(args));
    ((vecs.push_back(VectorXd::Constant(1, args)), ...)); // Fold expression to unpack arguments
    return vecs;
}
