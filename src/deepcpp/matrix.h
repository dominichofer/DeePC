#pragma once
#include "vector.h"
#include <string>
#include <vector>

// Interface for matrix
class IMatrix
{
public:
    virtual double operator()(int i, int j) const = 0;
    virtual int rows() const = 0;
    virtual int cols() const = 0;
    virtual ~IMatrix() = default;

    bool operator==(const IMatrix &) const;
};

class DiagonalMatrix : public IMatrix
{
    std::vector<double> data;

public:
    DiagonalMatrix(std::vector<double> data) : data(std::move(data)) {}
    inline double operator()(int i, int j) const override { return i == j ? data[i] : 0; }
    inline int rows() const override { return data.size(); }
    inline int cols() const override { return data.size(); }
};

class TransposedMatrix : public IMatrix
{
    const IMatrix &matrix;

public:
    TransposedMatrix(const IMatrix &matrix) noexcept : matrix(matrix) {}
    inline double operator()(int i, int j) const override { return matrix(j, i); }
    inline int rows() const override { return matrix.cols(); }
    inline int cols() const override { return matrix.rows(); }
};

class VStackMatrix : public IMatrix
{
    const IMatrix &a;
    const IMatrix &b;

public:
    VStackMatrix(const IMatrix &a, const IMatrix &b) noexcept;
    double operator()(int i, int j) const override;
    inline int rows() const override { return a.rows() + b.rows(); }
    inline int cols() const override { return a.cols(); }
};

class HankelMatrix : public IMatrix
{
    const std::vector<double> &data;
    int cols_;

public:
    HankelMatrix(const std::vector<double> &data, int rows) noexcept : data(data), cols_(data.size() - rows + 1) {}
    inline double operator()(int i, int j) const override { return data[i + j]; }
    inline int rows() const override { return data.size() - cols_ + 1; }
    inline int cols() const override { return cols_; }
};

class SubMatrix : public IMatrix
{
    const IMatrix &matrix;
    int row_begin, rows_, col_begin, cols_;

public:
    SubMatrix(const IMatrix &, int row_begin, int row_end, int col_begin, int col_end) noexcept;
    inline double operator()(int i, int j) const override { return matrix(row_begin + i, col_begin + j); }
    inline int rows() const override { return rows_; }
    inline int cols() const override { return cols_; }
};

class DenseMatrix : public IMatrix
{
    std::vector<double> data;
    int cols_;

public:
    DenseMatrix() noexcept = default;
    DenseMatrix(std::vector<double> data, int cols) noexcept;
    explicit DenseMatrix(const IMatrix &) noexcept;
    static DenseMatrix Zeros(int rows, int cols) noexcept;

    inline double operator()(int i, int j) const override { return data[i * cols_ + j]; }
    inline double &operator()(int i, int j) { return data[i * cols_ + j]; }
    inline int rows() const override { return data.size() / cols_; }
    inline int cols() const override { return cols_; }
};

std::string to_string(const IMatrix &);

DiagonalMatrix IdentityMatrix(int size);
DiagonalMatrix ZeroMatrix(int size);

TransposedMatrix transposed(const IMatrix &);

VStackMatrix vstack(const IMatrix &, const IMatrix &);
VStackMatrix vstack(const IMatrix &, const IMatrix &, const IMatrix &);

std::vector<double> operator*(const IMatrix &, const std::vector<double> &);
DenseMatrix operator*(const IMatrix &, const IMatrix &);
DenseMatrix operator+(const IMatrix &, const IMatrix &);
DenseMatrix operator-(const IMatrix &, const IMatrix &);
